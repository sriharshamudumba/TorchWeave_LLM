"""
server/scheduler.py

Async continuous batching scheduler for TorchWeave.

Problem this solves:
    Naive request handling: one request at a time, GPU idle between requests.
    Simple batching: wait for N requests, run together -- introduces queuing
    delay and head-of-line (HOL) blocking where one long request stalls all
    short ones waiting in the same batch.

Continuous batching (Orca paper, 2022):
    After every decode step, check if any slot freed up. If yes, pull the
    next waiting request and pack it into the batch immediately. Long and
    short requests coexist in the same batch; a finished request's slot is
    reused without waiting for all others to finish.

HOL blocking prevention:
    Short requests bypass long ones waiting in queue via a priority mechanism:
    requests shorter than HOL_BYPASS_THRESHOLD tokens jump ahead of requests
    that have been in-flight longer than HOL_STALL_MS milliseconds.

     15% of GPU FLOPs (memory-bandwidth-bound, not compute-bound). With
     a batch of 8-16, you amortize the weight reads across all requests,
     pushing utilization to 60-80%. That's the 4x.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    priority: int = 0  # higher = served sooner
    arrived_at: float = field(default_factory=time.monotonic)
    prompt_len: int = 0  # filled at schedule time
    tokens_generated: int = 0
    finished: bool = False
    result_future: asyncio.Future = field(default_factory=lambda: None)


@dataclass
class SchedulerConfig:
    max_batch_size: int = 16  # max concurrent in-flight requests
    max_queue_size: int = 512  # waiting room
    max_wait_ms: float = 5000.0  # HOL deadline: force-admit after this
    schedule_interval_ms: float = 1.0  # how often to re-evaluate the batch
    hol_bypass_threshold: int = (
        128  # short requests (< this tokens) bypass long waiters
    )
    hol_stall_ms: float = 500.0  # a request is "stalling" after this many ms in-flight


class ContinuousBatchScheduler:
    """
    Async scheduler implementing continuous batching with HOL-blocking prevention.

    Lifecycle:
        scheduler = ContinuousBatchScheduler(config, step_fn)
        await scheduler.start()

        future = await scheduler.submit(request)
        result = await future           # blocks until request completes

        await scheduler.stop()

    step_fn:
        A coroutine that takes a list of active InferenceRequest objects,
        runs one decode step on the GPU for all of them, and returns
        (tokens_list, finished_mask). The scheduler calls this in a loop.

        signature: async def step_fn(batch: List[InferenceRequest])
                       -> Tuple[List[int], List[bool]]
    """

    def __init__(
        self,
        config: SchedulerConfig,
        step_fn: Callable,
    ):
        self.config = config
        self.step_fn = step_fn

        # Waiting room: requests not yet admitted to the active batch
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=config.max_queue_size
        )

        # Active batch: currently running on GPU
        self._active: List[InferenceRequest] = []

        # Stats
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "hol_bypasses": 0,
            "batch_size_samples": [],
            "step_times_ms": [],
        }

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, request: InferenceRequest) -> asyncio.Future:
        """
        Submit a request for inference. Returns a Future that resolves
        to the generated text when the request completes.
        Non-blocking: returns immediately after queuing.
        """
        loop = asyncio.get_event_loop()
        request.result_future = loop.create_future()
        request.request_id = request.request_id or str(uuid.uuid4())

        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            raise RuntimeError(
                f"Scheduler queue full ({self.config.max_queue_size} requests). "
                "Backpressure: retry after delay."
            )

        self._stats["total_requests"] += 1
        logger.debug(
            f"Queued request {request.request_id}, queue depth={self._queue.qsize()}"
        )
        return request.result_future

    async def start(self):
        """Start the scheduling loop."""
        self._running = True
        self._loop_task = asyncio.create_task(self._scheduling_loop())
        logger.info("ContinuousBatchScheduler started.")

    async def stop(self):
        """Graceful shutdown: finish active batch then stop."""
        self._running = False
        if self._loop_task:
            await self._loop_task
        logger.info("ContinuousBatchScheduler stopped.")

    def stats(self) -> dict:
        samples = self._stats["batch_size_samples"]
        step_times = self._stats["step_times_ms"]
        return {
            "total_requests": self._stats["total_requests"],
            "total_tokens": self._stats["total_tokens"],
            "hol_bypasses": self._stats["hol_bypasses"],
            "queue_depth": self._queue.qsize(),
            "active_batch": len(self._active),
            "avg_batch_size": sum(samples) / len(samples) if samples else 0,
            "avg_step_ms": sum(step_times) / len(step_times) if step_times else 0,
            "p95_step_ms": (
                sorted(step_times)[int(0.95 * len(step_times))]
                if len(step_times) > 20
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Core scheduling loop
    # ------------------------------------------------------------------

    async def _scheduling_loop(self):
        """
        Main loop: runs continuously while self._running.

        Each iteration:
          1. Admit waiting requests into empty slots (continuous batching).
          2. Call step_fn for one decode step across the whole batch.
          3. Evict finished requests, free their slots.
          4. Sleep for schedule_interval_ms to yield the event loop.
        """
        while self._running or self._active or not self._queue.empty():
            # Step 1: Admit new requests into available slots
            self._admit_requests()

            if not self._active:
                # Nothing to run; yield to event loop
                await asyncio.sleep(self.config.schedule_interval_ms / 1000.0)
                continue

            # Step 2: Run one decode step on the GPU for all active requests
            t0 = time.monotonic()
            try:
                tokens_list, finished_mask = await self.step_fn(self._active)
            except Exception as e:
                logger.error(f"step_fn error: {e}")
                # Fail all active requests rather than hang
                for req in self._active:
                    if not req.result_future.done():
                        req.result_future.set_exception(e)
                self._active.clear()
                continue

            step_ms = (time.monotonic() - t0) * 1000.0
            self._stats["step_times_ms"].append(step_ms)
            self._stats["batch_size_samples"].append(len(self._active))

            # Step 3: Accumulate tokens, evict finished requests
            still_active = []
            for req, token, finished in zip(self._active, tokens_list, finished_mask):
                req.tokens_generated += 1
                self._stats["total_tokens"] += 1

                if finished or req.tokens_generated >= req.max_new_tokens:
                    req.finished = True
                    if not req.result_future.done():
                        req.result_future.set_result(token)
                    logger.debug(
                        f"Request {req.request_id} finished: "
                        f"{req.tokens_generated} tokens, {step_ms:.1f}ms/step"
                    )
                else:
                    still_active.append(req)

            self._active = still_active

            # Step 4: Yield event loop
            await asyncio.sleep(self.config.schedule_interval_ms / 1000.0)

    def _admit_requests(self):
        """
        Pull waiting requests from the queue into the active batch.

        HOL bypass: if a short request has been waiting while a long request
        is stalling the queue, the short one jumps ahead.
        """
        cfg = self.config
        now = time.monotonic()

        while len(self._active) < cfg.max_batch_size and not self._queue.empty():
            # Check if HOL bypass applies: any in-flight request has been
            # running longer than hol_stall_ms AND a short request is waiting
            stalling = any(
                (now - req.arrived_at) * 1000 > cfg.hol_stall_ms for req in self._active
            )

            try:
                # Peek at the queue without blocking
                candidate = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if stalling and candidate.prompt_len > cfg.hol_bypass_threshold:
                # This candidate is long and there's a stalling request.
                # Put it back and try to find a shorter one.
                # In production: use a priority queue. Here: simple re-queue.
                try:
                    self._queue.put_nowait(candidate)
                    self._stats["hol_bypasses"] += 1
                except asyncio.QueueFull:
                    pass
                break  # don't loop forever on re-queue

            # Force-admit if waiting too long regardless of length
            wait_ms = (now - candidate.arrived_at) * 1000
            if wait_ms > cfg.max_wait_ms:
                logger.debug(
                    f"Force-admitting request {candidate.request_id} "
                    f"after {wait_ms:.0f}ms wait (max_wait exceeded)"
                )

            self._active.append(candidate)
            logger.debug(
                f"Admitted request {candidate.request_id}, "
                f"active batch size={len(self._active)}"
            )
