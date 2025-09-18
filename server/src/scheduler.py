import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple
import uuid

from model_runtime import ModelRuntime

logger = logging.getLogger(__name__)

class ContinuousBatchingScheduler:
    def __init__(self, runtime: ModelRuntime, max_batch_size: int = 16, batch_timeout: float = 0.02):
        self.runtime = runtime
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_queue = asyncio.Queue()
        self.running = False
        logger.info(f"[Scheduler] Initialized with max batch size {max_batch_size} and timeout {batch_timeout*1000:.0f}ms.")

    async def submit_request(self, **kwargs) -> Any:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        # Store the submission time with the request
        await self.request_queue.put(((request_id, time.monotonic(), kwargs), future))
        return await future

    async def _batch_generator(self):
        while self.running:
            batch_start_time = time.monotonic()
            batch = []
            while len(batch) < self.max_batch_size and (time.monotonic() - batch_start_time) < self.batch_timeout:
                try:
                    timeout = self.batch_timeout - (time.monotonic() - batch_start_time)
                    item = await asyncio.wait_for(self.request_queue.get(), timeout=max(0, timeout))
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            if batch:
                yield batch

    async def run(self):
        self.running = True
        logger.info("[Scheduler] Started.")
        loop = asyncio.get_event_loop()

        async for batch in self._batch_generator():
            try:
                requests, futures = zip(*batch)
                
                prompts_by_model: Dict[str, List[Tuple[int, str]]] = {}
                params_by_model: Dict[str, Dict[str, Any]] = {}

                for i, (req_id, req_time, req_kwargs) in enumerate(requests):
                    # Extract model_id and prompt from request
                    model_id = req_kwargs.get("model_id")
                    prompt = req_kwargs.get("prompt")
                    
                    if not prompt:
                        logger.error(f"Request {req_id} missing prompt")
                        continue
                    
                    # Remove prompt and model_id from kwargs to avoid passing them as extra args
                    generation_params = {k: v for k, v in req_kwargs.items() 
                                       if k not in ['prompt', 'model_id']}
                    
                    if model_id not in prompts_by_model:
                        prompts_by_model[model_id] = []
                        params_by_model[model_id] = generation_params
                    prompts_by_model[model_id].append((i, prompt))

                full_results = [None] * len(requests)
                for model_id, indexed_prompts in prompts_by_model.items():
                    indices, prompts = zip(*indexed_prompts)
                    params = params_by_model[model_id]

                    # Fixed: Pass arguments in correct order - prompts, model_id as positional, then **params
                    batch_results = await loop.run_in_executor(
                        None, 
                        lambda: self.runtime.generate_batch(list(prompts), model_id, **params)
                    )
                    
                    for i, result in zip(indices, batch_results):
                        full_results[i] = result
                
                # Fulfill futures with a dictionary containing the text and the total time
                for i, future in enumerate(futures):
                    _, submission_time, _ = requests[i]
                    total_time = time.monotonic() - submission_time
                    
                    if full_results[i] is not None:
                        future.set_result({
                            "text": full_results[i],
                            "total_time": total_time
                        })
                    else:
                        future.set_exception(Exception("No result generated"))

            except Exception as e:
                logger.error(f"[Scheduler] Batch processing failed: {e}", exc_info=True)
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)

    async def stop(self):
        self.running = False
        logger.info("[Scheduler] Stopped.")
