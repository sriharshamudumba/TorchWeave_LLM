import asyncio
import logging
import time
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: Optional[int]
    status: RequestStatus
    created_at: float
    tokens: List[str]
    ttft: Optional[float] = None  # Time to first token

class ContinuousBatchScheduler:
    """Continuous batching scheduler for LLM inference"""
    
    def __init__(self, model_runtime, max_batch_size: int = 16):
        self.model_runtime = model_runtime
        self.max_batch_size = max_batch_size
        self.requests: Dict[str, GenerationRequest] = {}
        self.pending_requests: List[str] = []
        self.active_batches: List[List[str]] = []
        self.running = False
        
        # Configuration - can be set via environment or parameters
        self.schedule_tick_ms = 15  # Fixed schedule tick time
        
        logger.info(f"[Scheduler] Initialized with max_batch_size={max_batch_size}")
    
    async def submit_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: Optional[int] = None
    ) -> str:
        """Submit a generation request"""
        request_id = str(uuid.uuid4())
        
        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            status=RequestStatus.PENDING,
            created_at=time.time(),
            tokens=[]
        )
        
        self.requests[request_id] = request
        self.pending_requests.append(request_id)
        
        logger.info(f"[Scheduler] Submitted request {request_id}")
        return request_id
    
    async def stream_tokens(self, request_id: str) -> AsyncGenerator[str, None]:
        """Stream generated tokens for a request"""
        if request_id not in self.requests:
            yield "data: [ERROR] Request not found\n"
            return
        
        request = self.requests[request_id]
        last_token_count = 0
        ttft_sent = False
        
        # Wait for processing to start or complete
        while request.status == RequestStatus.PENDING:
            await asyncio.sleep(0.01)
        
        # Send TTFT when first token is available
        if request.ttft is not None and not ttft_sent:
            yield f"event: ttft\ndata: {request.ttft:.3f}\n"
            ttft_sent = True
        
        # Stream tokens as they become available
        while request.status in [RequestStatus.PROCESSING, RequestStatus.PENDING]:
            current_token_count = len(request.tokens)
            
            # Yield new tokens
            for i in range(last_token_count, current_token_count):
                yield f"data: {request.tokens[i]}\n"
            
            last_token_count = current_token_count
            
            # Check if generation is complete
            if request.status == RequestStatus.COMPLETED:
                break
                
            await asyncio.sleep(0.01)
        
        # Send any remaining tokens
        current_token_count = len(request.tokens)
        for i in range(last_token_count, current_token_count):
            yield f"data: {request.tokens[i]}\n"
        
        # Send completion event
        yield "event: done\ndata:\n"
        
        # Clean up request
        if request_id in self.requests:
            del self.requests[request_id]
    
    async def run(self):
        """Main scheduler loop"""
        self.running = True
        logger.info("[Scheduler] Started")
        
        while self.running:
            try:
                await self._process_pending_requests()
                await self._process_active_batches()
                await asyncio.sleep(self.schedule_tick_ms / 1000.0)
            except Exception as e:
                logger.error(f"[Scheduler] Error in main loop: {e}")
                await asyncio.sleep(0.1)
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("[Scheduler] Stopped")
    
    async def _process_pending_requests(self):
        """Process pending requests and form new batches"""
        if not self.pending_requests:
            return
        
        # Form new batch
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        batch_request_ids = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        if batch_request_ids:
            # Mark requests as processing
            for req_id in batch_request_ids:
                if req_id in self.requests:
                    self.requests[req_id].status = RequestStatus.PROCESSING
            
            # Start batch processing
            asyncio.create_task(self._process_batch(batch_request_ids))
            logger.info(f"[Scheduler] Started batch with {len(batch_request_ids)} requests")
    
    async def _process_active_batches(self):
        """Monitor and manage active batches"""
        # This could include batch merging, rebalancing, etc.
        # For now, just log active batch count
        active_count = sum(1 for req in self.requests.values() 
                         if req.status == RequestStatus.PROCESSING)
        if active_count > 0:
            logger.debug(f"[Scheduler] {active_count} requests processing")
    
    async def _process_batch(self, request_ids: List[str]):
        """Process a batch of requests"""
        try:
            # Get requests
            requests = [self.requests[req_id] for req_id in request_ids if req_id in self.requests]
            if not requests:
                return
            
            logger.info(f"[Scheduler] Processing batch of {len(requests)} requests")
            
            # For each request in the batch, generate individually
            # In a real implementation, this would be done in parallel/batched
            for request in requests:
                try:
                    start_time = time.time()
                    
                    # Generate tokens using model runtime
                    generated_text = self.model_runtime.generate(
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        seed=request.seed
                    )
                    
                    # Record TTFT (approximated as generation start time)
                    request.ttft = time.time() - start_time
                    
                    # Tokenize the generated text for streaming
                    # This is a simplified version - real implementation would stream during generation
                    tokens = generated_text.split()
                    request.tokens = tokens
                    request.status = RequestStatus.COMPLETED
                    
                    logger.info(f"[Scheduler] Completed request {request.request_id}")
                    
                except Exception as e:
                    logger.error(f"[Scheduler] Failed to process request {request.request_id}: {e}")
                    request.status = RequestStatus.FAILED
            
        except Exception as e:
            logger.error(f"[Scheduler] Batch processing failed: {e}")
            # Mark all requests as failed
            for req_id in request_ids:
                if req_id in self.requests:
                    self.requests[req_id].status = RequestStatus.FAILED
