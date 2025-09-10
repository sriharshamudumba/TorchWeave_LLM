import asyncio
import time
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
import torch
from .model_runtime import ModelRuntime

@dataclass
class Request:
    """Represents a single generation request"""
    id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    
    # Runtime state
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    past_key_values: Optional[tuple] = None
    generated_tokens: List[int] = field(default_factory=list)
    steps: int = 0
    start_time: float = field(default_factory=time.time)
    ttft_time: Optional[float] = None
    finished: bool = False
    
    # Communication
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

class ContinuousBatchScheduler:
    """Continuous batching scheduler for LLM inference"""
    
    def __init__(self, 
                 model_runtime: ModelRuntime,
                 max_batch_size: int = 16,
                 schedule_tick_ms: int = 15):
        """Initialize the scheduler
        
        Args:
            model_runtime: Model runtime instance
            max_batch_size: Maximum batch size
            schedule_tick_ms: Scheduling tick interval in milliseconds
        """
        self.model_runtime = model_runtime
        self.max_batch_size = max_batch_size
        self.schedule_tick_ms = schedule_tick_ms
        
        # Request management
        self.active_requests: List[Request] = []
        self.pending_requests: List[Request] = []
        self.request_counter = 0
        
        # Scheduler state
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        print(f"[Scheduler] Initialized with max_batch_size={max_batch_size}, tick_ms={schedule_tick_ms}")
    
    async def start(self):
        """Start the scheduler"""
        if self.running:
            return
            
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        print(f"[Scheduler] Started")
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        print(f"[Scheduler] Stopped")
    
    async def submit_request(self,
                           prompt: str,
                           max_new_tokens: int = 128,
                           temperature: float = 0.7,
                           top_k: int = 0,
                           top_p: float = 0.9,
                           seed: int = 42) -> str:
        """Submit a new generation request
        
        Returns:
            Request ID for tracking
        """
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        request = Request(
            id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed
        )
        
        # Tokenize the prompt
        request.input_ids = self.model_runtime.tokenize(prompt)
        request.attention_mask = torch.ones_like(request.input_ids)
        
        self.pending_requests.append(request)
        
        print(f"[Scheduler] Submitted request {request_id}")
        return request_id
    
    async def stream_tokens(self, request_id: str) -> AsyncIterator[Dict]:
        """Stream tokens for a specific request
        
        Args:
            request_id: Request ID to stream
            
        Yields:
            Dictionary with event information
        """
        # Find the request
        request = None
        for req in self.pending_requests + self.active_requests:
            if req.id == request_id:
                request = req
                break
        
        if not request:
            yield {"type": "error", "error": "Request not found"}
            return
        
        try:
            while not request.finished:
                try:
                    # Wait for next token with timeout
                    token_data = await asyncio.wait_for(
                        request.token_queue.get(), 
                        timeout=30.0
                    )
                    
                    if token_data["type"] == "ttft":
                        yield {"type": "ttft", "time": token_data["time"]}
                    elif token_data["type"] == "token":
                        yield {"type": "token", "token": token_data["token"]}
                    elif token_data["type"] == "done":
                        yield {"type": "done"}
                        break
                        
                except asyncio.TimeoutError:
                    yield {"type": "error", "error": "Request timeout"}
                    break
                    
        except Exception as e:
            yield {"type": "error", "error": str(e)}
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Process pending and active requests
                await self._process_requests()
                
                # Sleep for the tick interval
                await asyncio.sleep(self.schedule_tick_ms / 1000.0)
                
            except Exception as e:
                print(f"[Scheduler] Error in scheduler loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_requests(self):
        """Process pending and active requests"""
        # Move pending requests to active if we have space
        while (len(self.active_requests) < self.max_batch_size and 
               len(self.pending_requests) > 0):
            request = self.pending_requests.pop(0)
            self.active_requests.append(request)
            print(f"[Scheduler] Activated request {request.id}")
        
        if not self.active_requests:
            return
        
        # Prepare batch for new requests (prefill phase)
        new_requests = [req for req in self.active_requests if req.past_key_values is None]
        if new_requests:
            await self._process_prefill_batch(new_requests)
        
        # Process decode step for all active requests
        if self.active_requests:
            await self._process_decode_batch()
        
        # Remove finished requests
        finished = [req for req in self.active_requests if req.finished]
        for req in finished:
            self.active_requests.remove(req)
            await req.token_queue.put({"type": "done"})
            req.done_event.set()
            print(f"[Scheduler] Finished request {req.id}")
    
    async def _process_prefill_batch(self, requests: List[Request]):
        """Process prefill phase for new requests"""
        if not requests:
            return
        
        # Pad input sequences to same length
        max_len = max(req.input_ids.shape[1] for req in requests)
        
        batch_input_ids = []
        batch_attention_mask = []
        
        for req in requests:
            seq_len = req.input_ids.shape[1]
            if seq_len < max_len:
                # Pad on the left
                pad_len = max_len - seq_len
                padded_input = torch.cat([
                    torch.full((1, pad_len), self.model_runtime.tokenizer.pad_token_id, 
                              dtype=req.input_ids.dtype, device=req.input_ids.device),
                    req.input_ids
                ], dim=1)
                padded_attention = torch.cat([
                    torch.zeros((1, pad_len), dtype=req.attention_mask.dtype, 
                               device=req.attention_mask.device),
                    req.attention_mask
                ], dim=1)
            else:
                padded_input = req.input_ids
                padded_attention = req.attention_mask
            
            batch_input_ids.append(padded_input)
            batch_attention_mask.append(padded_attention)
        
        # Stack into batch
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
        
        # Forward pass
        outputs = self.model_runtime.forward_batch(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # Update requests with past key values
        for i, req in enumerate(requests):
            # Extract this request's past_key_values
            req.past_key_values = tuple(
                (layer_past[0][i:i+1], layer_past[1][i:i+1]) 
                for layer_past in outputs["past_key_values"]
            )
            
            # Update attention mask for future decode steps
            req.attention_mask = batch_attention_mask[i:i+1]
    
    async def _process_decode_batch(self):
        """Process decode step for all active requests"""
        if not self.active_requests:
            return
        
        # Prepare batch inputs (all requests get their last token)
        batch_input_ids = []
        batch_attention_masks = []
        batch_past_key_values = []
        
        for req in self.active_requests:
            if req.generated_tokens:
                # Use last generated token
                last_token = torch.tensor([[req.generated_tokens[-1]]], 
                                        device=self.model_runtime.device)
            else:
                # Use last token from original input
                last_token = req.input_ids[:, -1:]
            
            batch_input_ids.append(last_token)
            
            # Extend attention mask
            extended_attention = torch.cat([
                req.attention_mask,
                torch.ones((1, 1), device=req.attention_mask.device)
            ], dim=1)
            batch_attention_masks.append(extended_attention)
            req.attention_mask = extended_attention
            
            batch_past_key_values.append(req.past_key_values)
        
        # Stack batch inputs
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_masks, dim=0)
        
        # Reorganize past_key_values for batching
        if batch_past_key_values[0] is not None:
            # Combine past_key_values from all requests
            combined_past_key_values = []
            num_layers = len(batch_past_key_values[0])
            
            for layer_idx in range(num_layers):
                layer_keys = torch.cat([
                    req_past[layer_idx][0] for req_past in batch_past_key_values
                ], dim=0)
                layer_values = torch.cat([
                    req_past[layer_idx][1] for req_past in batch_past_key_values  
                ], dim=0)
                combined_past_key_values.append((layer_keys, layer_values))
            
            batch_past_key_values = tuple(combined_past_key_values)
        else:
            batch_past_key_values = None
        
        # Forward pass
        outputs = self.model_runtime.forward_batch(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            past_key_values=batch_past_key_values
        )
        
        # Sample tokens
        temperatures = [req.temperature for req in self.active_requests]
        top_ks = [req.top_k for req in self.active_requests]
        top_ps = [req.top_p for req in self.active_requests]
        seeds = [req.seed + req.steps for req in self.active_requests]  # Vary seed by step
        
        sampled_tokens = self.model_runtime.sample_tokens(
            outputs["logits"], temperatures, top_ks, top_ps, seeds
        )
        
        # Update requests with results
        for i, req in enumerate(self.active_requests):
            token_id = sampled_tokens[i]
            req.generated_tokens.append(token_id)
            req.steps += 1
            
            # Update past_key_values for this request
            req.past_key_values = tuple(
                (layer_past[0][i:i+1], layer_past[1][i:i+1])
                for layer_past in outputs["past_key_values"]
            )
            
            # Send TTFT event for first token
            if req.ttft_time is None:
                req.ttft_time = time.time()
                ttft_duration = req.ttft_time - req.start_time
                await req.token_queue.put({"type": "ttft", "time": ttft_duration})
            
            # Decode and send token
            token_text = self.model_runtime.decode_single(token_id)
            await req.token_queue.put({"type": "token", "token": token_text})
            
            # Check if finished
            if (token_id == self.model_runtime.get_eos_token_id() or 
                req.steps >= req.max_new_tokens):
                req.finished = True

