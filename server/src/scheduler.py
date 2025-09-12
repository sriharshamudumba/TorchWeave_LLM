import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import asyncio
import logging

class ContinuousBatchScheduler:
    def __init__(self, model_runtime, max_batch_size=16, tick_ms=15):
        self.model_runtime = model_runtime
        self.max_batch_size = max_batch_size
        self.tick_ms = tick_ms
        self.pending_requests = []
        self.active_requests = {}
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    async def schedule_loop(self):
        """Main scheduler loop with proper error handling"""
        self.running = True
        self.logger.info(f"[Scheduler] Started with max_batch_size={self.max_batch_size}, tick_ms={self.tick_ms}")
        
        while self.running:
            try:
                await self.process_batch()
                await asyncio.sleep(self.tick_ms / 1000.0)
            except Exception as e:
                self.logger.error(f"[Scheduler] Error in scheduler loop: {e}")
                # Continue running instead of crashing
                await asyncio.sleep(0.1)  # Brief pause on error
                
    async def process_batch(self):
        """Process a batch of requests with proper tensor alignment"""
        if not self.pending_requests:
            return
            
        # Take up to max_batch_size requests
        batch_requests = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        if not batch_requests:
            return
            
        try:
            # Group requests by similar sequence length for better batching
            batch_groups = self._group_by_length(batch_requests)
            
            for group in batch_groups:
                await self._process_group(group)
                
        except Exception as e:
            self.logger.error(f"[Scheduler] Error processing batch: {e}")
            # Return failed requests to pending queue
            self.pending_requests.extend(batch_requests)
            
    def _group_by_length(self, requests):
        """Group requests by similar input lengths to avoid tensor mismatch"""
        groups = []
        current_group = []
        current_length = None
        
        # Sort by input length first
        requests.sort(key=lambda r: len(r['input_ids']))
        
        for req in requests:
            req_length = len(req['input_ids'])
            
            # Start new group if length differs significantly or group is full
            if (current_length is None or 
                abs(req_length - current_length) > 100 or  # Allow 100 token difference
                len(current_group) >= 4):  # Smaller groups for stability
                
                if current_group:
                    groups.append(current_group)
                current_group = [req]
                current_length = req_length
            else:
                current_group.append(req)
                
        if current_group:
            groups.append(current_group)
            
        return groups
        
    async def _process_group(self, requests):
        """Process a group of requests with similar lengths"""
        if len(requests) == 1:
            # Single request - no batching needed
            await self._process_single_request(requests[0])
            return
            
        try:
            # Prepare batch tensors with proper padding
            batch_data = self._prepare_batch_tensors(requests)
            
            if batch_data is None:
                # Fall back to individual processing on tensor prep failure
                for req in requests:
                    await self._process_single_request(req)
                return
                
            # Run batched inference
            with torch.no_grad():
                batch_outputs = self.model_runtime.generate_batch(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    max_new_tokens=batch_data['max_new_tokens'],
                    temperature=batch_data['temperature']
                )
                
            # Distribute results back to individual requests
            for i, req in enumerate(requests):
                if i < len(batch_outputs):
                    await self._send_response(req, batch_outputs[i])
                else:
                    await self._send_error(req, "Batch processing failed")
                    
        except Exception as e:
            self.logger.error(f"[Scheduler] Group processing failed: {e}")
            # Fall back to individual processing
            for req in requests:
                await self._process_single_request(req)
                
    def _prepare_batch_tensors(self, requests):
        """Prepare properly padded tensors for batching"""
        try:
            input_ids_list = []
            attention_mask_list = []
            max_length = 0
            
            # Find maximum sequence length
            for req in requests:
                seq_len = len(req['input_ids'])
                max_length = max(max_length, seq_len)
                
            # Pad all sequences to max_length
            for req in requests:
                input_ids = req['input_ids']
                
                # Pad sequence
                if len(input_ids) < max_length:
                    # Pad with tokenizer pad token (usually 0)
                    pad_length = max_length - len(input_ids)
                    padded_input_ids = input_ids + [0] * pad_length
                    attention_mask = [1] * len(input_ids) + [0] * pad_length
                else:
                    padded_input_ids = input_ids[:max_length]
                    attention_mask = [1] * max_length
                    
                input_ids_list.append(padded_input_ids)
                attention_mask_list.append(attention_mask)
                
            # Convert to tensors
            batch_input_ids = torch.tensor(input_ids_list, device=self.model_runtime.device)
            batch_attention_mask = torch.tensor(attention_mask_list, device=self.model_runtime.device)
            
            # Use parameters from first request (could be made more sophisticated)
            first_req = requests[0]
            
            return {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
                'max_new_tokens': first_req.get('max_new_tokens', 50),
                'temperature': first_req.get('temperature', 0.7)
            }
            
        except Exception as e:
            self.logger.error(f"[Scheduler] Tensor preparation failed: {e}")
            return None
            
    async def _process_single_request(self, request):
        """Process a single request (fallback)"""
        try:
            with torch.no_grad():
                output = self.model_runtime.generate(
                    input_ids=torch.tensor([request['input_ids']], device=self.model_runtime.device),
                    max_new_tokens=request.get('max_new_tokens', 50),
                    temperature=request.get('temperature', 0.7)
                )
                
            await self._send_response(request, output)
            
        except Exception as e:
            self.logger.error(f"[Scheduler] Single request failed: {e}")
            await self._send_error(request, str(e))
            
    async def _send_response(self, request, output):
        """Send successful response"""
        try:
            response_queue = request['response_queue']
            await response_queue.put({"status": "success", "output": output})
        except Exception as e:
            self.logger.error(f"[Scheduler] Failed to send response: {e}")
            
    async def _send_error(self, request, error_msg):
        """Send error response"""
        try:
            response_queue = request['response_queue']
            await response_queue.put({"status": "error", "error": error_msg})
        except Exception as e:
            self.logger.error(f"[Scheduler] Failed to send error: {e}")
            
    async def add_request(self, request):
        """Add request to pending queue"""
        self.pending_requests.append(request)
        
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        self.logger.info("[Scheduler] Stopped")


# Additional method needed in model_runtime.py for batch processing
class ModelRuntime:
    # ... existing code ...
    
    def generate_batch(self, input_ids, attention_mask, max_new_tokens=50, temperature=0.7):
        """Generate for a batch of inputs"""
        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode each output in the batch
            batch_results = []
            for i in range(outputs.shape[0]):
                # Remove input tokens to get only generated text
                generated_tokens = outputs[i][input_ids.shape[1]:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_results.append(decoded)
                
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise
