"""
Enhanced Model Runtime with Per-Request KV-Cache Management
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
from kv_cache_manager import KVCacheManager, RequestCache

logger = logging.getLogger(__name__)


class ModelRuntimeWithKVCache:
    """
    Enhanced model runtime with explicit per-request KV-cache management.
    
    Key features:
    - Per-request cache allocation and tracking
    - Dynamic cache updates during generation
    - Independent request completion and cache freeing
    - Memory-efficient cache pooling and reuse
    """
    
    def __init__(self):
        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.generation_configs: Dict[str, GenerationConfig] = {}
        self.kv_cache_managers: Dict[str, KVCacheManager] = {}
        
        self.device = self._get_device()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._lock = threading.RLock()
        
        logger.info(f"ModelRuntimeWithKVCache initialized on {self.device}")

    def _get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_id: str, max_batch_size: int = 16) -> Dict[str, Any]:
        """Load model and initialize KV-cache manager"""
        with self._lock:
            if model_id in self.models:
                return {"status": "success", "message": "Model already loaded"}
        
        logger.info(f"Loading model {model_id} on {self.device}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Load model
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True if self.device != "cpu" else False
            )
            
            if self.device == "cpu":
                model = model.to("cpu")
            model.eval()

            # Create generation config
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            except:
                generation_config = GenerationConfig(
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=50
                )

            # Initialize KV-cache manager
            model_config = model.config
            num_layers = model_config.num_hidden_layers
            num_heads = model_config.num_attention_heads
            head_dim = model_config.hidden_size // num_heads
            
            kv_cache_manager = KVCacheManager(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=2048,
                device=self.device,
                dtype=torch_dtype
            )

            with self._lock:
                self.models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                self.generation_configs[model_id] = generation_config
                self.kv_cache_managers[model_id] = kv_cache_manager
            
            logger.info(
                f"Model {model_id} loaded successfully with KV-cache manager: "
                f"{num_layers} layers, {num_heads} heads, dim={head_dim}"
            )
            
            return {
                "status": "success",
                "message": f"Model {model_id} loaded with per-request KV-cache",
                "config": {
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "max_batch_size": max_batch_size
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    def _get_model_components(self, model_id: Optional[str]):
        """Get model, tokenizer, config, and cache manager"""
        with self._lock:
            if not self.models:
                raise ValueError("No models are loaded")
            
            model_key = model_id or next(iter(self.models))
            if model_key not in self.models:
                raise ValueError(f"Model '{model_key}' not found")

            return (
                self.models[model_key],
                self.tokenizers[model_key],
                self.generation_configs[model_key],
                self.kv_cache_managers[model_key]
            )
    
    def generate_with_cache(
        self,
        request_id: str,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with explicit per-request KV-cache management.
        
        This method demonstrates:
        1. Cache allocation for new request
        2. Incremental cache updates during generation
        3. Cache retrieval for each decoding step
        4. Cache cleanup after completion
        
        Args:
            request_id: Unique identifier for this request
            prompt: Input text
            model_id: Model to use (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        
        Returns:
            Generated text and cache statistics
        """
        model, tokenizer, gen_config, cache_manager = self._get_model_components(model_id)
        
        try:
            # Step 1: Allocate cache for this request
            logger.debug(f"Allocating cache for request {request_id}")
            request_cache = cache_manager.allocate_request_cache(request_id)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            initial_seq_len = input_ids.shape[1]
            generated_tokens = []
            
            # Step 2: Generation loop with cache updates
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # Forward pass
                    if step == 0:
                        # First pass: process entire prompt
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True
                        )
                    else:
                        # Subsequent passes: only process new token
                        # Retrieve cached K,V for all layers
                        past_key_values = self._get_past_key_values(
                            cache_manager,
                            request_id
                        )
                        
                        outputs = model(
                            input_ids=input_ids[:, -1:],  # Only last token
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                    
                    # Step 3: Update cache with new K,V from this step
                    self._update_request_cache(
                        cache_manager,
                        request_id,
                        outputs.past_key_values,
                        step
                    )
                    
                    # Sample next token
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        logits = logits / temperature
                    
                    # Apply top-k
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus sampling)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1),
                            dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1,
                            sorted_indices,
                            sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Check for EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Append to input_ids for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=self.device)
                    ], dim=1)
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Get cache statistics
            cache_stats = cache_manager.get_stats()
            request_stats = {
                "request_id": request_id,
                "prompt_tokens": initial_seq_len,
                "generated_tokens": len(generated_tokens),
                "total_tokens": initial_seq_len + len(generated_tokens),
                "cache_stats": cache_stats
            }
            
            logger.info(
                f"Request {request_id} completed: "
                f"{len(generated_tokens)} tokens generated, "
                f"cache memory: {cache_stats['active_memory_mb']:.1f}MB"
            )
            
            return generated_text, request_stats
            
        except Exception as e:
            logger.error(f"Generation failed for request {request_id}: {e}", exc_info=True)
            # Ensure cache is freed even on error
            cache_manager.free_request_cache(request_id)
            raise
        
        finally:
            # Step 4: Free cache after request completion
            if request_id in cache_manager.request_caches:
                cache_manager.free_request_cache(request_id)
                logger.debug(f"Cache freed for request {request_id}")
    
    def _get_past_key_values(
        self,
        cache_manager: KVCacheManager,
        request_id: str
    ) -> Tuple:
        """
        Retrieve cached K,V tensors for all layers.
        Converts our cache format to HuggingFace's expected format.
        """
        request_cache = cache_manager.request_caches[request_id]
        past_key_values = []
        
        for layer_idx in range(len(request_cache.blocks)):
            keys, values = cache_manager.get_cache(request_id, layer_idx)
            # HuggingFace expects tuple of (key, value) per layer
            past_key_values.append((keys, values))
        
        return tuple(past_key_values)
    
    def _update_request_cache(
        self,
        cache_manager: KVCacheManager,
        request_id: str,
        past_key_values: Tuple,
        step: int
    ):
        """
        Update cache with new K,V tensors from model output.
        
        Args:
            cache_manager: Cache manager instance
            request_id: Request identifier
            past_key_values: Tuple of (key, value) per layer from model output
            step: Current generation step
        """
        if past_key_values is None:
            return
        
        for layer_idx, (keys, values) in enumerate(past_key_values):
            cache_manager.update_cache(
                request_id=request_id,
                layer_idx=layer_idx,
                new_keys=keys,
                new_values=values
            )
    
    def generate_batch_with_cache(
        self,
        requests: List[Dict[str, Any]],
        model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate for multiple requests using batched inference with per-request caching.
        
        This demonstrates continuous batching where:
        - Each request has independent cache
        - Requests can complete at different times
        - Cache is freed immediately upon completion
        
        Args:
            requests: List of request dicts with 'request_id' and 'prompt'
            model_id: Model to use
        
        Returns:
            List of results with generated text and stats
        """
        model, tokenizer, gen_config, cache_manager = self._get_model_components(model_id)
        
        results = []
        active_requests = {req['request_id']: req for req in requests}
        
        # Allocate cache for all requests
        for req in requests:
            cache_manager.allocate_request_cache(req['request_id'])
        
        try:
            # Process batch (simplified - in practice, use dynamic batching)
            for req in requests:
                request_id = req['request_id']
                prompt = req['prompt']
                
                # Generate with cache
                generated_text, stats = self.generate_with_cache(
                    request_id=request_id,
                    prompt=prompt,
                    model_id=model_id,
                    max_new_tokens=req.get('max_new_tokens', 50)
                )
                
                results.append({
                    'request_id': request_id,
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'stats': stats
                })
                
                # Cache automatically freed in generate_with_cache
        
        except Exception as e:
            logger.error(f"Batch generation failed: {e}", exc_info=True)
            # Cleanup any remaining caches
            for req_id in active_requests:
                if req_id in cache_manager.request_caches:
                    cache_manager.free_request_cache(req_id)
            raise
        
        return results

    def get_cache_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current cache manager statistics"""
        _, _, _, cache_manager = self._get_model_components(model_id)
        return cache_manager.get_stats()

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload model and clear all caches"""
        with self._lock:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            # Clear cache manager
            if model_id in self.kv_cache_managers:
                self.kv_cache_managers[model_id].clear_all()
                del self.kv_cache_managers[model_id]
            
            del self.models[model_id]
            del self.tokenizers[model_id]
            del self.generation_configs[model_id]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"status": "success", "message": f"Model {model_id} unloaded"}

    def cleanup(self):
        """Cleanup all resources"""
        with self._lock:
            for cache_manager in self.kv_cache_managers.values():
                cache_manager.clear_all()
            
            self.kv_cache_managers.clear()
            self.models.clear()
            self.tokenizers.clear()
            self.generation_configs.clear()
        
        self.executor.shutdown(wait=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ModelRuntimeWithKVCache cleaned up")
