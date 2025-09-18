import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import logging
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class ModelRuntime:
    def __init__(self):
        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.generation_configs: Dict[str, GenerationConfig] = {}
        self.device = self._get_device()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._lock = threading.RLock()
        logger.info(f"ModelRuntime initialized with device: {self.device}")

    def _get_device(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def load_model(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            if model_id in self.models:
                return {"status": "success", "message": "Model already loaded"}
        
        logger.info(f"Loading model {model_id} on {self.device}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None: 
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True if self.device != "cpu" else False
            )
            if self.device == "cpu": model = model.to("cpu")
            model.eval()

            try:
                generation_config = GenerationConfig.from_pretrained(model_id, pad_token_id=tokenizer.eos_token_id)
            except:
                generation_config = GenerationConfig(
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=50
                )

            with self._lock:
                self.models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                self.generation_configs[model_id] = generation_config
            
            return {"status": "success", "message": f"Model {model_id} loaded successfully"}
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    def _get_model_for_generation(self, model_id: Optional[str]):
        with self._lock:
            if not self.models: raise ValueError("No models are loaded.")
            
            model_key = model_id or next(iter(self.models))
            if model_key not in self.models:
                raise ValueError(f"Model '{model_key}' not found.")

            return self.models[model_key], self.tokenizers[model_key], self.generation_configs[model_key]
    
    def generate_batch(self, prompts: List[str], model_id: Optional[str] = None, **kwargs) -> List[str]:
        """
        Generates text for a batch of prompts in a single model pass.
        """
        try:
            model, tokenizer, generation_config = self._get_model_for_generation(model_id)
            
            # Batch tokenization with padding
            tokenizer.padding_side = "left" # For decoder-only models
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            
            # Update generation config with kwargs
            gen_config = generation_config.to_dict()
            gen_config.update({k: v for k, v in kwargs.items() if v is not None})
            
            # Force sampling for text generation
            gen_config["do_sample"] = True
            gen_config["pad_token_id"] = tokenizer.eos_token_id
            gen_config["eos_token_id"] = tokenizer.eos_token_id
            
            # Ensure reasonable generation length
            if "max_new_tokens" not in gen_config and "max_length" not in gen_config:
                gen_config["max_new_tokens"] = 50

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **{k: v for k, v in gen_config.items() 
                       if k in ['max_new_tokens', 'max_length', 'temperature', 'top_p', 'top_k', 
                               'do_sample', 'repetition_penalty', 'pad_token_id', 'eos_token_id']}
                )

            # Decode each sequence in the batch, skipping the prompt tokens
            input_length = inputs.input_ids.shape[1]
            decoded_texts = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
            
            return decoded_texts
        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
            return [f"Error: {e}" for _ in prompts]

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            if model_id not in self.models: raise ValueError(f"Model {model_id} not found")
            del self.models[model_id]
            del self.tokenizers[model_id]
            del self.generation_configs[model_id]
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return {"status": "success", "message": f"Model {model_id} unloaded"}

    def list_models(self) -> Dict[str, Any]:
        with self._lock:
            models_list = []
            for model_id, model in self.models.items():
                params = sum(p.numel() for p in model.parameters())
                models_list.append({"model_id": model_id, "device": str(model.device), "parameters": params})
        return {"loaded_models": models_list}

    def cleanup(self):
        with self._lock:
            self.models.clear(); self.tokenizers.clear(); self.generation_configs.clear()
        self.executor.shutdown(wait=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
