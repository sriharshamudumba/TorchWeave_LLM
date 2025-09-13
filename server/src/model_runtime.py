import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import logging
import os
import json
from pathlib import Path
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class ModelRuntime:
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize ModelRuntime with optional model directory
        
        Args:
            model_dir: Directory containing pre-loaded model files
        """
        self.models = {}
        self.tokenizers = {}
        self.generation_configs = {}
        self.model_dir = model_dir or "/artifacts/model"
        self.device = self._get_device()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        
        logger.info(f"ModelRuntime initialized with device: {self.device}")
        
        # Auto-load model if directory exists and contains model files
        if self.model_dir and os.path.exists(self.model_dir):
            self._auto_load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _auto_load_model(self):
        """Auto-load model from the model directory if it exists"""
        try:
            # Check if model files exist
            config_path = os.path.join(self.model_dir, "config.json")
            if not os.path.exists(config_path):
                logger.info(f"No config.json found in {self.model_dir}, skipping auto-load")
                return
            
            # Try to determine model name from config or use default
            model_id = "loaded_model"
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_id = config.get("_name_or_path", "loaded_model")
                    if "/" in model_id:
                        model_id = model_id.split("/")[-1]  # Use just the model name part
            except Exception as e:
                logger.warning(f"Could not read model config: {e}")
            
            logger.info(f"Auto-loading model from {self.model_dir} as '{model_id}'")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=None if self.device == "cpu" else "auto",
                trust_remote_code=True
            )
            
            # Move to device if needed
            if self.device == "cpu":
                model = model.to("cpu")
            
            # Load generation config if available
            generation_config = None
            try:
                generation_config = GenerationConfig.from_pretrained(self.model_dir, local_files_only=True)
            except Exception:
                generation_config = GenerationConfig(
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Store components
            with self._lock:
                self.models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                self.generation_configs[model_id] = generation_config
            
            logger.info(f"Model '{model_id}' auto-loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to auto-load model: {str(e)}")
            logger.exception("Auto-load error details:")
    
    def load_model(self, model_config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Load a model based on configuration
        
        Args:
            model_config: Dictionary with model configuration or model ID string
            
        Returns:
            Dictionary with loading results
        """
        try:
            # Handle different input types
            if isinstance(model_config, str):
                model_id = model_config
                source = "huggingface"
                device = self.device
            elif isinstance(model_config, dict):
                model_id = model_config.get("model_id")
                source = model_config.get("source", "huggingface")
                device = model_config.get("device", self.device)
            else:
                # Handle Pydantic model or object with attributes
                model_id = getattr(model_config, "model_id", None)
                source = getattr(model_config, "source", "huggingface")
                device = getattr(model_config, "device", self.device)
            
            if not model_id:
                raise ValueError("model_id is required in model configuration")
            
            # Check if already loaded
            if model_id in self.models:
                logger.info(f"Model {model_id} already loaded")
                return {
                    "status": "success",
                    "model_id": model_id,
                    "device": str(self.models[model_id].device),
                    "message": "Model already loaded"
                }
            
            logger.info(f"Loading model {model_id} from {source} on {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine torch dtype
            torch_dtype = torch.float32 if device == "cpu" else torch.float16
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to specified device if needed
            if device == "cpu":
                model = model.to("cpu")
            
            # Create generation config
            try:
                generation_config = GenerationConfig.from_pretrained(model_id)
            except Exception:
                generation_config = GenerationConfig(
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Store components
            with self._lock:
                self.models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                self.generation_configs[model_id] = generation_config
            
            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            
            logger.info(f"Model {model_id} loaded successfully")
            
            return {
                "status": "success",
                "model_id": model_id,
                "device": str(model.device),
                "parameters": param_count,
                "torch_dtype": str(torch_dtype),
                "source": source
            }
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")
    
    def _get_model_for_generation(self, model_id: Optional[str] = None):
        """Get model, tokenizer, and generation config for text generation"""
        with self._lock:
            # If no model_id specified, try to use any available model
            if model_id is None:
                if "loaded_model" in self.models:
                    model_id = "loaded_model"
                elif self.models:
                    model_id = list(self.models.keys())[0]
                else:
                    raise ValueError("No models loaded")
            
            # Check if model exists
            if model_id not in self.models:
                available_models = list(self.models.keys())
                if available_models:
                    model_id = available_models[0]
                    logger.warning(f"Requested model not found, using '{model_id}' instead")
                else:
                    raise ValueError("No models loaded")
            
            return (
                self.models[model_id],
                self.tokenizers[model_id],
                self.generation_configs[model_id],
                model_id
            )
    
    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using the specified model
        
        Args:
            prompt: Input text prompt
            model_id: Model to use (uses auto-loaded model if None)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        try:
            model, tokenizer, generation_config, used_model_id = self._get_model_for_generation(model_id)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move inputs to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Update generation config with parameters
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )
            
            # Decode generated tokens (excluding input)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming output
        
        Args:
            prompt: Input text prompt
            model_id: Model to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to sample
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text tokens as they are produced
        """
        def _generate():
            try:
                model, tokenizer, generation_config, used_model_id = self._get_model_for_generation(model_id)
                
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]
                
                # Generation config
                gen_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Generate with streaming
                with torch.no_grad():
                    for outputs in model.generate(
                        **inputs,
                        generation_config=gen_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                        streamer=None  # Custom streaming would go here
                    ):
                        if hasattr(outputs, 'sequences'):
                            new_tokens = outputs.sequences[0][input_length:]
                            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                            yield text
                        
            except Exception as e:
                logger.error(f"Streaming generation error: {str(e)}")
                yield f"Error: {str(e)}"
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        for token in await loop.run_in_executor(self.executor, lambda: list(_generate())):
            yield token
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """
        Unload a model from memory
        
        Args:
            model_id: ID of model to unload
            
        Returns:
            Status dictionary
        """
        try:
            with self._lock:
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")
                
                # Remove from storage
                del self.models[model_id]
                del self.tokenizers[model_id]
                del self.generation_configs[model_id]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_id} unloaded successfully")
            
            return {
                "status": "success",
                "message": f"Model {model_id} unloaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            raise RuntimeError(f"Failed to unload model {model_id}: {str(e)}")
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all loaded models with their information
        
        Returns:
            Dictionary containing model information
        """
        with self._lock:
            models_info = {}
            for model_id, model in self.models.items():
                try:
                    param_count = sum(p.numel() for p in model.parameters())
                    models_info[model_id] = {
                        "model_id": model_id,
                        "device": str(model.device),
                        "parameters": param_count,
                        "dtype": str(model.dtype),
                        "loaded": True
                    }
                except Exception as e:
                    models_info[model_id] = {
                        "model_id": model_id,
                        "loaded": True,
                        "error": str(e)
                    }
        
        return {
            "models": models_info,
            "count": len(models_info),
            "device": self.device
        }
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model or the default model
        
        Args:
            model_id: Model ID to get info for
            
        Returns:
            Model information dictionary
        """
        try:
            model, tokenizer, generation_config, used_model_id = self._get_model_for_generation(model_id)
            
            return {
                "model_id": used_model_id,
                "vocab_size": tokenizer.vocab_size,
                "device": str(model.device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "dtype": str(model.dtype),
                "eos_token": tokenizer.eos_token,
                "pad_token": tokenizer.pad_token,
                "max_length": getattr(tokenizer, 'model_max_length', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise RuntimeError(f"Failed to get model info: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            with self._lock:
                self.models.clear()
                self.tokenizers.clear()
                self.generation_configs.clear()
            
            self.executor.shutdown(wait=True)
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("ModelRuntime cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Global runtime instance - will be initialized by server.py
runtime = None
