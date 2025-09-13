from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Generator
import logging
import asyncio
import json
import time
from datetime import datetime
import psutil
import os
import sys
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    TextStreamer
)
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Manager", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for loaded models and tokenizers
loaded_models: Dict[str, dict] = {}
model_status: Dict[str, str] = {}

class ModelLoadRequest(BaseModel):
    model_id: str
    config: Optional[dict] = None

class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.9
    num_return_sequences: Optional[int] = 1

class StreamGenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.9

class BatchRequest(BaseModel):
    model_id: str
    prompts: List[str]
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7

@app.on_event("startup")
async def startup_event():
    logger.info("Model Manager with SSE Streaming started successfully")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

def get_model_info(model_id: str) -> dict:
    """Get model configuration info"""
    model_configs = {
        "microsoft/DialoGPT-small": {
            "type": "causal_lm",
            "tokenizer_class": "AutoTokenizer",
            "model_class": "AutoModelForCausalLM",
            "size": "117MB"
        },
        "microsoft/DialoGPT-medium": {
            "type": "causal_lm", 
            "tokenizer_class": "AutoTokenizer",
            "model_class": "AutoModelForCausalLM",
            "size": "345MB"
        },
        "gpt2": {
            "type": "causal_lm",
            "tokenizer_class": "AutoTokenizer", 
            "model_class": "AutoModelForCausalLM",
            "size": "548MB"
        },
        "distilgpt2": {
            "type": "causal_lm",
            "tokenizer_class": "AutoTokenizer",
            "model_class": "AutoModelForCausalLM", 
            "size": "353MB"
        },
        "facebook/blenderbot_small-90M": {
            "type": "seq2seq_lm",
            "tokenizer_class": "BlenderbotTokenizer",
            "model_class": "BlenderbotForConditionalGeneration",
            "size": "356MB"
        }
    }
    return model_configs.get(model_id, {
        "type": "causal_lm",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModelForCausalLM",
        "size": "Unknown"
    })

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "model-manager"}

@app.get("/models/available")
async def get_available_models():
    """Get list of available models that can be loaded"""
    available_models = [
        {
            "model_id": "microsoft/DialoGPT-small",
            "description": "Small conversational model - Fast inference",
            "size": "117MB",
            "type": "causal_lm",
            "capabilities": ["streaming", "batch", "chat"]
        },
        {
            "model_id": "microsoft/DialoGPT-medium", 
            "description": "Medium conversational model - Better quality",
            "size": "345MB",
            "type": "causal_lm",
            "capabilities": ["streaming", "batch", "chat"]
        },
        {
            "model_id": "gpt2",
            "description": "GPT-2 base model - Creative text generation",
            "size": "548MB",
            "type": "causal_lm",
            "capabilities": ["streaming", "batch", "generation"]
        },
        {
            "model_id": "distilgpt2",
            "description": "Distilled GPT-2 - Faster, smaller model",
            "size": "353MB", 
            "type": "causal_lm",
            "capabilities": ["streaming", "batch", "generation"]
        },
        {
            "model_id": "facebook/blenderbot_small-90M",
            "description": "Small BlenderBot - Conversational AI",
            "size": "356MB",
            "type": "seq2seq_lm",
            "capabilities": ["batch", "chat"]
        }
    ]
    return {"available_models": available_models}

@app.get("/models/loaded")
async def get_loaded_models():
    """Get list of currently loaded models"""
    models_list = []
    for model_id, model_info in loaded_models.items():
        models_list.append({
            "model_id": model_id,
            "status": model_status.get(model_id, "unknown"),
            "memory_usage": model_info.get("memory_usage", "Unknown"),
            "load_time": model_info.get("load_time", "Unknown"),
            "type": model_info.get("type", "Unknown"),
            "device": model_info.get("device", "Unknown")
        })
    return {"loaded_models": models_list}

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model for real inference"""
    model_id = request.model_id
    
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    
    # Check if model is already loaded
    if model_id in loaded_models:
        return {
            "status": "success",
            "message": f"Model {model_id} is already loaded",
            "model_id": model_id
        }
    
    try:
        model_status[model_id] = "loading"
        logger.info(f"Loading model: {model_id}")
        
        # Get model configuration
        model_config = get_model_info(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_id}")
        if model_config["tokenizer_class"] == "BlenderbotTokenizer":
            tokenizer = BlenderbotTokenizer.from_pretrained(model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model weights for {model_id}")
        if model_config["type"] == "seq2seq_lm":
            if model_config["model_class"] == "BlenderbotForConditionalGeneration":
                model = BlenderbotForConditionalGeneration.from_pretrained(model_id)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Move to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Get memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = f"{memory_info.percent}%"
        
        # Store model and tokenizer
        loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "model_id": model_id,
            "load_time": datetime.now().isoformat(),
            "memory_usage": memory_usage,
            "device": device,
            "type": model_config["type"],
            "config": request.config or {}
        }
        
        model_status[model_id] = "loaded"
        logger.info(f"Model {model_id} loaded successfully on {device}")
        
        return {
            "status": "success", 
            "message": f"Model {model_id} loaded successfully",
            "model_id": model_id,
            "memory_usage": memory_usage,
            "device": device,
            "type": model_config["type"]
        }
        
    except Exception as e:
        model_status[model_id] = "error"
        logger.error(f"Error loading model {model_id}: {str(e)}")
        # Clean up on error
        if model_id in loaded_models:
            del loaded_models[model_id]
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/models/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using a loaded model - Regular inference"""
    model_id = request.model_id
    
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        model_info = loaded_models[model_id]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        device = model_info["device"]
        
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[-1] + request.max_length,
                temperature=request.temperature,
                do_sample=request.do_sample,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if request.prompt in generated_text:
            generated_text = generated_text.replace(request.prompt, "").strip()
        
        processing_time = time.time() - start_time
        
        return {
            "generated_text": generated_text,
            "model_id": model_id,
            "prompt": request.prompt,
            "processing_time": processing_time,
            "inference_type": "standard",
            "parameters": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

def generate_stream(model, tokenizer, prompt: str, max_length: int = 100, 
                   temperature: float = 0.7, top_p: float = 0.9) -> Generator[str, None, None]:
    """Generator function for streaming text generation"""
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for i in range(max_length):
            # Generate next token
            outputs = model(inputs)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for end token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Decode and yield the token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # Add token to input for next iteration
            inputs = torch.cat([inputs, next_token], dim=-1)
            
            yield token_text

@app.post("/models/generate/stream")
async def stream_generate_text(request: StreamGenerateRequest):
    """Generate text with SSE streaming"""
    model_id = request.model_id
    
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    model_info = loaded_models[model_id]
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    def event_stream():
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting generation...'})}\n\n"
            
            start_time = time.time()
            full_response = ""
            
            # Generate tokens
            for token in generate_stream(model, tokenizer, request.prompt, 
                                       request.max_length, request.temperature, request.top_p):
                full_response += token
                
                # Send token event
                yield f"data: {json.dumps({'type': 'token', 'content': token, 'full_text': full_response})}\n\n"
                
                # Small delay to make streaming visible
                time.sleep(0.05)
            
            processing_time = time.time() - start_time
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_text': full_response, 'processing_time': processing_time, 'model_id': model_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.post("/models/generate/batch")
async def batch_generate_text(request: BatchRequest):
    """Generate text for multiple prompts in batch"""
    model_id = request.model_id
    
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    try:
        model_info = loaded_models[model_id]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        device = model_info["device"]
        
        start_time = time.time()
        results = []
        
        for i, prompt in enumerate(request.prompts):
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[-1] + request.max_length,
                    temperature=request.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "index": i
            })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "model_id": model_id,
            "batch_size": len(request.prompts),
            "processing_time": processing_time,
            "inference_type": "batch",
            "avg_time_per_prompt": processing_time / len(request.prompts)
        }
        
    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.delete("/models/unload/{model_id}")
async def unload_model(model_id: str):
    """Unload a specific model and free memory"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not found or not loaded")
    
    try:
        model_info = loaded_models[model_id]
        
        # Delete model and tokenizer to free memory
        if "model" in model_info:
            del model_info["model"]
        if "tokenizer" in model_info:
            del model_info["tokenizer"]
            
        del loaded_models[model_id]
        model_status[model_id] = "unloaded"
        
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
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.get("/models/status/{model_id}")
async def get_model_status(model_id: str):
    """Get status of a specific model"""
    if model_id not in loaded_models and model_id not in model_status:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = loaded_models.get(model_id, {})
    safe_info = {k: v for k, v in model_info.items() if k not in ["model", "tokenizer"]}
    
    return {
        "model_id": model_id,
        "status": model_status.get(model_id, "unknown"),
        "details": safe_info
    }

@app.get("/system/stats")
async def get_system_stats():
    """Get system resource statistics"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            }
        else:
            gpu_info = {"gpu_available": False}
        
        return {
            "memory": {
                "total": f"{memory.total / (1024**3):.2f} GB",
                "used": f"{memory.used / (1024**3):.2f} GB", 
                "available": f"{memory.available / (1024**3):.2f} GB",
                "percentage": f"{memory.percent}%"
            },
            "disk": {
                "total": f"{disk.total / (1024**3):.2f} GB",
                "used": f"{disk.used / (1024**3):.2f} GB",
                "free": f"{disk.free / (1024**3):.2f} GB"
            },
            "cpu": {"usage_percent": f"{cpu_percent}%"},
            "gpu": gpu_info,
            "loaded_models_count": len(loaded_models),
            "system": sys.platform
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return {"error": "Could not retrieve system stats", "loaded_models_count": len(loaded_models)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TorchWeave Model Manager",
        "version": "1.0.0",
        "status": "running",
        "features": ["SSE Streaming", "Batch Processing", "Real-time Inference"],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "available_models": "/models/available",
            "loaded_models": "/models/loaded",
            "load_model": "/models/load",
            "generate": "/models/generate",
            "stream_generate": "/models/generate/stream",
            "batch_generate": "/models/generate/batch",
            "system_stats": "/system/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
