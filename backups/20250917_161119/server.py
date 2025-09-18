import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scheduler import ContinuousBatchingScheduler
from model_runtime import ModelRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

runtime: Optional[ModelRuntime] = None
scheduler: Optional[ContinuousBatchingScheduler] = None

class GenerateRequest(BaseModel):
    model_id: Optional[str] = None
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.2

class ModelLoadRequest(BaseModel):
    model_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime, scheduler
    logger.info("Starting TorchWeave inference server...")
    try:
        runtime = ModelRuntime()
        scheduler = ContinuousBatchingScheduler(runtime)
        asyncio.create_task(scheduler.run())
        logger.info("Server with continuous batching scheduler initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down TorchWeave inference server...")
    if scheduler: 
        await scheduler.stop()
    if runtime: 
        runtime.cleanup()

app = FastAPI(title="TorchWeave LLM", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {
        "service": "TorchWeave Batched Inference Server",
        "version": "1.0.0", 
        "status": "running",
        "features": ["continuous_batching", "multi_model", "performance_metrics"],
        "port": 8000
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "torchweave-server", "port": 8000}

@app.get("/models/loaded")
async def get_loaded_models():
    if not runtime: 
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    return runtime.list_models()

@app.get("/models/available")
async def get_available_models():
    return {"available_models": [
        {"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "description": "Fast small chat model", "size": "1.1B"},
        {"model_id": "gpt2", "description": "GPT-2 base model", "size": "124M"},
        {"model_id": "distilgpt2", "description": "Distilled GPT-2", "size": "82M"},
        {"model_id": "microsoft/DialoGPT-small", "description": "Small conversational model", "size": "117M"}
    ]}

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    if not runtime: 
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, runtime.load_model, request.model_id)
        return {
            "success": True,
            "message": f"Model {request.model_id} loaded successfully",
            "model_id": request.model_id,
            "status_code": 200,
            **result
        }
    except Exception as e:
        logger.error(f"Error loading model {request.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.delete("/models/unload/{model_id:path}")
async def unload_model(model_id: str):
    if not runtime: 
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    try:
        return runtime.unload_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    if not scheduler: 
        raise HTTPException(status_code=503, detail="Service not available")
    
    if not runtime or not runtime.models:
        raise HTTPException(status_code=400, detail="No models loaded. Please load a model first.")
    
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")
    
    try:
        start_time = time.time()
        request_submitted_time = time.monotonic()
        
        result = await scheduler.submit_request(**request.dict())
        
        end_time = time.time()
        
        # Calculate metrics
        total_latency = result["total_time"]
        queue_time = max(0, total_latency - (end_time - start_time))  # Approximate queue time
        
        # Estimate TTFT and throughput
        text_length = len(result["text"])
        estimated_tokens = max(1, text_length // 4)  # Rough token estimate
        tokens_per_second = estimated_tokens / total_latency if total_latency > 0 else 0
        
        return {
            "generated_text": result["text"],
            "metrics": {
                "total_time": total_latency,
                "queue_time": queue_time,
                "inference_time": end_time - start_time,
                "ttft_estimate": min(0.1, total_latency * 0.1),  # Better TTFT estimate
                "tokens_per_second": tokens_per_second,
                "estimated_token_count": estimated_tokens,
                "method": "continuous_batching"
            },
            "model_info": {
                "model_id": request.model_id or "default",
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature
            }
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    if not runtime: 
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    
    memory = psutil.virtual_memory()
    gpu_info = {"gpu_available": torch.cuda.is_available()}
    if gpu_info["gpu_available"]:
        gpu_info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
    
    return {
        "memory": {"percentage": f"{memory.percent}%"}, 
        "gpu": gpu_info,
        "loaded_models_count": len(runtime.models),
        "scheduler_active": scheduler.running if scheduler else False,
        "service": "torchweave-server"
    }
