import os
import asyncio
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from contextlib import asynccontextmanager

# Import your existing modules
from .model_runtime import ModelRuntime
from .scheduler import ContinuousBatchScheduler

# Global variables
model_runtime: Optional[ModelRuntime] = None
scheduler: Optional[ContinuousBatchScheduler] = None

# Configuration from environment
MODEL_DIR = os.getenv("ARTIFACT_MODEL_DIR", "/artifacts/model")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("TOP_K", "0"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
SEED = int(os.getenv("SEED", "42"))
SCHEDULE_TICK_MS = int(os.getenv("SCHEDULE_TICK_MS", "15"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and scheduler on startup"""
    global model_runtime, scheduler
    
    try:
        # Initialize model runtime
        print(f"[server] Loading model from {MODEL_DIR}")
        model_runtime = ModelRuntime(MODEL_DIR)
        
        # Initialize continuous batch scheduler
        print(f"[server] Starting continuous batch scheduler")
        scheduler = ContinuousBatchScheduler(
            model_runtime=model_runtime,
            max_batch_size=MAX_BATCH,
            schedule_tick_ms=SCHEDULE_TICK_MS
        )
        
        # Start the scheduler
        await scheduler.start()
        print(f"[server] Server ready")
        
    except Exception as e:
        print(f"[server] Failed to initialize: {e}")
        raise
    
    yield
    
    # Cleanup
    if scheduler:
        await scheduler.stop()

# Create FastAPI app with lifespan
app = FastAPI(
    title="TorchWeave LLM Server",
    description="Inference Compiler for LLM Optimization with Continuous Batching",
    version="1.0.0",
    lifespan=lifespan
)

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

# Health endpoint
@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}

# Model info endpoint
@app.get("/model")
def model_info():
    """Get model information"""
    if not model_runtime:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        vocab_size = model_runtime.get_vocab_size()
        eos_token = model_runtime.get_eos_token()
        return {
            "vocab_size": vocab_size,
            "eos": eos_token,
            "device": str(model_runtime.device),
            "max_batch_size": MAX_BATCH
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Streaming generation endpoint with continuous batching
@app.post("/v1/generate")
async def generate_streaming(request: GenerateRequest):
    """Generate text with continuous batching and SSE streaming"""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    
    # Use request params or defaults
    max_new_tokens = request.max_new_tokens or MAX_NEW_TOKENS
    temperature = request.temperature if request.temperature is not None else TEMPERATURE
    top_k = request.top_k if request.top_k is not None else TOP_K
    top_p = request.top_p if request.top_p is not None else TOP_P
    seed = request.seed if request.seed is not None else SEED
    
    async def event_stream():
        try:
            # Submit request to scheduler
            request_id = await scheduler.submit_request(
                prompt=request.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed
            )
            
            # Stream tokens from scheduler
            ttft_sent = False
            async for event in scheduler.stream_tokens(request_id):
                if event["type"] == "ttft":
                    yield f"event: ttft\ndata: {event['time']:.4f}\n\n"
                    ttft_sent = True
                elif event["type"] == "token":
                    yield f"data: {event['token']}\n\n"
                elif event["type"] == "done":
                    yield f"event: done\ndata: \n\n"
                    break
                elif event["type"] == "error":
                    yield f"event: error\ndata: {event['error']}\n\n"
                    break
                    
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Baseline generation endpoint (no batching)
@app.post("/v1/generate_nobatch")
def generate_nobatch(request: GenerateRequest):
    """Generate text without batching (baseline comparison)"""
    if not model_runtime:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Use request params or defaults
    max_new_tokens = request.max_new_tokens or MAX_NEW_TOKENS
    temperature = request.temperature if request.temperature is not None else TEMPERATURE
    top_k = request.top_k if request.top_k is not None else TOP_K
    top_p = request.top_p if request.top_p is not None else TOP_P
    seed = request.seed if request.seed is not None else SEED
    
    try:
        # Direct generation without batching
        result = model_runtime.generate_single(
            prompt=request.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed
        )
        
        return {"text": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with basic info"""
    return {
        "service": "TorchWeave LLM Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "model_info": "/model", 
            "generate_streaming": "/v1/generate",
            "generate_baseline": "/v1/generate_nobatch"
        }
    }
