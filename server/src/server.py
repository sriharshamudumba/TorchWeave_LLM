import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import uvicorn

from scheduler import ContinuousBatchScheduler
from model_runtime import ModelRuntime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler: Optional[ContinuousBatchScheduler] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global scheduler
    
    try:
        # Load model
        model_dir = os.environ.get("ARTIFACT_MODEL_DIR", "/artifacts/model")
        logger.info(f"[server] Loading model from {model_dir}")
        
        model_runtime = ModelRuntime(model_dir)
        
        # Initialize scheduler - Fix: Remove schedule_tick_ms parameter
        logger.info("[server] Starting continuous batch scheduler")
        max_batch = int(os.environ.get("MAX_BATCH", "16"))
        
        # Create scheduler without the problematic parameter
        scheduler = ContinuousBatchScheduler(
            model_runtime=model_runtime,
            max_batch_size=max_batch
        )
        
        # Start the scheduler
        asyncio.create_task(scheduler.run())
        
        logger.info("[server] Server initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"[server] Failed to initialize: {e}")
        raise
    finally:
        if scheduler:
            await scheduler.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/model")
async def model_info():
    """Get model information"""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    runtime = scheduler.model_runtime
    return {
        "vocab_size": runtime.tokenizer.vocab_size,
        "eos": runtime.tokenizer.eos_token,
        "device": str(runtime.device),
        "max_batch_size": scheduler.max_batch_size
    }

@app.post("/v1/generate")
async def generate_stream(request: GenerateRequest):
    """Generate text with streaming response"""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Submit request to scheduler
            request_id = await scheduler.submit_request(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed
            )
            
            # Stream tokens
            async for token in scheduler.stream_tokens(request_id):
                if token.startswith("event: ttft"):
                    yield f"{token}\n"
                elif token.startswith("event: done"):
                    yield f"{token}\n"
                    break
                else:
                    yield f"data: {token}\n"
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"data: [ERROR] {str(e)}\n"
            yield "event: done\ndata:\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/v1/generate_nobatch", response_model=GenerateResponse)
async def generate_nobatch(request: GenerateRequest):
    """Generate text without batching (baseline)"""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use the model runtime directly for non-batched generation
        runtime = scheduler.model_runtime
        text = runtime.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed
        )
        return GenerateResponse(text=text)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
