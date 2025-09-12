import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import httpx
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInfo(BaseModel):
    model_id: str
    source: str
    display_name: str
    description: str
    status: str
    path: Optional[str] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    model_id: Optional[str] = None
    path: Optional[str] = None

app = FastAPI(title="TorchWeave Model Manager", version="1.0.0")

# Configuration
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/artifacts")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/model_cache")
HUGGINGFACE_CACHE_DIR = os.environ.get("HUGGINGFACE_CACHE_DIR", "/model_cache/huggingface")
SERVER_URL = os.environ.get("SERVER_URL", "http://server-cpu:8000")

# Ensure directories exist
Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(HUGGINGFACE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# In-memory model registry
model_registry: Dict[str, ModelInfo] = {}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "model-manager"}

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(models=list(model_registry.values()))

@app.post("/models/load", response_model=StatusResponse)
async def load_model(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received load request: {data}")
        
        if "model_config" in data:
            # Structured request
            config = data["model_config"]
            model_id = config.get("model_id")
            source = config.get("source", "huggingface")
            display_name = config.get("display_name", model_id)
            description = config.get("description", f"Model {model_id}")
        else:
            # Flat request (fallback)
            model_id = data.get("model_id")
            source = data.get("source", "huggingface")
            display_name = data.get("display_name", model_id)
            description = data.get("description", f"Model {model_id}")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        return await _load_model_internal(model_id, source, display_name, description)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load_flat", response_model=StatusResponse)
async def load_model_flat(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received flat load request: {data}")
        
        model_id = data.get("model_id")
        source = data.get("source", "huggingface")
        display_name = data.get("display_name", model_id)
        description = data.get("description", f"Model {model_id}")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        return await _load_model_internal(model_id, source, display_name, description)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _load_model_internal(model_id: str, source: str, display_name: str, description: str) -> StatusResponse:
    logger.info(f"Loading model: {model_id} from {source}")
    
    if model_id in model_registry:
        logger.info(f"Model {model_id} already in registry")
        return StatusResponse(status="success", message="Model already loaded", model_id=model_id)
    
    model_info = ModelInfo(
        model_id=model_id,
        source=source,
        display_name=display_name,
        description=description,
        status="loading"
    )
    model_registry[model_id] = model_info
    
    try:
        if source.lower() == "huggingface":
            await _download_huggingface_model(model_id)
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        model_path = os.path.join(ARTIFACT_DIR, "model")
        model_info.path = model_path
        model_info.status = "ready"
        
        logger.info(f"Model {model_id} loaded successfully")
        return StatusResponse(
            status="success",
            message="Model loaded successfully",
            model_id=model_id,
            path=model_path
        )
        
    except Exception as e:
        model_info.status = "failed"
        logger.error(f"Failed to load model {model_id}: {e}")
        if model_id in model_registry:
            del model_registry[model_id]
        raise

async def _download_huggingface_model(model_id: str):
    logger.info(f"Downloading {model_id} from Hugging Face")
    
    cache_path = os.path.join(HUGGINGFACE_CACHE_DIR, model_id.replace("/", "--"))
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=cache_path,
            local_dir_use_symlinks=False
        )
        
        artifact_model_path = os.path.join(ARTIFACT_DIR, "model")
        if os.path.exists(artifact_model_path):
            shutil.rmtree(artifact_model_path)
        
        shutil.copytree(cache_path, artifact_model_path)
        logger.info(f"Model {model_id} staged to {artifact_model_path}")
        
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise

@app.delete("/models/{model_id}")
async def unload_model(model_id: str):
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        del model_registry[model_id]
        logger.info(f"Model {model_id} unloaded")
        return StatusResponse(status="success", message="Model unloaded", model_id=model_id)
        
    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str) -> ModelInfo:
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_registry[model_id]

@app.get("/server/status")
async def check_server_status():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SERVER_URL}/health", timeout=5.0)
            return {"server_status": "online", "details": response.json()}
    except Exception as e:
        return {"server_status": "offline", "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
