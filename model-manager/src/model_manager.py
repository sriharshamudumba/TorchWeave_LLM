from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import json

# Define proper Pydantic models for request validation
class ModelConfig(BaseModel):
    model_id: str = Field(..., description="Model identifier (e.g., 'gpt2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')")
    source: str = Field(default="huggingface", description="Model source (huggingface, local, etc.)")
    display_name: Optional[str] = Field(None, description="Human-readable name for the model")
    description: Optional[str] = Field(None, description="Description of the model")
    
class ModelLoadRequest(BaseModel):
    model_config: ModelConfig = Field(..., description="Model configuration")
    
class ModelLoadResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    status_code: int = 200

class ModelListResponse(BaseModel):
    models: list = Field(default_factory=list)
    count: int = 0

# Updated endpoint handlers
class ModelManager:
    def __init__(self):
        self.app = FastAPI(title="TorchWeave Model Manager", version="1.0.0")
        self.loaded_models = {}
        self.model_registry = {}
        self.logger = logging.getLogger(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes with proper parameter validation"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "service": "model-manager"}
            
        @self.app.get("/models", response_model=ModelListResponse)
        async def list_models():
            """List all available models"""
            try:
                models_list = [
                    {
                        "model_id": model_id,
                        "status": "loaded" if model_id in self.loaded_models else "available",
                        **model_info
                    }
                    for model_id, model_info in self.model_registry.items()
                ]
                
                return ModelListResponse(
                    models=models_list,
                    count=len(models_list)
                )
                
            except Exception as e:
                self.logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail="Failed to list models")
        
        @self.app.post("/models/load", response_model=ModelLoadResponse)
        async def load_model_endpoint(request: ModelLoadRequest):
            """Load a model with proper parameter validation"""
            try:
                # Now we have properly validated Pydantic objects
                model_config = request.model_config
                
                self.logger.info(f"Loading model: {model_config.model_id}")
                
                # Validate required fields
                if not model_config.model_id:
                    return ModelLoadResponse(
                        success=False,
                        error="model_id is required",
                        status_code=400
                    )
                
                # Check if already loaded
                if model_config.model_id in self.loaded_models:
                    return ModelLoadResponse(
                        success=True,
                        message=f"Model {model_config.model_id} is already loaded",
                        status_code=200
                    )
                
                # Load the model
                success = await self._load_model_impl(model_config)
                
                if success:
                    return ModelLoadResponse(
                        success=True,
                        message=f"Model {model_config.model_id} loaded successfully",
                        status_code=200
                    )
                else:
                    return ModelLoadResponse(
                        success=False,
                        error=f"Failed to load model {model_config.model_id}",
                        status_code=500
                    )
                    
            except Exception as e:
                self.logger.error(f"Model loading error: {e}")
                return ModelLoadResponse(
                    success=False,
                    error=f"Internal server error: {str(e)}",
                    status_code=500
                )
        
        # Alternative endpoint that accepts flat JSON (backward compatibility)
        @self.app.post("/models/load_flat")
        async def load_model_flat(request: Request):
            """Alternative endpoint that accepts flat JSON structure"""
            try:
                # Get raw JSON
                json_data = await request.json()
                
                # Handle both nested and flat structures
                if "model_config" in json_data:
                    # Nested structure: {"model_config": {...}}
                    config_data = json_data["model_config"]
                else:
                    # Flat structure: {"model_id": "...", "source": "...", ...}
                    config_data = json_data
                
                # Create ModelConfig from dict
                model_config = ModelConfig(**config_data)
                
                # Create request object
                model_request = ModelLoadRequest(model_config=model_config)
                
                # Reuse the main load endpoint logic
                return await load_model_endpoint(model_request)
                
            except Exception as e:
                self.logger.error(f"Flat model loading error: {e}")
                return ModelLoadResponse(
                    success=False,
                    error=f"Invalid request format: {str(e)}",
                    status_code=400
                )
        
        @self.app.delete("/models/{model_id}")
        async def unload_model(model_id: str):
            """Unload a specific model"""
            try:
                if model_id not in self.loaded_models:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
                # Unload model implementation
                success = await self._unload_model_impl(model_id)
                
                if success:
                    return {"success": True, "message": f"Model {model_id} unloaded"}
                else:
                    return {"success": False, "error": f"Failed to unload {model_id}"}
                    
            except Exception as e:
                self.logger.error(f"Model unloading error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/models/{model_id}/info")
        async def get_model_info(model_id: str):
            """Get information about a specific model"""
            try:
                if model_id in self.loaded_models:
                    model_info = self.loaded_models[model_id]
                    return {
                        "model_id": model_id,
                        "status": "loaded",
                        "info": model_info
                    }
                elif model_id in self.model_registry:
                    return {
                        "model_id": model_id,
                        "status": "available",
                        "info": self.model_registry[model_id]
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                    
            except Exception as e:
                self.logger.error(f"Get model info error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_model_impl(self, model_config: ModelConfig) -> bool:
        """Implementation of model loading logic"""
        try:
            self.logger.info(f"Loading model {model_config.model_id} from {model_config.source}")
            
            # Add your actual model loading logic here
            # For now, simulate loading
            if model_config.source == "huggingface":
                # Simulate HuggingFace model loading
                await self._load_huggingface_model(model_config)
            else:
                raise ValueError(f"Unsupported model source: {model_config.source}")
            
            # Store in loaded models
            self.loaded_models[model_config.model_id] = {
                "model_id": model_config.model_id,
                "source": model_config.source,
                "display_name": model_config.display_name,
                "description": model_config.description,
                "loaded_at": "2024-01-01T00:00:00Z"  # Add timestamp
            }
            
            self.logger.info(f"Model {model_config.model_id} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_config.model_id}: {e}")
            return False
    
    async def _load_huggingface_model(self, model_config: ModelConfig):
        """Load model from HuggingFace"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Download and load tokenizer and model
            self.logger.info(f"Downloading tokenizer for {model_config.model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
            
            self.logger.info(f"Downloading model for {model_config.model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Store model and tokenizer (you might want to save these somewhere accessible)
            # This is where you'd integrate with your model storage system
            
            return True
            
        except Exception as e:
            self.logger.error(f"HuggingFace model loading failed: {e}")
            raise
    
    async def _unload_model_impl(self, model_id: str) -> bool:
        """Implementation of model unloading logic"""
        try:
            if model_id in self.loaded_models:
                # Add actual unloading logic here (free memory, etc.)
                del self.loaded_models[model_id]
                self.logger.info(f"Model {model_id} unloaded")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_id}: {e}")
            return False

# Create the FastAPI app instance
model_manager = ModelManager()
app = model_manager.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
