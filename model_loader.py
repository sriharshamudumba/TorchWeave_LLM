#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TorchWeave Model Loader - Load models into your TorchWeave server
"""
import argparse
import asyncio
import json
import sys
from typing import Dict, Any, List, Optional

try:
    import httpx
except ImportError:
    print("Missing httpx. Install with: pip install httpx")
    sys.exit(1)

class TorchWeaveModelLoader:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    async def check_server_health(self):
        """Check if server is running"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    health = response.json()
                    print(f"[SUCCESS] TorchWeave server is healthy: {health.get('service', 'unknown')}")
                    return True
                else:
                    print(f"[ERROR] Server health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to server at {self.base_url}: {e}")
            return False
    
    async def list_loaded_models(self):
        """List currently loaded models"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/models")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    
                    if models:
                        print(f"[INFO] Currently loaded models ({len(models)}):")
                        for model in models:
                            model_id = model.get("model_id", "unknown")
                            status = model.get("status", "unknown")
                            print(f"   - {model_id} (Status: {status})")
                    else:
                        print(f"[INFO] No models currently loaded")
                    
                    return models
                else:
                    print(f"[ERROR] Failed to list models: {response.status_code}")
                    return []
        except Exception as e:
            print(f"[ERROR] Error listing models: {e}")
            return []
    
    async def load_model(self, model_name: str, model_path: Optional[str] = None):
        """Load a model into TorchWeave using the correct endpoint"""
        print(f"[INFO] Attempting to load model: {model_name}")
        
        # Correct endpoint and payload based on model_manager.py
        endpoint = "/models/load/huggingface"
        payload = {"model_id": model_name}
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for model loading
            try:
                print(f"[DEBUG] Trying POST {self.base_url}{endpoint} with payload: {payload}")
                
                response = await client.post(f"{self.base_url}{endpoint}", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[SUCCESS] Model load request successful!")
                    print(f"   Response: {result}")
                    return True
                else:
                    print(f"   [ERROR] HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"   [ERROR] Request failed: {str(e)}")
        
        print(f"[ERROR] Model loading request failed.")
        return False
    
    async def wait_for_model_load(self, model_name: str, max_wait_seconds: int = 300):
        """Wait for model to appear in loaded models list and become available"""
        print(f"[INFO] Waiting for {model_name} to load (max {max_wait_seconds}s)...")
        
        for i in range(max_wait_seconds):
            models_data = await self.list_loaded_models()
            
            # Find the model in the list
            model_info = next((m for m in models_data if m.get("model_id", "").lower() == model_name.lower()), None)
            
            if model_info:
                status = model_info.get("status")
                if status == "available":
                    print(f"[SUCCESS] Model {model_name} successfully loaded!")
                    return True
                elif status == "loading":
                    progress = model_info.get("load_progress", 0)
                    print(f"   [INFO] Still loading... Progress: {progress:.1f}% ({i}s elapsed)")
                elif status == "error":
                    error_msg = model_info.get("error_message", "unknown error")
                    print(f"[ERROR] Model {model_name} failed to load: {error_msg}")
                    return False
            
            await asyncio.sleep(1)
        
        print(f"[ERROR] Model {model_name} did not load within {max_wait_seconds} seconds")
        return False
    
    async def test_model_generation(self, model_name: str):
        """Test if the loaded model can generate text"""
        print(f"[TEST] Testing text generation with {model_name}...")
        
        test_payload = {
            "model_id": model_name,
            "prompt": "Hello, how are you?",
            "max_new_tokens": 20,
            "temperature": 0.7
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/models/generate", json=test_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    generated_text = result.get("generated_text", "")
                    
                    if generated_text.strip():
                        print(f"[SUCCESS] Text generation successful!")
                        print(f"   Generated: '{generated_text}'")
                        return True
                    else:
                        print(f"[WARNING] Model responded but generated empty text")
                        print(f"   Response: {result}")
                        return False
                else:
                    print(f"[ERROR] Generation test failed: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Generation test failed: {e}")
            return False
    
    async def load_model_comprehensive(self, model_name: str):
        """Comprehensive model loading with all discovery and testing"""
        print(f"[INFO] Starting comprehensive model loading for: {model_name}")
        print("=" * 60)
        
        # 1. Check server health
        if not await self.check_server_health():
            return False
        
        # 2. List current models and check if already loaded
        current_models = await self.list_loaded_models()
        model_already_loaded = any(model.get("model_id", "").lower() == model_name.lower() and model.get("status") == "available" for model in current_models)
        
        if model_already_loaded:
            print(f"[INFO] Model {model_name} is already loaded")
            return await self.test_model_generation(model_name)
        
        # 3. Try to load model
        load_success = await self.load_model(model_name)
        
        if load_success:
            # 4. Wait for model to be loaded
            return await self.wait_for_model_load(model_name)
        
        return False

async def main():
    parser = argparse.ArgumentParser(description="Load models into TorchWeave server")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name to load")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="TorchWeave Model Manager URL")
    
    args = parser.parse_args()
    
    # IMPORTANT: The model loading endpoint is on the Model Manager service, not the main server
    if args.url.endswith("8000"):
        print("[WARNING] The model manager runs on port 8001. Updating URL to http://127.0.0.1:8001")
        args.url = "http://127.0.0.1:8001"
    
    loader = TorchWeaveModelLoader(args.url)
    
    print("TORCHWEAVE MODEL LOADER")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Model: {args.model}")
    print()
    
    # Full model loading process
    success = await loader.load_model_comprehensive(args.model)
    
    if success:
        print("\n[SUCCESS] Model loading and testing completed successfully!")
        print("You can now run benchmarks with this model.")
    else:
        print("\n[ERROR] Model loading failed!")
        print("Check TorchWeave server logs for more details.")
        
if __name__ == "__main__":
    asyncio.run(main())
