#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TorchWeave Model Loader - Fixed version to load models only into the Model Manager.
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
    def __init__(self, model_manager_url: str = "http://127.0.0.1:8001"):
        self.model_manager_url = model_manager_url
        
    async def check_server_health(self):
        """Check if the Model Manager is running"""
        print("Checking server health...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.model_manager_url}/health")
                if response.status_code == 200:
                    health = response.json()
                    print(f"[SUCCESS] Model Manager is healthy: {health.get('service', 'unknown')}")
                    return True
                else:
                    print(f"[ERROR] Model Manager health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to Model Manager at {self.model_manager_url}: {e}")
            return False
    
    async def list_loaded_models_safe(self, server_url: str, server_name: str, quiet: bool = False):
        """Safely list loaded models with error handling"""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(f"{server_url}/models/loaded")
                    if response.status_code == 200:
                        models_data = response.json()
                        models = models_data.get("loaded_models", [])
                        
                        if not quiet and models:
                            print(f"[INFO] {server_name} - loaded models ({len(models)}):")
                            for model in models:
                                model_id = model.get("model_id", "unknown")
                                status = model.get("status", "unknown")
                                print(f"   - {model_id} (Status: {status})")
                        elif not quiet:
                            print(f"[INFO] {server_name} - loaded models (0):")
                        
                        return models
                    else:
                        if not quiet:
                            print(f"[WARNING] Failed to list models from {server_name}: {response.status_code}")
            except Exception as e:
                if not quiet:
                    print(f"[WARNING] Error listing models from {server_name} (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
        
        return []
    
    async def load_model_on_server(self, model_name: str):
        """Load a model on the Model Manager with better timeout handling"""
        print(f"[INFO] Loading model '{model_name}' on Model Manager...")
        
        payload = {"model_id": model_name}
        endpoint = "/models/load"
        
        try:
            # Use longer timeout for model loading
            async with httpx.AsyncClient(timeout=600.0) as client:
                print(f"[DEBUG] POST {self.model_manager_url}{endpoint} with payload: {payload}")
                response = await client.post(f"{self.model_manager_url}{endpoint}", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[SUCCESS] Model load request successful on Model Manager!")
                    print(f"   Response: {result}")
                    return True
                else:
                    print(f"[ERROR] Load failed on Model Manager: HTTP {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Load request failed on Model Manager: {str(e)}")
            return False
    
    async def is_model_ready(self, model_name: str, models_data: Dict) -> bool:
        """Check if model is ready on the Model Manager"""
        mm_ready = False
        
        if "model_manager" in models_data:
            for model in models_data["model_manager"]:
                if model.get("model_id", "").lower() == model_name.lower():
                    status = model.get("status", "").lower()
                    if status in ["loaded", "available", "ready", "unknown"]:
                        mm_ready = True
                        break
        
        return mm_ready
    
    async def wait_for_model_load(self, model_name: str, max_wait_seconds: int = 180):
        """Wait for model with more flexible status checking"""
        print(f"[INFO] Waiting for {model_name} to load (max {max_wait_seconds}s)...")
        
        for i in range(max_wait_seconds):
            models = await self.list_loaded_models_safe(self.model_manager_url, "Model Manager", quiet=True)
            
            mm_ready = any(m.get("model_id", "").lower() == model_name.lower() and 
                          m.get("status", "").lower() in ["loaded", "available", "unknown"] 
                          for m in models)
            
            if mm_ready:
                print(f"[SUCCESS] Model {model_name} appears to be loaded!")
                return True
            
            if i % 15 == 0 and i > 0:
                print(f"   [INFO] Still loading... ({i}s elapsed)")
            
            await asyncio.sleep(1)
        
        print(f"[WARNING] Model {model_name} did not fully load within {max_wait_seconds} seconds")
        return False
    
    async def test_model_generation(self, model_name: str):
        """Test if the loaded model can generate text on the Model Manager"""
        print(f"[TEST] Testing text generation with {model_name}...")
        
        test_payload = {
            "model_id": model_name,
            "prompt": "Hello, how are you?",
            "max_new_tokens": 20,
            "temperature": 0.7
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.model_manager_url}/models/generate", json=test_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("generated_text", "")
                    
                    if generated_text and generated_text.strip():
                        print(f"[SUCCESS] Model Manager generation successful!")
                        print(f"   Generated: '{generated_text.strip()[:50]}...'")
                        return True
                    else:
                        print(f"[WARNING] Model Manager responded but generated empty text")
                        return False
                else:
                    print(f"[ERROR] Model Manager generation test failed: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Model Manager generation test failed: {e}")
            return False
    
    async def list_available_models(self):
        """List available models that can be loaded from the Model Manager"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.model_manager_url}/models/available")
                if response.status_code == 200:
                    models_data = response.json()
                    available = models_data.get("available_models", [])
                    
                    if available:
                        print(f"[INFO] Available models ({len(available)}):")
                        for model in available:
                            model_id = model.get("model_id", "unknown")
                            description = model.get("description", "")
                            size = model.get("size", "")
                            print(f"   - {model_id} ({size}) - {description}")
                    else:
                        print(f"[INFO] No available models found")
                    
                    return available
                else:
                    print(f"[WARNING] Could not fetch available models: {response.status_code}")
                    return []
        except Exception as e:
            print(f"[WARNING] Error fetching available models: {e}")
            return []
    
    async def load_model_comprehensive(self, model_name: str, force_load: bool = False):
        """Comprehensive model loading with all discovery and testing"""
        print(f"[INFO] Starting comprehensive model loading for: {model_name}")
        print("=" * 60)
        
        if not await self.check_server_health():
            return False
        
        await self.list_available_models()
        
        current_models = await self.list_loaded_models_safe(self.model_manager_url, "Model Manager")
        
        model_is_loaded = any(m.get("model_id", "").lower() == model_name.lower() and 
                                 m.get("status", "").lower() in ["loaded", "available"]
                                 for m in current_models)
        
        if model_is_loaded and not force_load:
            print(f"[INFO] Model {model_name} appears to be loaded.")
            return await self.test_model_generation(model_name)
        
        if not await self.load_model_on_server(model_name):
            return False
        
        if await self.wait_for_model_load(model_name):
            return await self.test_model_generation(model_name)
        
        print(f"[WARNING] Wait for load failed, but attempting generation test anyway...")
        return await self.test_model_generation(model_name)

async def main():
    parser = argparse.ArgumentParser(description="Load models into TorchWeave servers")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help="Model name to load")
    parser.add_argument("--url", default="http://127.0.0.1:8001", 
                       help="TorchWeave Model Manager URL")
    parser.add_argument("--list-available", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--list-loaded", action="store_true",
                       help="List loaded models and exit")
    parser.add_argument("--force-load", action="store_true",
                       help="Force reload model even if it appears loaded")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test generation without loading")
    
    args = parser.parse_args()
    
    loader = TorchWeaveModelLoader(args.url)
    
    print("TORCHWEAVE MODEL LOADER (Fixed Version)")
    print("=" * 60)
    print(f"Model Manager: {args.url}")
    print(f"Model: {args.model}")
    print()
    
    if args.list_available:
        await loader.list_available_models()
        return
    
    if args.list_loaded:
        await loader.list_loaded_models_safe(args.url, "Model Manager")
        return
    
    if args.test_only:
        success = await loader.test_model_generation(args.model)
    else:
        success = await loader.load_model_comprehensive(args.model, args.force_load)
    
    if success:
        print("\n" + "="*60)
        print("[SUCCESS] Model loading and testing completed successfully!")
        print("The Model Manager is ready.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[ERROR] Model loading failed or incomplete!")
        print("Check TorchWeave Model Manager logs for more details.")
        print("="*60)
        
if __name__ == "__main__":
    asyncio.run(main())
