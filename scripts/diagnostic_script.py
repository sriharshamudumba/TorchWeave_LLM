#!/usr/bin/env python3
"""
TorchWeave Debug Script - Diagnose benchmark failures
"""
import asyncio
import httpx
import json

async def debug_torchweave_setup():
    """Comprehensive debug of TorchWeave setup"""
    print("=== TORCHWEAVE DEBUG DIAGNOSTIC ===\n")
    
    server_url = "http://127.0.0.1:8000"
    model_manager_url = "http://127.0.0.1:8001"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test 1: Server Health
        print("1. Testing Server Health...")
        try:
            response = await client.get(f"{server_url}/health")
            print(f"   Main Server: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   Main Server: FAILED - {e}")
        
        try:
            response = await client.get(f"{model_manager_url}/health")
            print(f"   Model Manager: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   Model Manager: FAILED - {e}")
        
        # Test 2: Available Models
        print("\n2. Checking Available Models...")
        try:
            response = await client.get(f"{model_manager_url}/models/available")
            if response.status_code == 200:
                data = response.json()
                print(f"   Available models: {json.dumps(data, indent=2)}")
            else:
                print(f"   Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 3: Loaded Models
        print("\n3. Checking Loaded Models...")
        for server_name, url in [("Main Server", server_url), ("Model Manager", model_manager_url)]:
            try:
                response = await client.get(f"{url}/models/loaded")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   {server_name}: {json.dumps(data, indent=2)}")
                else:
                    print(f"   {server_name}: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   {server_name}: Error - {e}")
        
        # Test 4: Test Generation Endpoints
        print("\n4. Testing Generation Endpoints...")
        test_payload = {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": "Hello",
            "max_new_tokens": 5,
            "temperature": 0.7
        }
        
        # Test Model Manager endpoint
        try:
            response = await client.post(f"{model_manager_url}/models/generate", json=test_payload)
            print(f"   Model Manager (/models/generate): {response.status_code}")
            if response.status_code == 200:
                print(f"      Response: {response.json()}")
            else:
                print(f"      Error: {response.text}")
        except Exception as e:
            print(f"   Model Manager: Error - {e}")
        
        # Test Main Server endpoint
        try:
            response = await client.post(f"{server_url}/v1/generate", json=test_payload)
            print(f"   Main Server (/v1/generate): {response.status_code}")
            if response.status_code == 200:
                print(f"      Response: {response.json()}")
            else:
                print(f"      Error: {response.text}")
        except Exception as e:
            print(f"   Main Server: Error - {e}")
        
        # Test 5: Alternative endpoints (in case paths are different)
        print("\n5. Testing Alternative Endpoints...")
        alt_endpoints = ["/generate", "/api/generate", "/inference", "/predict"]
        
        for endpoint in alt_endpoints:
            try:
                response = await client.post(f"{server_url}{endpoint}", json=test_payload)
                print(f"   {server_url}{endpoint}: {response.status_code}")
            except:
                pass
            
            try:
                response = await client.post(f"{model_manager_url}{endpoint}", json=test_payload)
                print(f"   {model_manager_url}{endpoint}: {response.status_code}")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(debug_torchweave_setup())
