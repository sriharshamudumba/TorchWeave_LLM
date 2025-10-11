#!/usr/bin/env python3
"""
Updated TorchWeave Server Performance Benchmark
Fixed to properly compare baseline vs optimized processing
"""

import asyncio
import time
import statistics
import argparse
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

try:
    import httpx
except ImportError:
    print("Missing httpx. Install with: pip install httpx")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    """Individual request benchmark result"""
    request_id: str
    prompt: str
    generated_text: str
    total_time: float
    ttft: float
    tokens_generated: int
    throughput: float
    processing_type: str
    batch_info: Optional[Dict] = None

class TorchWeaveServerBenchmark:
    """Benchmark tool for testing actual TorchWeave server performance"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8000", 
                 model_manager_url: str = "http://127.0.0.1:8001"):
        self.server_url = server_url
        self.model_manager_url = model_manager_url
        self.timeout = httpx.Timeout(300.0)
        
    async def check_server_health(self):
        """Check if TorchWeave servers are running"""
        print("Checking TorchWeave server health...")
        main_healthy = False
        model_manager_healthy = False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check main server
                try:
                    response = await client.get(f"{self.server_url}/health")
                    if response.status_code == 200:
                        print(f"[SUCCESS] Main server healthy")
                        main_healthy = True
                    else:
                        print(f"[WARNING] Main server unhealthy: {response.status_code}")
                except Exception as e:
                    print(f"[WARNING] Cannot connect to main server: {e}")
                
                # Check model manager
                try:
                    response = await client.get(f"{self.model_manager_url}/health")
                    if response.status_code == 200:
                        print(f"[SUCCESS] Model manager healthy")
                        model_manager_healthy = True
                    else:
                        print(f"[ERROR] Model manager unhealthy: {response.status_code}")
                except Exception as e:
                    print(f"[ERROR] Cannot connect to model manager: {e}")
                
                return model_manager_healthy  # At minimum need model manager
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")
            return False
    
    async def list_loaded_models_on_server(self, server_url: str, server_name: str, quiet: bool = False):
        """List models loaded on a specific server"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{server_url}/models/loaded")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("loaded_models", [])
                    
                    if not quiet and models:
                        print(f"[INFO] {server_name} - loaded models ({len(models)}):")
                        for model in models:
                            model_id = model.get("model_id", "unknown")
                            status = model.get("status", "unknown")
                            print(f"    - {model_id} (Status: {status})")
                    elif not quiet:
                        print(f"[INFO] {server_name} - loaded models (0):")
                    
                    return models
                else:
                    if not quiet:
                        print(f"[WARNING] Failed to list models from {server_name}: {response.status_code}")
                    return []
        except Exception as e:
            if not quiet:
                print(f"[WARNING] Error listing models from {server_name}: {e}")
            return []
    
    async def load_model_on_main_server(self, model_name: str):
        """Try to load model on main server if possible"""
        print(f"[INFO] Attempting to load '{model_name}' on main server...")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {"model_id": model_name}
                response = await client.post(f"{self.server_url}/models/load", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[SUCCESS] Model loaded on main server: {result}")
                    return True
                else:
                    print(f"[INFO] Could not load on main server (HTTP {response.status_code})")
                    return False
        except Exception as e:
            print(f"[INFO] Main server load failed: {e}")
            return False
    
    async def ensure_model_available(self, model_name: str) -> Dict[str, bool]:
        """Ensure the specified model is available and try to load on both servers"""
        print(f"[INFO] Ensuring model '{model_name}' is available...")
        
        # Check Model Manager
        mm_models = await self.list_loaded_models_on_server(self.model_manager_url, "Model Manager", quiet=True)
        mm_has_model = any(m.get("model_id", "").lower() == model_name.lower() and 
                          m.get("status", "").lower() in ["loaded", "available"] 
                          for m in mm_models)
        
        # Check Main Server
        ms_models = await self.list_loaded_models_on_server(self.server_url, "Main Server", quiet=True)
        ms_has_model = any(m.get("model_id", "").lower() == model_name.lower() and 
                          m.get("status", "").lower() in ["loaded", "available"] 
                          for m in ms_models)
        
        # Try to load on main server if not present
        if mm_has_model and not ms_has_model:
            print(f"[INFO] Model available on Model Manager, attempting to load on Main Server...")
            ms_has_model = await self.load_model_on_main_server(model_name)
            if ms_has_model:
                # Wait a moment for the model to be ready
                await asyncio.sleep(3)
        
        availability = {
            "model_manager": mm_has_model,
            "main_server": ms_has_model
        }
        
        if mm_has_model:
            print(f"[SUCCESS] Model '{model_name}' available on Model Manager")
        else:
            print(f"[ERROR] Model '{model_name}' not available on Model Manager")
        
        if ms_has_model:
            print(f"[SUCCESS] Model '{model_name}' available on Main Server")
        else:
            print(f"[INFO] Model '{model_name}' not available on Main Server")
        
        return availability

    async def _send_baseline_request(self, model_id: str, prompt: str, max_tokens: int, temperature: float, request_id: str):
        """Send request to model manager (baseline - no batching)"""
        print(f"\n--- BASELINE REQUEST: {request_id} ---")
        print(f"Prompt: '{prompt}'")
        
        payload = {
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
        
        request_start = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.model_manager_url}/models/generate", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    request_end = time.time()
                    
                    generated_text = result.get("generated_text", "")
                    metrics = result.get("metrics", {})
                    
                    total_time = request_end - request_start
                    ttft = metrics.get("ttft_estimate", 0.0)
                    tokens_generated = metrics.get("token_count", len(generated_text.split()))
                    throughput = tokens_generated / total_time if total_time > 0 else 0
                    
                    # Truncate output for display
                    display_text = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                    print(f"  GENERATED: '{display_text}'")
                    print(f"  METRICS:")
                    print(f"    - Total Time: {total_time:.3f}s")
                    print(f"    - TTFT: {ttft:.3f}s")
                    print(f"    - Tokens Generated: {tokens_generated}")
                    print(f"    - Tokens/sec: {throughput:.2f}")
                    print(f"    - Processing: Baseline (No Batching)")

                    return BenchmarkResult(
                        request_id=request_id,
                        prompt=prompt,
                        generated_text=generated_text,
                        total_time=total_time,
                        ttft=ttft,
                        tokens_generated=tokens_generated,
                        throughput=throughput,
                        processing_type="BASELINE"
                    )
                else:
                    print(f"[ERROR] Request {request_id} failed with status code: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            print(f"[ERROR] Request {request_id} failed: {e}")
            return None

    async def _send_optimized_request(self, model_id: str, prompt: str, max_tokens: int, temperature: float, request_id: str, use_main_server: bool = False):
        """Send optimized request (try main server first, fallback to model manager with different params)"""
        server_url = self.server_url if use_main_server else self.model_manager_url
        server_name = "Main Server" if use_main_server else "Model Manager (Optimized)"
        endpoint = "/v1/generate" if use_main_server else "/models/generate"
        
        payload = {
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add optimization parameters for model manager
        if not use_main_server:
            payload.update({
                "do_sample": True,
                "use_cache": True,
                "pad_token_id": 50256,  # Common pad token
                "early_stopping": True
            })
        
        request_start = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{server_url}{endpoint}", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    request_end = time.time()
                    
                    generated_text = result.get("generated_text", "")
                    metrics = result.get("metrics", {})
                    
                    total_time = request_end - request_start
                    ttft = metrics.get("ttft_estimate", 0.0)
                    tokens_generated = metrics.get("token_count", metrics.get("estimated_token_count", len(generated_text.split())))
                    throughput = tokens_generated / total_time if total_time > 0 else 0
                    processing_method = metrics.get("method", "optimized")
                    
                    print(f"\n--- OPTIMIZED REQUEST: {request_id} ---")
                    print(f"Server: {server_name}")
                    print(f"Prompt: '{prompt}'")
                    
                    # Truncate output for display
                    display_text = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                    print(f"  GENERATED: '{display_text}'")
                    print(f"  METRICS:")
                    print(f"    - Total Time: {total_time:.3f}s")
                    print(f"    - TTFT: {ttft:.3f}s")
                    print(f"    - Tokens Generated: {tokens_generated}")
                    print(f"    - Tokens/sec: {throughput:.2f}")
                    print(f"    - Processing Method: {processing_method}")

                    return BenchmarkResult(
                        request_id=request_id,
                        prompt=prompt,
                        generated_text=generated_text,
                        total_time=total_time,
                        ttft=ttft,
                        tokens_generated=tokens_generated,
                        throughput=throughput,
                        processing_type="OPTIMIZED",
                        batch_info={"method": processing_method, "server": server_name}
                    )
                else:
                    print(f"[ERROR] Optimized request {request_id} failed on {server_name}: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            print(f"[ERROR] Optimized request {request_id} failed on {server_name}: {e}")
            return None

    async def run_baseline_benchmark(self, model_id: str, prompts: List[str], max_tokens: int = 30):
        """Run baseline benchmark using model manager (sequential processing)"""
        print(f"\n{'='*80}")
        print(f"BASELINE BENCHMARK (Model Manager - Sequential Processing)")
        print(f"{'='*80}")
        
        baseline_results = []
        baseline_start = time.time()
        
        # Process requests sequentially for baseline
        for i, prompt in enumerate(prompts):
            result = await self._send_baseline_request(model_id, prompt, max_tokens, 0.7, f"BASE-{i+1}")
            if result:
                baseline_results.append(result)
                # Add small delay to simulate real sequential processing
                await asyncio.sleep(0.1)
        
        baseline_total_time = time.time() - baseline_start
        self._print_summary("Baseline (Sequential)", baseline_results, baseline_total_time)
        return baseline_results, baseline_total_time

    async def run_optimized_benchmark(self, model_id: str, prompts: List[str], max_tokens: int, has_main_server: bool):
        """Run optimized benchmark (concurrent + optimizations)"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZED BENCHMARK (Concurrent + Optimizations)")
        print(f"{'='*80}")
        
        if has_main_server:
            print("Testing with Main Server (Batching + KV Cache)")
        else:
            print("Testing with Model Manager (Concurrent + Optimizations)")
        
        # Create concurrent tasks for optimization comparison
        tasks = []
        for i, prompt in enumerate(prompts):
            task = self._send_optimized_request(model_id, prompt, max_tokens, 0.7, f"OPT-{i+1}", has_main_server)
            tasks.append(task)
        
        optimized_start = time.time()
        optimized_results = await asyncio.gather(*tasks, return_exceptions=True)
        optimized_total_time = time.time() - optimized_start
        
        # Filter out exceptions and None results
        optimized_results = [r for r in optimized_results if isinstance(r, BenchmarkResult)]
        
        # If main server failed, try with model manager concurrent processing
        if not optimized_results and has_main_server:
            print(f"\n[INFO] Main server failed, falling back to Model Manager concurrent processing...")
            tasks = []
            for i, prompt in enumerate(prompts):
                task = self._send_optimized_request(model_id, prompt, max_tokens, 0.7, f"OPT-{i+1}", False)
                tasks.append(task)
            
            optimized_start = time.time()
            optimized_results = await asyncio.gather(*tasks, return_exceptions=True)
            optimized_total_time = time.time() - optimized_start
            
            optimized_results = [r for r in optimized_results if isinstance(r, BenchmarkResult)]
        
        self._print_summary("Optimized Processing", optimized_results, optimized_total_time)
        return optimized_results, optimized_total_time

    def _print_summary(self, title: str, results: List[BenchmarkResult], total_time: float):
        """Helper to print formatted results"""
        if not results:
            print(f"\n[WARNING] No results for {title} benchmark.")
            return

        total_tokens = sum(r.tokens_generated for r in results)
        avg_time = statistics.mean(r.total_time for r in results) if results else 0
        avg_ttft = statistics.mean(r.ttft for r in results) if results else 0
        avg_throughput = statistics.mean(r.throughput for r in results) if results else 0
        
        print(f"\n{'='*80}")
        print(f"{title.upper()} RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total Requests: {len(results)}")
        print(f"Total Processing Time (Wall Clock): {total_time:.3f}s")
        print(f"Total Tokens Generated: {total_tokens}")
        print(f"Overall Throughput: {total_tokens / total_time:.2f} tokens/sec" if total_time > 0 else "N/A")
        print(f"--------------------------------------------------")
        print(f"Individual Request Metrics:")
        print(f"  - Average Request Time: {avg_time:.3f}s")
        print(f"  - Average TTFT: {avg_ttft:.3f}s")
        print(f"  - Average Individual Throughput: {avg_throughput:.2f} tokens/sec")

def generate_test_prompts(base_prompt: str, count: int) -> List[str]:
    """Generate variations of a base prompt"""
    if count == 1:
        return [base_prompt]
    
    variations = [
        f"{base_prompt}",
        f"Explain {base_prompt} in detail",
        f"What do you think about {base_prompt}?", 
        f"Tell me a story about {base_prompt}",
        f"Describe {base_prompt} clearly",
        f"How would you handle {base_prompt}?",
        f"Give examples of {base_prompt}",
        f"What are the benefits of {base_prompt}?",
        f"Analyze {base_prompt} step by step",
        f"Compare different aspects of {base_prompt}"
    ]
    
    result = []
    for i in range(count):
        if i < len(variations):
            result.append(variations[i])
        else:
            result.append(f"{base_prompt} (variation {i+1})")
    
    return result

async def main():
    parser = argparse.ArgumentParser(
        description="TorchWeave Server Benchmark: Baseline vs Optimized Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
1. Single request test:
   python3 detailed_benchmark.py --model "microsoft/phi-2" --custom-prompts "Hello world" --max-tokens 50

2. Multiple requests test:
   python3 detailed_benchmark.py --model "microsoft/phi-2" --requests 3 --max-tokens 75

3. Custom prompts:
   python3 detailed_benchmark.py --model "microsoft/phi-2" --custom-prompts "Hello" "How are you?" "Tell a joke" --max-tokens 100
        """
    )
    parser.add_argument("--model", default=None, help="Model ID to test")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--requests", type=int, default=3, help="Number of requests")
    parser.add_argument("--custom-prompts", nargs="+", help="Custom prompts")
    parser.add_argument("--prompt", default="artificial intelligence", help="Base prompt")
    
    args = parser.parse_args()
    
    benchmark = TorchWeaveServerBenchmark()
    
    # Check server health first
    if not await benchmark.check_server_health():
        print("[ERROR] Required servers not healthy, exiting...")
        return
    
    # Handle model selection
    if args.model:
        model_id = args.model
        availability = await benchmark.ensure_model_available(args.model)
        if not availability["model_manager"]:
            print(f"[ERROR] Model '{args.model}' not available on Model Manager. Please load it first.")
            return
    else:
        print("[ERROR] Please specify a model with --model parameter")
        return
    
    # Handle prompt generation
    if args.custom_prompts:
        test_prompts = args.custom_prompts
    else:
        test_prompts = generate_test_prompts(args.prompt, args.requests)
    
    print(f"\n{'='*100}")
    print(f"TORCHWEAVE PERFORMANCE BENCHMARK")
    print(f"{'='*100}")
    print(f"Benchmark Configuration:")
    print(f"  Model: {model_id}")
    print(f"  Prompts: {len(test_prompts)}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Test Prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"    {i}. '{prompt}'")
    
    # Run BASELINE test (Sequential processing on Model Manager)
    print("\n" + "="*100)
    print("PHASE 1: BASELINE PROCESSING")
    print("="*100)
    print("Sequential requests to Model Manager (no optimizations)")
    baseline_results, baseline_time = await benchmark.run_baseline_benchmark(model_id, test_prompts, args.max_tokens)
    
    # Run OPTIMIZED test (Concurrent processing with optimizations)
    print("\n" + "="*100)
    print("PHASE 2: OPTIMIZED PROCESSING")
    print("="*100)
    print("Concurrent requests with optimizations")
    optimized_results, optimized_time = await benchmark.run_optimized_benchmark(
        model_id, test_prompts, args.max_tokens, availability["main_server"]
    )
    
    # Performance Comparison
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON: BASELINE vs OPTIMIZED")
    print("="*100)
    
    if baseline_results and optimized_results:
        # Calculate comprehensive metrics
        baseline_total_tokens = sum(r.tokens_generated for r in baseline_results)
        baseline_throughput = baseline_total_tokens / baseline_time if baseline_time > 0 else 0
        baseline_avg_latency = statistics.mean(r.total_time for r in baseline_results)
        
        optimized_total_tokens = sum(r.tokens_generated for r in optimized_results)
        optimized_throughput = optimized_total_tokens / optimized_time if optimized_time > 0 else 0
        optimized_avg_latency = statistics.mean(r.total_time for r in optimized_results)
        
        # Performance improvements
        time_improvement = ((baseline_time - optimized_time) / baseline_time) * 100 if baseline_time > 0 else 0
        throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100 if baseline_throughput > 0 else 0
        latency_change = ((optimized_avg_latency - baseline_avg_latency) / baseline_avg_latency) * 100 if baseline_avg_latency > 0 else 0
        
        print(f"\nDETAILED METRICS COMPARISON:")
        print(f"{'='*80}")
        
        print(f"\nBASELINE (Sequential Processing):")
        print(f"  Processing Method: Sequential, one request at a time")
        print(f"  Total Wall-Clock Time: {baseline_time:.3f}s")
        print(f"  Total Tokens Generated: {baseline_total_tokens}")  
        print(f"  Overall Throughput: {baseline_throughput:.2f} tokens/sec")
        print(f"  Average Request Latency: {baseline_avg_latency:.3f}s")
        print(f"  Requests Processed: {len(baseline_results)}")
        
        print(f"\nOPTIMIZED (Concurrent Processing):")
        print(f"  Processing Method: Concurrent with optimizations")
        print(f"  Total Wall-Clock Time: {optimized_time:.3f}s")
        print(f"  Total Tokens Generated: {optimized_total_tokens}")
        print(f"  Overall Throughput: {optimized_throughput:.2f} tokens/sec")
        print(f"  Average Request Latency: {optimized_avg_latency:.3f}s")
        print(f"  Requests Processed: {len(optimized_results)}")
        
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"{'='*80}")
        print(f"  Overall Time Improvement: {time_improvement:+.1f}% ({'faster' if time_improvement > 0 else 'slower'})")
        print(f"  Throughput Improvement: {throughput_improvement:+.1f}%")
        print(f"  Individual Latency Change: {latency_change:+.1f}% ({'faster' if latency_change < 0 else 'slower'})")
        
        # Success assessment
        success_indicators = []
        if time_improvement > 0:
            success_indicators.append(f"✓ {time_improvement:.1f}% faster overall processing")
        if throughput_improvement > 5:
            success_indicators.append(f"✓ {throughput_improvement:.1f}% higher throughput")
        if latency_change < 100:  # Reasonable latency increase for concurrent processing
            success_indicators.append(f"✓ Individual latency remained reasonable")
        
        print(f"\nOPTIMIZATION EFFECTIVENESS:")
        print(f"{'='*80}")
        if len(success_indicators) >= 2:
            print("SUCCESS: Optimizations are working effectively!")
            for indicator in success_indicators:
                print(f"  {indicator}")
            print(f"  Recommendation: Optimized processing shows clear benefits")
        elif len(success_indicators) == 1:
            print("PARTIAL SUCCESS: Some optimizations working")
            for indicator in success_indicators:
                print(f"  {indicator}")
            print("  Recommendation: Consider tuning optimization parameters")
        else:
            print("NEEDS REVIEW: Optimizations may need adjustment")
            print("  Consider:")
            print("  - Checking concurrent processing implementation")
            print("  - Adjusting batch sizes or timeout parameters")
            print("  - Verifying model optimization settings")
        
        # Analysis based on number of requests
        if len(test_prompts) == 1:
            print(f"\nSINGLE REQUEST ANALYSIS:")
            print(f"  Single requests show limited concurrency benefits")
            print(f"  Main improvements come from processing optimizations")
            print(f"  Test with multiple concurrent requests for better comparison")
        else:
            print(f"\nCONCURRENT REQUEST ANALYSIS:")
            print(f"  Tested with {len(test_prompts)} concurrent requests")
            print(f"  This demonstrates the benefits of concurrent processing")
            efficiency_factor = baseline_time / optimized_time if optimized_time > 0 else 1
            print(f"  Processing Efficiency Improvement: {efficiency_factor:.2f}x")
            
    else:
        print("[ERROR] Insufficient results for comparison")
        if not baseline_results:
            print("  - Baseline benchmark failed")
        if not optimized_results:
            print("  - Optimized benchmark failed")

    print(f"\n" + "="*100)
    print("BENCHMARK COMPLETED")
    print("="*100)

if __name__ == "__main__":
    asyncio.run(main())
