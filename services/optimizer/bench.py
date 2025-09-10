import time, torch

def _inputs(tok, bsz, prefill):
    prompts = ["Hello"] * bsz
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=prefill).input_ids

def run_bench(model, tok, bench: dict, device: str):
    res = []
    for b in bench["batch_sizes"]:
        ids = _inputs(tok, b, bench["prefill"]).to(device)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids=ids, max_new_tokens=bench["decode"], do_sample=False, use_cache=True)
        elapsed = time.time() - t0
        ttft_ms = round(elapsed*1000.0, 1)
        tps = round((b*bench["decode"])/(elapsed+1e-6), 1)
        mem_gb = 0.0
        if torch.cuda.is_available() and "cuda" in device:
            mem_gb = round(torch.cuda.max_memory_allocated()/1e9, 2)
        res.append({
            "batch": b, "prefill": bench["prefill"], "decode": bench["decode"],
            "ttft_ms": ttft_ms, "tps": tps, "vram_gb": mem_gb
        })
    best = max(res, key=lambda x: x["tps"])
    return {"by_batch": res, "best_bsz": best["batch"], "best_tps": best["tps"]}

