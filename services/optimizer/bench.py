import torch, time

def _inputs(tok, batch, prefill):
    prompts = ["Hello"] * batch
    ids = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=prefill).input_ids
    return ids

def run_bench(model, tok, cfg, device):
    results = []
    for bsz in cfg.batch_sizes:
        inp = _inputs(tok, bsz, cfg.prefill).to(device)
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids=inp, max_new_tokens=cfg.decode, do_sample=False, use_cache=True)
        elapsed = time.time() - t0
        ttft_ms = elapsed * 1000.0  # coarse but consistent for our starter
        tps = (bsz * cfg.decode) / (elapsed + 1e-6)
        mem = torch.cuda.max_memory_allocated()/1e9 if (torch.cuda.is_available() and device.startswith("cuda")) else 0.0
        results.append({"batch": bsz, "ttft_ms": round(ttft_ms,1), "tps": round(tps,1), "mem_gb": round(mem,2)})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    best = max(results, key=lambda r: r["tps"])
    return {"by_batch": results, "best_tps": best["tps"], "best_bsz": best["batch"]}
