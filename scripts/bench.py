# scripts/bench.py
import argparse, asyncio, time, httpx

PROMPT = "You are a helpful assistant. Briefly list 5 pros of CUDA for LLM inference:"

async def run_sse(client, url, max_new=64):
    started = time.perf_counter()
    ttft = None
    tokens = 0
    prev_event = None
    async with client.stream("POST", url, json={"prompt": PROMPT, "max_new_tokens": max_new}) as r:
        async for line in r.aiter_lines():
            if not line: continue
            if line.startswith("event:"):
                prev_event = line.split(":",1)[1].strip()
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if prev_event == "ttft":
                    try: ttft = float(payload)
                    except: ttft = time.perf_counter() - started
                    prev_event = None
                else:
                    # count non-empty token chunks (rough)
                    if payload: tokens += 1
            if line.startswith("event: done"):
                break
    duration = time.perf_counter() - started
    return ttft or 0.0, tokens, duration

async def run_nobatch(client, url, max_new=64):
    started = time.perf_counter()
    r = await client.post(url, json={"prompt": PROMPT, "max_new_tokens": max_new})
    r.raise_for_status()
    duration = time.perf_counter() - started
    text = r.json().get("text", "")
    tokens = len(text.split())  # rough proxy
    return 0.0, tokens, duration

async def bench(url, mode="sse", concurrency=8, iters=50):
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        sem = asyncio.Semaphore(concurrency)
        async def one():
            async with sem:
                if mode == "sse":
                    return await run_sse(client, url)
                return await run_nobatch(client, url)
        tasks = [asyncio.create_task(one()) for _ in range(iters)]
        for t in tasks:
            results.append(await t)
    ttfts = [x[0] for x in results if x[0] > 0]
    toks = sum(x[1] for x in results)
    total = sum(x[2] for x in results)
    avg = total / max(1, len(results))
    print(f"Mode={mode} Concurrency={concurrency} Iters={iters}")
    if ttfts:
        print(f"  mean_TTFT ~= {sum(ttfts)/len(ttfts):.3f}s")
    print(f"  tokens ~= {toks}  avg_req_time ~= {avg:.3f}s  agg_tok_per_sec ~= {toks/max(1e-9,total):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sse", help="SSE URL (batched)")
    g.add_argument("--nobatch", help="No-batch URL (baseline)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if args.sse:
        asyncio.run(bench(args.sse, mode="sse", concurrency=args.concurrency, iters=args.iters))
    else:
        asyncio.run(bench(args.nobatch, mode="nobatch", concurrency=args.concurrency, iters=args.iters))
