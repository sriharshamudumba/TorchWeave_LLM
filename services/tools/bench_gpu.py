import time, json, argparse, requests

def optimize(base_url, payload):
    r = requests.post(f"{base_url}/optimize", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()

def load(server_url, art, device):
    r = requests.post(f"{server_url}/load_artifact", json={"artifact": art, "device": device}, timeout=600)
    r.raise_for_status()
    return r.json()

ap = argparse.ArgumentParser()
ap.add_argument("--opt", default="http://localhost:8081")
ap.add_argument("--srv", default="http://localhost:8082")
ap.add_argument("--hf",  default="mistralai/Mistral-7B-Instruct-v0.2")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

payload = {
  "hf_id": args.hf, "precision":"bf16", "use_flash": True,
  "compile_backend":"inductor", "device": args.device,
  "bench":{"batch_sizes":[1,2,4],"prefill":512,"decode":64}
}
t0 = time.time()
out = optimize(args.opt, payload)
t1 = time.time()
art = out["artifact"]["path"]
before = out["metrics"]["before"]
after  = out["metrics"]["after"]
load(args.srv, art, args.device)

print(json.dumps({
  "model": args.hf, "device": args.device,
  "artifact": art, "optimize_sec": round(t1 - t0, 1),
  "before": before, "after": after
}, indent=2))
