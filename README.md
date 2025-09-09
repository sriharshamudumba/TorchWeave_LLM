# TorchWeave_LLM — Continuous‑Batching LLM Server (CUDA‑ready, SSE streaming)

FastAPI server with an **async continuous‑batching scheduler** that merges live requests into shared decode steps.  
Per‑request **KV‑cache** + attention masks, **SSE token streaming**, and an **optimizer sidecar** that stages model
artifacts into a shared volume for reproducible builds and future scale‑out (ECS/Kubernetes‑friendly).

---

## ✨ Features

- **Continuous batching**: merges concurrent requests into shared decode steps → **2–5× higher throughput** under load.
- **Per‑request KV‑cache** with attention masks; shared decode steps across the active batch.
- **SSE token streaming** (`/v1/generate`) with an **`event: ttft`** (time‑to‑first‑token).
- **Baseline** no‑batch path (`/v1/generate_nobatch`) for apples‑to‑apples comparison.
- **CUDA acceleration** (fp16) when available; **CPU fallback** out of the box.
- **Optimizer sidecar** pre‑pulls the model into `/artifacts/model` for **fast, reproducible** startup.
- **One‑command up** with Docker Compose v2.

---

## 📦 Repo layout

```text
TorchWeave_LLM/
├─ docker-compose.yml
├─ .env.example
├─ server/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ src/
│     ├─ __init__.py
│     ├─ server.py          # FastAPI + endpoints (SSE + baseline)
│     ├─ scheduler.py       # async continuous-batching scheduler
│     └─ model_runtime.py   # tokenizer/model load, decode steps, KV utils
├─ optimizer/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ src/
│     └─ optimizer.py       # stages HF model into /artifacts/model
└─ scripts/
   └─ bench.py              # TTFT & throughput benchmark
```

> If you still have the old `services/server` and `services/optimizer` folders, they’ve been replaced by the structure above.

---

## 🧰 Requirements

- Docker **27+** and Compose v2 (`docker compose version`)
- (Optional, for GPU) Working **NVIDIA driver** on the host (`nvidia-smi`)
- (Optional, for GPU) **NVIDIA Container Toolkit** so containers can access the GPU

### Fix Docker permissions (if needed)

If you’ve ever seen `permission denied` on `/var/run/docker.sock`:

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
```

---

## ⚙️ Environment config

Copy and customize:

```bash
cp .env.example .env
```

Key variables (defaults are sensible):

```ini
HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0  # small, chat-tuned model
MAX_NEW_TOKENS=128
TEMPERATURE=0.7
TOP_K=0
TOP_P=0.9
SEED=42
SCHEDULE_TICK_MS=15    # lower → better TTFT, slightly more scheduler overhead
MAX_BATCH=16           # raise for throughput if you have VRAM
# Artifacts (shared volume between server and optimizer)
ARTIFACT_DIR=/artifacts
ARTIFACT_MODEL_DIR=/artifacts/model
```

> Tip: After changing `.env`, run `docker compose restart server` to reload settings.

---

## 🚀 Quickstart (CPU)

Works everywhere; you can add GPU later.

```bash
docker compose up -d --build
```

Verify:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model
docker compose logs -f --tail=200 optimizer   # should show “[optimizer] done”
docker compose exec server bash -lc 'ls -lah /artifacts/model | head'
```

If the server started **before** the optimizer finished, do:

```bash
docker compose restart server
```

---

## ⚡ Enable CUDA (Ubuntu 24.04)

**Only if you want GPU.** The server automatically falls back to CPU otherwise.

1) **Install NVIDIA Container Toolkit (Ubuntu 24.04)**

```bash
# Pre-reqs and keyring dir
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings

# Import NVIDIA key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg

# Add the correct repo list for your distro (should print: ubuntu24.04)
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
echo "$distribution"   # expect: ubuntu24.04

curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Install toolkit and wire Docker
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2) **Sanity check GPU inside a container**:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

3) **Run TorchWeave with GPU**  
If your compose uses a `gpu` profile:

```bash
docker compose --profile gpu up -d --build
```

If your `docker-compose.yml` has `deploy:` / `gpus:` keys directly under `server`, just run:

```bash
docker compose up -d --build
```

---

## 🔌 Endpoints

- `GET /health` → `{"status":"ok"}`  
- `GET /model`  → `{ "vocab_size": ..., "eos": "..." }`

- `POST /v1/generate` (**SSE streaming**)  
  Request body:
  ```json
  {
    "prompt": "Explain compute-in-memory for edge AI in 3 sentences.",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_k": 0,
    "top_p": 0.9
  }
  ```
  Behavior:
  - Streams `data: <token_piece>` lines
  - Emits `event: ttft` once for TTFT (seconds)
  - Finishes with `event: done`

- `POST /v1/generate_nobatch` (**baseline**)  
  Returns `{ "text": "..." }` (no streaming, no batching)

### Curl examples

```bash
# Streaming (SSE)
curl -N -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain compute-in-memory for edge AI in 3 sentences.","max_new_tokens":64}'

# Baseline (no batching)
curl -s -X POST http://localhost:8000/v1/generate_nobatch \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain compute-in-memory for edge AI in 3 sentences.","max_new_tokens":64}' | jq
```

---

## 📈 Benchmark (TTFT & throughput)

Use the included script to compare **batched SSE** vs. **no‑batch**:

```bash
# Continuous batching (SSE), concurrent load
python scripts/bench.py --concurrency 8 --iters 24 --sse http://localhost:8000/v1/generate

# Baseline (no batching)
python scripts/bench.py --concurrency 1 --iters 24 --nobatch http://localhost:8000/v1/generate_nobatch
```

**Report:** mean TTFT (from SSE `ttft`), total tokens, avg request time, and **aggregate tokens/sec**.  
You should see **~2–5× higher throughput** under concurrent load with batching.

**Tuning tips:**
- Lower `SCHEDULE_TICK_MS` (e.g., `10`) → better TTFT
- Increase `MAX_BATCH` if you have VRAM headroom for higher throughput
- Use shorter prompts to reduce initial compute

---

## 🧠 How it works

- **Optimizer** (`optimizer/src/optimizer.py`) snapshots the Hugging Face model to the shared **`/artifacts/model`** directory.
- **Server** loads the model (CUDA fp16 if available; otherwise CPU fp32) and exposes:
  - `/v1/generate` (SSE streaming, batched)
  - `/v1/generate_nobatch` (baseline)
- **Scheduler** (`server/src/scheduler.py`):
  1. Accepts live requests and tokenizes prompts
  2. Batches newcomers for **first forward** to build their **KV cache**
  3. On each tick, executes a **shared decode step** for all active requests (one token each), updates their per‑request KV
  4. Streams tokens to clients via SSE

---

## 🔧 Configuration (via `.env`)

```ini
HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MAX_NEW_TOKENS=128
TEMPERATURE=0.7
TOP_K=0
TOP_P=0.9
SEED=42
SCHEDULE_TICK_MS=15
MAX_BATCH=16
# Artifacts (shared volume between server and optimizer)
ARTIFACT_DIR=/artifacts
ARTIFACT_MODEL_DIR=/artifacts/model
```

---

## 🧪 Dev & Ops

Validate Compose:

```bash
docker compose config
```

Tail logs:

```bash
docker compose logs -f --tail=200 server
docker compose logs -f --tail=200 optimizer
```

Restart server after optimizer finished or after changing `.env`:

```bash
docker compose restart server
```

---

## 🧹 Git hygiene

Keep secrets out of git. Recommended `.gitignore` bits:

```gitignore
# Python
__pycache__/
*.pyc

# Envs
.venv/
llm_env/
.env
!.env.example     # keep the example tracked

# Docker volumes / artifacts
artifacts/
db_data/
redis_data/
minio_data/

# OS/editor
.DS_Store
*.swp
```

Stage changes cleanly:

```bash
git add -A
git commit -m "Continuous batching server + optimizer + compose + bench"
git push origin main
```

---

## 🧯 Troubleshooting

**Docker socket permission denied**  
```bash
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
```

**NVIDIA toolkit not found / apt can’t locate**  
Ensure the repo is added for **ubuntu24.04** (not `noble`). Re‑add key + list (see CUDA section), then:
```bash
sudo apt-get update
apt-cache policy nvidia-container-toolkit
```

**GPU not visible in containers**  
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
If that works, your GPU is wired. Then:
```bash
docker compose --profile gpu up -d --build
```

**Model not staged / server can’t find model**  
- Check optimizer logs → should end with `[optimizer] done`  
- Confirm files exist:
  ```bash
  docker compose exec server bash -lc 'ls -lah /artifacts/model | head'
  ```
- Restart server:
  ```bash
  docker compose restart server
  ```

**High TTFT**  
Lower `SCHEDULE_TICK_MS` (e.g., `10`), shorten prompts, and/or send a warm‑up request after startup.

**OOM on GPU**  
Lower `MAX_BATCH`, or switch to a smaller model.



Happy weaving 🧵🤖
