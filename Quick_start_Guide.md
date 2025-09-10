# TorchWeave LLM - Quick Start Guide

## Try it in 3 Minutes!

### Step 1: Clone & Setup
```bash
git clone https://github.com/sriharshamudumba/TorchWeave_LLM.git
cd TorchWeave_LLM
cp .env.example .env
```

### Step 2: Start the Server
```bash
# CPU version (works on any machine)
docker compose up -d --build

# GPU version (if you have NVIDIA GPU + Docker toolkit)
docker compose --profile gpu up -d --build
```

### Step 3: Test It!
```bash
# Health check
curl http://localhost:8000/health

# Generate text with streaming
curl -N -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain machine learning in simple terms","max_new_tokens":100}'
```

## What You'll See

- **Real-time streaming**: Tokens appear as they're generated
- **TTFT metrics**: See how fast first token arrives
- **High throughput**: 2-5x improvement with concurrent requests

## Common Issues

**Port already in use?**
```bash
# Change port in docker-compose.yml or stop other services
docker compose down && docker compose up -d
```

**GPU not detected?**
```bash
# Check NVIDIA setup
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## Benchmark It Yourself

```bash
pip install httpx
python scripts/bench.py --concurrency 4 --iters 10 --sse http://localhost:8000/v1/generate
```

## Need Help?

- Check logs: `docker compose logs -f server`
- Open an issue: [GitHub Issues](https://github.com/sriharshamudumba/TorchWeave_LLM/issues)
- Read full docs: [README.md](README.md)

---
**Expected setup time: 2-3 minutes** | **Disk space needed: ~2GB**
