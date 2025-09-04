# TorchWeave-LLM (from scratch)
A tiny, production-style project that **optimizes** and **serves** LLMs:
- Optimizer: applies runtime/graph tweaks (flash attention flag, KV-cache config, optional quantization, torch.compile), benchmarks, and packages an artifact.
- Server: loads the artifact and exposes a streaming `/generate` API.

We’ll start with a CPU-friendly model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) so it runs everywhere. Later you can switch to GPU and 7B models.
EOF# TorchWeave-LLM (from scratch)
A tiny, production-style project that **optimizes** and **serves** LLMs:
- Optimizer: applies runtime/graph tweaks (flash attention flag, KV-cache config, optional quantization, torch.compile), benchmarks, and packages an artifact.
- Server: loads the artifact and exposes a streaming `/generate` API.

We’ll start with a CPU-friendly model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) so it runs everywhere. Later you can switch to GPU and 7B models.
# TorchWeave_LLM
