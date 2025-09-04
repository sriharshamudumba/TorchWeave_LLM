from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from passes.flash_attn import enable_flash
from passes.kv_cache import apply_kv_policy
from passes.compile_inductor import maybe_compile
from passes.quant_int8 import maybe_quantize
from bench import run_bench
from artifact import pack_artifact

app = FastAPI(title="TorchWeave-LLM Optimizer")

class KvCfg(BaseModel):
    type: str = "paged"
    block_size: int = 128
    max_length: int = 4096

class QuantCfg(BaseModel):
    type: str = "none"  # "int8" | "nf4"

class BenchCfg(BaseModel):
    batch_sizes: list[int] = [1,2]
    prefill: int = 128
    decode: int = 32

class OptReq(BaseModel):
    hf_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    precision: str = "bf16"    # "fp16"|"bf16"
    use_flash: bool = False
    kv_cache: KvCfg = KvCfg()
    quant: QuantCfg = QuantCfg()
    compile_backend: str | None = None
    device: str = "cpu"
    bench: BenchCfg = BenchCfg()

@app.post("/optimize")
def optimize(r: OptReq):
    dtype = torch.bfloat16 if r.precision=="bf16" else torch.float16
    device = r.device

    tok = AutoTokenizer.from_pretrained(r.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        r.hf_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    before = run_bench(model, tok, r.bench, device)

    if r.use_flash:
        enable_flash(model)
    apply_kv_policy(model, r.kv_cache)
    model = maybe_quantize(model, r.quant, device)
    model = maybe_compile(model, r.compile_backend)

    after = run_bench(model, tok, r.bench, device)

    artifact_path = pack_artifact(model, tok, {
        "hf_id": r.hf_id, "dtype": str(dtype), "use_flash": r.use_flash,
        "kv": r.kv_cache.model_dump(), "quant": r.quant.model_dump(),
        "compile_backend": r.compile_backend,
        "metrics": {"before": before, "after": after}
    })

    return {"artifact": {"path": artifact_path}, "metrics": {"before": before, "after": after}}
