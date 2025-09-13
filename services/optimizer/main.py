import torch
from fastapi import FastAPI
from pydantic import BaseModel
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

class BenchCfg(BaseModel):
    batch_sizes: list[int] = [1,2]
    prefill: int = 256
    decode: int = 64

class QuantCfg(BaseModel):
    type: str = "none"   # "none" | "int8" | "nf4" (stubbed)

class OptReq(BaseModel):
    hf_id: str
    precision: str = "bf16"    # "fp16"|"bf16"
    use_flash: bool = True
    kv_cache: KvCfg = KvCfg()
    quant: QuantCfg = QuantCfg()
    compile_backend: str | None = "inductor"
    device: str = "cpu"        # "cpu"|"cuda"
    bench: BenchCfg = BenchCfg()

@app.post("/optimize")
def optimize(r: OptReq):
    dtype = torch.bfloat16 if r.precision=="bf16" else torch.float16
    tok = AutoModelForCausalLM  # type alias just to ensure download first
    tokenizer = AutoTokenizer.from_pretrained(r.hf_id)
    model = AutoModelForCausalLM.from_pretrained(r.hf_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(r.device).eval()

    before = run_bench(model, tokenizer, r.bench.model_dump(), r.device)

    if r.use_flash: enable_flash(model)
    apply_kv_policy(model, r.kv_cache.model_dump())
    model = maybe_quantize(model, r.quant.model_dump(), r.device)
    model = maybe_compile(model, r.compile_backend)

    after = run_bench(model, tokenizer, r.bench.model_dump(), r.device)

    meta = {
        "hf_id": r.hf_id, "precision": r.precision, "use_flash": r.use_flash,
        "kv_cache": r.kv_cache.model_dump(), "quant": r.quant.model_dump(),
        "compile_backend": r.compile_backend, "metrics": {"before": before, "after": after}
    }
    artifact = pack_artifact(model, tokenizer, meta)
    return {"artifact": {"path": artifact}, "metrics": {"before": before, "after": after}}
