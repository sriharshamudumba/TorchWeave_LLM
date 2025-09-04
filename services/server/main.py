from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch, tarfile, tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="TorchWeave-LLM Server")
MODEL=None; TOK=None; DEVICE="cpu"

class LoadReq(BaseModel):
    artifact: str  # file:// path (or s3:// in future)
    device: str | None = None

@app.post("/load_artifact")
def load_artifact(r: LoadReq):
    global MODEL, TOK, DEVICE
    if r.device: DEVICE = r.device
    path = r.artifact.replace("file://","")
    tmp = tempfile.mkdtemp()
    with tarfile.open(path) as tar: tar.extractall(tmp)
    TOK = AutoTokenizer.from_pretrained(tmp)
    MODEL = AutoModelForCausalLM.from_pretrained(tmp, device_map=None).to(DEVICE).eval()
    return {"loaded": True, "device": DEVICE}

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7

@app.post("/generate")
def generate(req: GenReq):
    def stream():
        ids = TOK(req.prompt, return_tensors="pt").input_ids.to(DEVICE)
        past_len = ids.shape[1]
        with torch.no_grad():
            out_ids = MODEL.generate(
                input_ids=ids, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, do_sample=req.temperature>0, use_cache=True
            )
        text = TOK.decode(out_ids[0][past_len:], skip_special_tokens=True)
        for ch in text:
            yield f"data: {ch}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
