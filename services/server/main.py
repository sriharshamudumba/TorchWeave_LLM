import os, tarfile, tempfile, torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="TorchWeave-LLM Server")
MODEL=None; TOK=None; DEVICE="cuda" if torch.cuda.is_available() else "cpu"

class LoadReq(BaseModel):
    artifact: str
    device: str | None = None

@app.post("/load_artifact")
def load_artifact(r: LoadReq):
    global MODEL, TOK, DEVICE
    if r.device: DEVICE = r.device
    path = r.artifact.replace("file://","")
    tmp = tempfile.mkdtemp()
    with tarfile.open(path) as tar: tar.extractall(tmp)
    TOK = AutoTokenizer.from_pretrained(tmp)
    MODEL = AutoModelForCausalLM.from_pretrained(tmp).to(DEVICE).eval()
    return {"loaded": True, "device": DEVICE}

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7

@app.post("/generate")
def generate(req: GenReq):
    """SSE streaming endpoint to generate text from the loaded model."""
    def stream():
        ids = TOK(req.prompt, return_tensors="pt").input_ids.to(DEVICE)
        past = ids.shape[1]
        with torch.no_grad():
            out = MODEL.generate(
                input_ids=ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                use_cache=True,
            )
        text = TOK.decode(out[0][past:], skip_special_tokens=True)
        for ch in text:
            yield f"data: {ch}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
