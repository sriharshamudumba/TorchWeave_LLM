import asyncio
async def generate_sse(body: GenIn):
"""SSE streaming endpoint using the continuous batcher."""
rid = str(uuid.uuid4())
req = Request(
req_id=rid,
prompt=body.prompt,
max_new_tokens=body.max_new_tokens or 128,
temperature=body.temperature or 0.7,
top_k=body.top_k or 0,
top_p=body.top_p or 0.9,
)


start = time.perf_counter()
await BATCHER.submit(req)


async def event_stream() -> AsyncGenerator[bytes, None]:
ttft = None
while True:
if req.done.is_set() and req.out_queue.empty():
break
try:
chunk = await asyncio.wait_for(req.out_queue.get(), timeout=0.25)
except asyncio.TimeoutError:
continue
if chunk == "__TTFT__":
ttft = time.perf_counter() - start
yield f"event: ttft
data: {ttft:.6f}


".encode()
continue
yield f"data: {chunk}


".encode()
yield b"event: done
data:


"


return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/generate_nobatch")
async def generate_nobatch(body: GenIn):
"""Baseline: single-request greedy-ish decode without merging steps. Returns the full text (no SSE)."""
from .model_runtime import first_forward, next_forward
import torch


enc = tokenizer(body.prompt, return_tensors="pt")
logits, past = first_forward(enc.input_ids, enc.attention_mask)
# simple temperature + sample step-by-step
temp = max(1e-5, body.temperature or 0.7)
out_ids = []
for _ in range(body.max_new_tokens or 128):
logits = logits / temp
prob = torch.softmax(logits, dim=-1)
next_id = int(torch.multinomial(prob, num_samples=1)[0].item())
out_ids.append(next_id)
if next_id == tokenizer.eos_token_id:
break
logits, past = next_forward(torch.tensor([[next_id]]), enc.attention_mask, past)
text = tokenizer.decode(out_ids, skip_special_tokens=True)
return JSONResponse({"text": text})