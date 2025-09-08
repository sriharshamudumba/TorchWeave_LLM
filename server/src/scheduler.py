import asyncio
if tok == EOS_TOKEN_ID or r.steps >= r.max_new_tokens:
finished.append(r)
# Remove finished
if finished:
self.active = [r for r in self.active if r not in finished]
for r in finished:
r.done.set()


async def _emit(self, r: Request, token_id: int):
text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
await r.out_queue.put(text)
if not r.ttft_reported:
# first token just arrived
await r.out_queue.put("__TTFT__")
r.ttft_reported = True


def _sample(self, logits: torch.Tensor, temperatures: List[float], top_ks: List[int], top_ps: List[float]) -> List[int]:
# vectorized sampling per row
probs = []
for i in range(logits.size(0)):
logit = logits[i]
temp = max(1e-5, float(temperatures[i]))
logit = logit / temp
# top-k
k = int(top_ks[i])
if k and k < logit.size(-1):
vals, idx = torch.topk(logit, k)
mask = torch.full_like(logit, float('-inf'))
mask[idx] = vals
logit = mask
# top-p
p = float(top_ps[i])
if 0 < p < 1.0:
sorted_logits, sorted_idx = torch.sort(logit, descending=True)
cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
cutoff = cumulative_probs > p
if cutoff.any():
first = torch.argmax(cutoff.int()).item()
keep = sorted_idx[: first + 1]
mask = torch.full_like(logit, float('-inf'))
mask[keep] = logit[keep]
logit = mask
prob = torch.softmax(logit, dim=-1)
probs.append(prob)
probs = torch.stack(probs, dim=0)
next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
return [int(x.item()) for x in next_ids]