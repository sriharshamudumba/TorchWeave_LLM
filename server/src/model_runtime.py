import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick GPU if available, else CPU
Device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

HF_MODEL = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SEED = int(os.getenv("SEED", "42"))

_torch_rng = torch.Generator(device=Device)
_torch_rng.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load from artifact dir if optimizer staged weights
artifact_dir = os.getenv("ARTIFACT_MODEL_DIR")
if artifact_dir and os.path.exists(artifact_dir):
    model_path = artifact_dir
else:
    model_path = HF_MODEL

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE)
model.to(Device)
model.eval()


@torch.inference_mode()
def first_forward(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Initial forward to obtain logits and past for a fresh sequence batch.
    input_ids: (B, T)
    returns: logits (B, V), past_key_values
    """
    out = model(
        input_ids=input_ids.to(Device),
        attention_mask=attention_mask.to(Device),
        use_cache=True,
    )
    last_logits = out.logits[:, -1, :]
    return last_logits, out.past_key_values


@torch.inference_mode()
def next_forward(
    last_tokens: torch.Tensor, attention_mask: torch.Tensor, past_key_values
):
    """
    One-token decode step with shared past_key_values.
    last_tokens: (B, 1)
    attention_mask: (B, S+1)
    """
    out = model(
        input_ids=last_tokens.to(Device),
        attention_mask=attention_mask.to(Device),
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = out.logits[:, -1, :]
    return logits, out.past_key_values


def split_past(past_batched, batch_sizes: List[int]):
    """
    Split a batched past_key_values into per-request chunks by batch dimension.
    """
    idxs = []
    cur = 0
    for b in batch_sizes:
        idxs.append((cur, cur + b))
        cur += b
    per = []
    for (start, end) in idxs:
        sl = []
        for layer in past_batched:
            k, v = layer
            sl.append(
                (
                    k[start:end, ...].contiguous(),
                    v[start:end, ...].contiguous(),
                )
            )
        per.append(tuple(sl))
    return per


def cat_past(pasts: List[Tuple]):
    """
    Concatenate per-request pasts along batch axis.
    """
    layers = len(pasts[0])
    cat_k = []
    cat_v = []
    for li in range(layers):
        ks = [p[li][0] for p in pasts]
        vs = [p[li][1] for p in pasts]
        cat_k.append(torch.cat(ks, dim=0))
        cat_v.append(torch.cat(vs, dim=0))
    return tuple(zip(cat_k, cat_v))
