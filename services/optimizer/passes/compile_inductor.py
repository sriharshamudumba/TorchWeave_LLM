import torch

def maybe_compile(model, backend: str | None):
    """
    Wraps model with torch.compile for fused kernels (Inductor).
    """
    if backend:
        return torch.compile(model, backend=backend, mode="max-autotune")
    return model
