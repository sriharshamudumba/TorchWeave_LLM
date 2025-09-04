import torch

def maybe_compile(model, backend):
    if backend:
        try:
            model = torch.compile(model, backend=backend, mode="max-autotune")
        except Exception as e:
            print(f"[compile] disabled due to: {e}")
    return model
