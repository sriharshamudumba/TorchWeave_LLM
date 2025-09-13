import torch

def enable_flash(model):
    """
    Enables flash attention variants when supported.
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass
    if hasattr(model.config, "use_flash_attention_2"):
        model.config.use_flash_attention_2 = True
    return model
