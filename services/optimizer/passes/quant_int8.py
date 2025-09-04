import torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic

def maybe_quantize(model, quant_cfg, device):
    if quant_cfg.type == "none":
        return model
    if quant_cfg.type == "int8":
        try:
            return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            print(f"[quant] fallback (kept fp) due to: {e}")
            return model
    # Placeholder for 4-bit quant; requires bitsandbytes/AutoGPTQ paths.
    return model
