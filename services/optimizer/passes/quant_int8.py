def maybe_quantize(model, quant_cfg: dict, device: str):
    """
    Placeholder for INT8/NF4 integration (bitsandbytes / torch.ao.quantization).
    Safe to no-op initially.
    """
    if quant_cfg.get("type","none") == "none":
        return model
    # TODO: integrate actual quant flow when ready.
    return model
