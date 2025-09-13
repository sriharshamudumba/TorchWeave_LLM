def apply_kv_policy(model, cfg: dict):
    """
    Stores KV cache policy metadata on model.config for the server to read.
    """
    if not hasattr(model.config, "torchweave"):
        model.config.torchweave = {}
    model.config.torchweave["kv_cache"] = cfg
    return model
