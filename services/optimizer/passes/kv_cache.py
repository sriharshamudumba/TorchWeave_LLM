def apply_kv_policy(model, cfg):
    # Store metadata only; real paged cache needs custom kernels.
    if not hasattr(model.config, "torchweave"):
        model.config.torchweave = {}
    model.config.torchweave["kv_cache"] = cfg.model_dump()
    return model
