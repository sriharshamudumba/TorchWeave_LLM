def enable_flash(model):
    # Toggle Flash Attention v2 flag if model supports it.
    if hasattr(model.config, "use_flash_attention_2"):
        model.config.use_flash_attention_2 = True
    return model
