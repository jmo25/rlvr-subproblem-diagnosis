import vllm.config as _c
_orig = _c._get_and_verify_max_len
def _patched(*args, **kwargs):
    hf_config = args[0] if args else kwargs.get("hf_config")
    if hf_config is not None:
        rs = getattr(hf_config, "rope_scaling", None)
        if rs is not None and "factor" not in rs:
            hf_config.rope_scaling = None
    return _orig(*args, **kwargs)
_c._get_and_verify_max_len = _patched
