"""Load a local exported LVR model without DeepSpeed or base-model weights."""

import json
from pathlib import Path


def load_bundle(model_dir, device="cuda:0", max_pixels=None):
    from lvr.eval.runner import _build_processor
    from lvr.models import LatentVLM

    root = Path(model_dir).expanduser().resolve()
    config = json.loads((root / "lvr_config.json").read_text())
    if config.get("format_version") != 1:
        raise ValueError("Unsupported LVR bundle format")
    kwargs = dict(config["model"])
    for key in ("qwen_model_name_or_path", "lam_vision_model_path", "lam_checkpoint_path", "init_checkpoint_path"):
        if kwargs.get(key) is not None:
            kwargs[key] = str(root / kwargs[key])
    model = LatentVLM(**kwargs).to(device).eval()
    processor = _build_processor(kwargs["qwen_model_name_or_path"], kwargs["latent_pad_token"], True, max_pixels)
    return model, processor
