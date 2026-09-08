#!/usr/bin/env python3
"""Check standalone LAM inference against the LAM stored in the LVR bundle."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--question-image", required=True)
    parser.add_argument("--auxiliary-image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor
    from lam.data import LAMCollator
    from lvr.models.builder import build_lam
    torch.set_num_threads(8)
    root = args.model_dir
    config = json.loads((root / "lvr_config.json").read_text())["model"]
    lam = build_lam(str(root / "lam/lam.ckpt"), str(root / "qwen"),
                    config["lam_model_dim"], config["lam_latent_dim"], config["lam_patch_size"],
                    config["lam_enc_blocks"], config["lam_dec_blocks"], config["lam_num_heads"],
                    config["lam_num_latent"]).to(args.device, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(root / "qwen", local_files_only=True)
    collator = LAMCollator(image_root=".", image_processor=processor.image_processor, image_size=512)
    batch = collator([{"question_image": args.question_image, "auxiliary_image": args.auxiliary_image}])
    batch = {k: v.to(args.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in batch.items()}
    with torch.inference_mode():
        standalone = lam(batch)["z_mu"].clone()
    bundled = {}
    for shard in sorted(root.glob("model*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("lam."):
                    bundled[key.removeprefix("lam.")] = handle.get_tensor(key)
    lam.load_state_dict(bundled, strict=True)
    with torch.inference_mode():
        reference = lam(batch)["z_mu"]
    exact = torch.equal(standalone, reference)
    finite = bool(torch.isfinite(standalone).all())
    result = {"strict_load": True, "latent_shape": list(standalone.shape), "all_finite": finite,
              "matches_bundled_lam": exact,
              "latent_sha256": hashlib.sha256(standalone.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not exact or not finite:
        raise RuntimeError("Standalone LAM verification failed")


if __name__ == "__main__":
    main()
