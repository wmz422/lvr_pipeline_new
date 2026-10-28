#!/usr/bin/env python3
"""Export trusted DeepSpeed or Lightning checkpoints as complete LVR bundles."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor

from lvr.tokens import register_latent_tokens


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_lightning(args, model_config, lam_state):
    """Restore omitted frozen weights and the tied LM head without random weights."""
    from lvr.models.builder import build_latent_projector

    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    saved = checkpoint["state_dict"]
    base_config = AutoConfig.from_pretrained(str(args.base_model), local_files_only=True)
    index = args.base_model / "model.safetensors.index.json"
    if index.is_file():
        base_files = [args.base_model / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    else:
        base_files = [args.base_model / "model.safetensors"]
    recovered = {}
    for path in base_files:
        recovered.update({"qwen." + name: tensor for name, tensor in load_file(path).items()})
    recovered.update({"lam." + name: tensor for name, tensor in lam_state.items()})
    projector = build_latent_projector(
        model_config["latent_projector_type"], model_config["lam_latent_dim"],
        base_config.text_config.hidden_size, model_config.get("latent_projector_mlp_hidden_dim"),
    )
    shapes = {name: tensor.shape for name, tensor in recovered.items()}
    shapes.update({"latent_projector." + name: tensor.shape for name, tensor in projector.state_dict().items()})
    del projector
    embedding = "qwen.model.language_model.embed_tokens.weight"
    lm_head = "qwen.lm_head.weight"
    vocab_size = shapes[embedding][0]
    if model_config.get("add_latent_special_tokens", True):
        processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True)
        register_latent_tokens(processor.tokenizer, model_config.get("latent_pad_token", "<abs_vis_token_pad>"))
        vocab_size = len(processor.tokenizer)
    tied = bool(base_config.text_config.tie_word_embeddings)
    if tied:
        shapes[lm_head] = shapes[embedding]
    required = set()
    for flag, prefix in [("train_qwen_lm", "qwen.model.language_model."),
                         ("train_latent_projector", "latent_projector."), ("train_lam", "lam.")]:
        if model_config.get(flag):
            required.update(name for name in shapes if name.startswith(prefix))
    if model_config.get("train_qwen_lm_head"):
        required.add(embedding if tied else lm_head)
    if model_config.get("latent_head") or model_config.get("train_latent_head"):
        raise ValueError("Lightning export does not support a separate latent head")
    missing = required - saved.keys()
    unexpected = saved.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(f"Invalid saved parameters: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}")
    for name, tensor in saved.items():
        expected = shapes[name]
        if name in {embedding, lm_head}:
            valid_shape = tensor.ndim == 2 and tensor.shape[1:] == expected[1:] and tensor.shape[0] == vocab_size
        else:
            valid_shape = tensor.shape == expected
        if not valid_shape:
            raise ValueError(f"Shape mismatch for {name}: {tensor.shape} versus {expected}")
    recovered.update(saved)
    if tied:
        if lm_head in saved and not torch.equal(saved[lm_head], recovered[embedding]):
            raise ValueError("Saved LM head differs from the tied embedding")
        recovered[lm_head] = recovered[embedding]
    if set(recovered) != set(shapes):
        raise ValueError(f"Unreconstructed parameters: {sorted(set(shapes) - recovered.keys())[:5]}")
    provenance = {"format": "lightning", "epoch": checkpoint.get("epoch"),
                  "global_step": checkpoint.get("global_step"), "saved_tensors": len(saved),
                  "complete_tensors": len(recovered), "tied_lm_head_restored": tied,
                  "restored_frozen_tensors": len(set(recovered) - saved.keys() - ({lm_head} if tied else set())),
                  "base_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in base_files}}
    return recovered, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; choose a new directory to preserve existing exports.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    config_document = yaml.safe_load(args.train_config.read_text())
    model_config = config_document.get("fit", config_document)["model"]
    original = torch.load(args.lam_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    lam_state = {k.removeprefix("lam."): v for k, v in original.get("state_dict", original).items()}
    print("Reconstructing all parameters, including frozen parameters.", flush=True)
    if args.checkpoint.is_file():
        recovered, reconstruction = recover_lightning(args, model_config, lam_state)
    else:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        recovered = get_fp32_state_dict_from_zero_checkpoint(
            str(args.checkpoint), exclude_frozen_parameters=False, lazy_mode=True
        )
        reconstruction = {"format": "deepspeed_zero"}
    dtype = getattr(torch, args.dtype)
    state = {}
    for name, value in recovered.items():
        value = value.contiguous()
        state[name] = value.to(dtype=dtype if value.is_floating_point() else value.dtype).clone()
    del recovered
    for prefix in ("qwen.", "lam.", "latent_projector."):
        if not any(key.startswith(prefix) for key in state):
            raise RuntimeError(f"Checkpoint is missing {prefix}")

    shards = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size=args.max_shard_size
    )
    for filename, names in shards.filename_to_tensors.items():
        print(f"Writing and checking {filename}", flush=True)
        part = {name: state[name] for name in names}
        save_file(part, args.output / filename, metadata={"format": "pt"})
        restored = load_file(args.output / filename)
        if set(restored) != set(part) or any(not torch.equal(restored[k], part[k]) for k in part):
            raise RuntimeError(f"Tensor round-trip mismatch: {filename}")
        del restored
    if shards.is_sharded:
        (args.output / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": shards.metadata, "weight_map": shards.tensor_to_filename
        }, indent=2) + "\n")

    qwen_dir = args.output / "qwen"
    processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True)
    register_latent_tokens(processor.tokenizer)
    processor.save_pretrained(qwen_dir)
    config = AutoConfig.from_pretrained(str(args.base_model), local_files_only=True)
    embedding_key = "qwen.model.language_model.embed_tokens.weight"
    config.text_config.vocab_size = state[embedding_key].shape[0]
    config.save_pretrained(qwen_dir)

    print("Exporting standalone LAM without optimizer state.", flush=True)
    # The source is explicitly supplied by the owner; Lightning metadata may
    # contain objects outside torch.load's weights_only allowlist.
    lam_state = {k.removeprefix("lam."): v.contiguous().clone() for k, v in lam_state.items()}
    model_lam = {k.removeprefix("lam."): v for k, v in state.items() if k.startswith("lam.")}
    if set(lam_state) != set(model_lam):
        raise RuntimeError("Original LAM and checkpoint LAM keys differ.")
    mismatches = [k for k in lam_state if not torch.equal(lam_state[k].to(model_lam[k].dtype), model_lam[k])]
    if mismatches:
        raise RuntimeError(f"Original LAM differs from the frozen checkpoint LAM: {mismatches[:5]}")
    lam_dir = args.output / "lam"
    lam_dir.mkdir()
    lam_config = {k: v for k, v in original.get("hyper_parameters", {}).items()
                  if k != "vision_model_path" and isinstance(v, (str, int, float, bool, type(None)))}
    torch.save({"state_dict": lam_state, "hyper_parameters": lam_config,
                "source_epoch": original.get("epoch"), "source_global_step": original.get("global_step"),
                "weights_only": True}, lam_dir / "lam.ckpt")
    restored_lam = torch.load(lam_dir / "lam.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    if any(not torch.equal(lam_state[k], restored_lam[k]) for k in lam_state):
        raise RuntimeError("LAM round-trip mismatch")
    del restored_lam, original, lam_state

    model_config.update(qwen_model_name_or_path="qwen", lam_vision_model_path="qwen",
                        lam_checkpoint_path=None, init_checkpoint_path=".",
                        initialize_from_config=True, save_trainable_only=False,
                        qwen_gradient_checkpointing=False, qwen_torch_dtype=args.dtype,
                        generation_output_path="runs/predictions.jsonl")
    for name in ["train_qwen_lm", "train_qwen_lm_head", "train_latent_projector", "train_latent_head", "train_lam"]:
        model_config[name] = False
    (args.output / "lvr_config.json").write_text(json.dumps({
        "format_version": 1, "architecture": "LatentVLM", "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "source_checkpoint": args.checkpoint.name, "model": model_config,
    }, indent=2) + "\n")
    manifest = {
        "source_checkpoint": args.checkpoint.name, "dtype": args.dtype,
        "tensors": len(state), "parameters": sum(v.numel() for v in state.values()),
        "tensor_round_trip": "exact", "lam_round_trip": "exact",
        "lam_matches_model_at_export_dtype": True, "optimizer_state_included": False,
        "reconstruction": reconstruction,
        "source_files": {str(p.relative_to(args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent)):
                         {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in (sorted(args.checkpoint.rglob("*")) if args.checkpoint.is_dir() else [args.checkpoint]) if p.is_file()},
        "source_lam": {"filename": args.lam_checkpoint.name, "sha256": sha256(args.lam_checkpoint)},
        "files": {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(args.output.rglob("*")) if p.is_file()},
    }
    (args.output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["dtype", "tensors", "parameters", "tensor_round_trip", "lam_round_trip"]}), flush=True)


if __name__ == "__main__":
    main()
