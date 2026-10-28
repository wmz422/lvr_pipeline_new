"""Create an UNTRAINED original+crops teacher for resource/flow preflight only.

Run in a separate CPU process before launching training workers. No Trainer,
forward/backward, optimizer, or dataset is needed. Imports are lazy for dry-run.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def initialization_spec(base_model):
    """Pin structure/seed and detect changed local base files without reading GBs."""
    base_model = Path(base_model).resolve()
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Missing base model config: {base_model / 'config.json'}")
    files = sorted({path for pattern in ("*.safetensors", "*.bin", "*.json")
                    for path in base_model.glob(pattern) if path.is_file()})
    if not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise FileNotFoundError(f"No local pretrained weights in {base_model}")
    inputs = []
    for path in files:
        stat = path.stat()
        item = dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if path.suffix == ".json":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.append(item)
    parameters = dict(encoder_model_name_or_path=str(base_model), qa_model_name_or_path=str(base_model),
                      num_latents=8, latent_dim=512, gaussian=False, train_qa=True,
                      encoder_type="qwen", encoder_uses_question=False, encoder_includes_original=True,
                      qformer_use_geometry=False, description_image_mode="none", description_weight=1.0,
                      qa_loss_weight=1.0, kl_weight=0.0, gradient_checkpointing=False, seed=42)
    return dict(schema_version=1, kind="initial_teacher_preflight_only", teacher_trained=False,
                base_model=str(base_model), model_parameters=parameters, base_files=inputs)


def prepare_initial_teacher(base_model, output):
    """Atomically create/reuse this initializer's checkpoint; never replace others."""
    spec = initialization_spec(base_model)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch

        if output.exists():
            if not output.is_file():
                raise FileExistsError(f"Not a checkpoint file: {output}; use a new output")
            checkpoint = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
            if (checkpoint.get("preflight_initialization") != spec or checkpoint.get("global_step") != 0
                    or not checkpoint.get("state_dict")):
                raise ValueError(f"Existing initial teacher differs/cannot be verified: {output}; use a new output")
            print(f"[initial-teacher] Reusing UNTRAINED checkpoint: {output}", flush=True)
            return spec

        from lvr.models.visual_bottleneck import VisualBottleneck

        print(f"[initial-teacher] Loading Qwen base on CPU; seed=42, no teacher training: {spec['base_model']}",
              flush=True)
        # The model seeds its interface modules internally. Also make any fallback
        # base initialization deterministic while preserving the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(42)
            model = VisualBottleneck(**spec["model_parameters"]).requires_grad_(False).eval()
        checkpoint = dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                          global_step=0, epoch=0, preflight_initialization=spec)
        # Keep the constructor's FP32 language/head and BF16 frozen weight dtypes;
        # no optimizer allocation, precision compression, or tensor cloning.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                print(f"[initial-teacher] Saving initialization checkpoint: {output}", flush=True)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        print(f"[initial-teacher] Ready: {output}; global_step=0, NOT a trained teacher", flush=True)
        return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(initialization_spec(args.base_model), indent=2))
        print(f"Would write/reuse UNTRAINED teacher: {args.output.resolve()}")
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Initialize once, not with torchrun; use a separate CPU process before training")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    prepare_initial_teacher(args.base_model, args.output)


if __name__ == "__main__":
    main()
