#!/usr/bin/env python3
"""Run all five latent conditions using existing Qwen3-4B or Qwen2.5-7B weights."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import (HERE, ROOT, PROJECT, COUNTS, FAMILIES, MODES, bind_json, digest,
                    item_path, latent_source, load_data, model_item, required_latent_rows,
                    qa_identity, select_rows, sha256, validate_family, write_json)


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    before = path.stat()
    value = {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
             "sha256": sha256(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Checkpoint changed while reading: {path}")
    return value


def make_config(args, selected):
    model_files = {p.name: sha256(p) for p in sorted(args.base_model.iterdir())
                   if p.is_file() and p.suffix in (".json", ".jinja", ".txt", ".model")}
    if not list(args.base_model.glob("*.safetensors")):
        raise ValueError("Local base model safetensors are required")
    code = [*HERE.glob("*.py"), *sorted((PROJECT / "src/lvr").rglob("*.py")),
            ROOT / "paper_experiments/2026-10-23-unified-benchmark/configs/answer_parsing.json"]
    return {"schema": "latent-five-mode-run-v1", "family": args.model_family,
            "suite": str(args.benchmark), "suite_manifest_sha256": sha256(args.benchmark / "manifest.json"),
            "base_model": str(args.base_model), "base_model_files": model_files,
            "student": checkpoint_identity(args.checkpoint),
            "teacher": checkpoint_identity(args.teacher_checkpoint),
            "modes": list(MODES), "selected_ids": [row["id"] for row in selected],
            "scope": "smoke" if args.smoke_per_benchmark else "full",
            "seed": 42, "latent_shape": [1, 8, 512],
            "code": {str(p.relative_to(ROOT)): sha256(p) for p in code}}


def check_binding(config):
    for name, expected in config["code"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Code changed during this run: {name}")
    for name, expected in config["base_model_files"].items():
        if sha256(Path(config["base_model"]) / name) != expected:
            raise ValueError(f"Base processor/configuration changed: {name}")
    for key in ("student", "teacher"):
        value = config[key]
        stat = Path(value["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (value["bytes"], value["mtime_ns"]):
            raise ValueError(f"Checkpoint changed during this run: {key}")


def read_result(path, identity, mode, binding):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("id"), row.get("mode"), row.get("run_sha256")) != (identity, mode, binding):
        raise ValueError(f"Foreign/corrupt prediction: {path}")
    if not isinstance(row.get("prediction"), str):
        raise ValueError(f"Missing prediction: {path}")
    return row


def read_latent(path, identity, binding):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if stream.metadata() != {"id": identity, "run_sha256": binding}:
            raise ValueError(f"Foreign/corrupt latent: {path}")
    values = load_file(str(path), device="cpu")
    if set(values) != {"teacher", "predicted"}:
        raise ValueError(f"Incomplete cached latent: {path}")
    for value in values.values():
        if tuple(value.shape) != (1, 8, 512) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid cached latent: {path}")
    return values


def worker(args):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    from lvr.models.latent_prediction import LatentPrediction

    config = json.loads((args.output / "run_config.json").read_text())
    check_binding(config)
    binding = digest(config)
    rows, protocol, _, donors = load_data(config["suite"], verify_images=False)
    lookup = {row["id"]: row for row in rows}
    selected = [lookup[identity] for identity in config["selected_ids"]]
    pool = required_latent_rows(rows, selected, donors) if args.phase == "latents" else selected
    assigned = pool[args.rank::args.workers]
    if args.phase == "latents":
        pending = []
        for row in assigned:
            path = item_path(args.output, "latent", row["id"])
            if path.exists():
                read_latent(path, row["id"], binding)
            else:
                pending.append(row)
    else:
        pending = [row for row in assigned if any(read_result(item_path(args.output, "answer", row["id"], mode),
                    row["id"], mode, binding) is None for mode in MODES)]
    if not pending:
        return
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = f"cuda:{args.rank}"
    torch.cuda.set_device(device)
    print("LOADING", config["family"], args.phase, "worker", args.rank, flush=True)
    processor = AutoProcessor.from_pretrained(config["base_model"], local_files_only=True)
    processor.tokenizer.padding_side = "right"
    model = LatentPrediction.from_checkpoint(config["student"]["path"], device,
                source_checkpoint=config["teacher"]["path"], base_model_path=config["base_model"])
    network = model.network
    if network.qa.config.model_type != FAMILIES[config["family"]][0]:
        raise ValueError("Loaded checkpoint uses the wrong Qwen family")
    if (network.hparams.num_latents, network.hparams.latent_dim, network.hparams.gaussian,
            network.hparams.encoder_includes_original) != (8, 512, False, True):
        raise ValueError("This ablation requires the deterministic original+helper 8x512 interface")
    inference = protocol["inference"]
    kwargs = dict(processor=processor, image_root=config["suite"], num_latents=8,
                  original_pixels=None, crop_pixels=protocol["helper_pixels"],
                  max_length=inference["max_input_tokens"], system_prompt=inference["system_prompt"],
                  include_answers=False, include_descriptions=False, include_reconstruction=False)
    qa_collator = CropQACollator(**kwargs, include_encoder=False)
    teacher_collator = CropQACollator(**kwargs, include_encoder=True,
                      encoder_uses_question=network.hparams.encoder_uses_question,
                      encoder_includes_original=network.hparams.encoder_includes_original,
                      encoder_type=network.hparams.encoder_type)
    write_json(args.output / "logs" / f"processor_{args.phase}_{args.rank}.json",
               {"image_processor": processor.image_processor.to_dict(),
                "encoder_uses_question": bool(network.hparams.encoder_uses_question),
                "prediction_num_loops": model.prediction_loop_count(), "checkpoint_step": model.checkpoint_step})

    def move(batch):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def check_context(batch, backbone, generation=False):
        config_text = getattr(backbone.config, "text_config", backbone.config)
        tokens = int(batch["input_ids"].shape[-1])
        if tokens + (inference["max_new_tokens"] if generation else 0) > config_text.max_position_embeddings:
            raise ValueError("Context overflow; input will not be truncated")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(pending, 1):
            started = time.monotonic()
            if args.phase == "latents":
                qa = move(qa_collator([model_item(row, config["suite"])])["qa"])
                teacher = move(teacher_collator([model_item(row, config["suite"], teacher=True)])["encoder"])
                check_context(qa, network.qa)
                check_context(teacher, network.encoder)
                values = {"predicted": model.predict_latents(qa).detach().float().cpu().contiguous(),
                          "teacher": network.encode(teacher)[0].detach().float().cpu().contiguous()}
                path = item_path(args.output, "latent", row["id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + f".{os.getpid()}.partial")
                save_file(values, str(temporary), metadata={"id": row["id"], "run_sha256": binding})
                read_latent(temporary, row["id"], binding)
                temporary.replace(path)
                del qa, teacher, values
            else:
                batch = qa_collator([model_item(row, config["suite"])])["qa"]
                qa = move(batch)
                check_context(qa, network.qa, generation=True)
                grid = batch["image_grid_thw"].tolist()
                patch = processor.image_processor.patch_size
                for mode in MODES:
                    path = item_path(args.output, "answer", row["id"], mode)
                    if read_result(path, row["id"], mode, binding) is not None:
                        continue
                    source, kind = latent_source(mode, row["id"], donors)
                    if mode == "zero":
                        latent = torch.zeros((1, 8, 512), dtype=torch.float32, device=device)
                    else:
                        latent = read_latent(item_path(args.output, "latent", source), source, binding)[kind].to(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    mode_started = time.monotonic()
                    tokens = network.generate_answers(qa, latent, max_new_tokens=inference["max_new_tokens"])
                    result = {"id": row["id"], "mode": mode, "run_sha256": binding,
                              "benchmark": row["benchmark"], "task": row["task"],
                              "prediction": processor.tokenizer.decode(tokens[0], skip_special_tokens=True),
                              "generated_tokens": int(tokens.shape[-1]),
                              "reached_token_limit": int(tokens.shape[-1]) >= inference["max_new_tokens"],
                              "latent_source_id": source, "latent_kind": kind or "zero_before_projector",
                              "qa_inputs_sha256": qa_identity(row),
                              "input_tokens": int(qa["input_ids"].shape[-1]), "image_grid_thw": grid,
                              "original_sizes_wh": [[im["width"], im["height"]] for im in row["images"]],
                              "processed_sizes_wh": [[w * patch, h * patch] for _, h, w in grid],
                              "peak_memory_reserved": torch.cuda.max_memory_reserved(device),
                              "seconds": time.monotonic() - mode_started}
                    write_json(path, result)
                    del latent, tokens
                del batch, qa
            print("DONE", args.phase, index, "/", len(pending), row["id"],
                  f"{time.monotonic() - started:.2f}s", flush=True)


def launch_phase(args, phase, environment):
    children = []
    try:
        for rank in range(args.gpus):
            command = [sys.executable, str(HERE / "run.py"), "--worker", "--output", str(args.output),
                       "--phase", phase, "--rank", str(rank), "--workers", str(args.gpus)]
            log = args.output / "logs" / f"{phase}_{rank:02d}.log"
            with log.open("a") as stream:
                children.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=environment, cwd=ROOT, start_new_session=True))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f"{phase} worker failed; inspect {args.output / 'logs'}")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError(f"{phase} worker failed")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=FAMILIES)
    parser.add_argument("--base-model", type=Path, help="Local base weights, config and native processor")
    parser.add_argument("--checkpoint", type=Path, help="Complete trained Joint checkpoint, e.g. last_full.ckpt")
    parser.add_argument("--teacher-checkpoint", type=Path, help="Matching teacher student_source.ckpt")
    parser.add_argument("--benchmark", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, help="Number of visible GPUs; default 1 or the --devices count")
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value, e.g. 0,2,3,5")
    parser.add_argument("--dry-run", action="store_true", help="Validate all data/paths without loading weights or using GPUs")
    parser.add_argument("--smoke-per-benchmark", type=int, default=0, help="Separate small run; 0 means all 2787 records")
    parser.add_argument("--score-python", type=Path, default=ROOT / ".venv-score/bin/python")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("latents", "answers"), help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.worker:
        worker(args)
        return
    for name in ("model_family", "base_model", "checkpoint", "teacher_checkpoint"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    if args.devices is not None:
        devices = [part.strip() for part in args.devices.split(",")]
        if not all(part.strip() for part in devices) or len(set(devices)) != len(devices):
            parser.error("--devices must contain distinct CUDA device IDs")
        if args.gpus is not None and args.gpus != len(devices):
            parser.error("--gpus must equal the number of --devices")
        args.gpus = len(devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    args.gpus = args.gpus if args.gpus is not None else 1
    if args.gpus < 1 or args.smoke_per_benchmark < 0:
        parser.error("--gpus must be positive; --smoke-per-benchmark must be nonnegative")
    args.base_model = args.base_model.expanduser().resolve()
    args.benchmark = args.benchmark.expanduser().resolve()
    for path in (args.checkpoint, args.teacher_checkpoint):
        if not path.expanduser().is_file():
            parser.error(f"Complete checkpoint file missing: {path}; consolidate ZeRO shards first")
    validate_family(args.base_model, args.model_family)
    if not list(args.base_model.glob("*.safetensors")):
        parser.error("Local base model safetensors are missing")
    rows, _, _, donors = load_data(args.benchmark)
    selected = select_rows(rows, args.smoke_per_benchmark)
    print(json.dumps({"family": args.model_family, "benchmark_counts": COUNTS, "modes": MODES,
                      "targets": len(selected), "generations": len(selected) * len(MODES),
                      "latent_cache_records": len(required_latent_rows(rows, selected, donors)),
                      "gpus": args.gpus, "scope": "smoke" if args.smoke_per_benchmark else "full",
                      "gpu_started": False}, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.score_python.is_file():
        parser.error("Scoring Python missing; install scoring requirements or pass --score-python")
    import torch
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} GPUs, visible {torch.cuda.device_count()}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Binding checkpoint hashes; large files may take several minutes.", flush=True)
        config = make_config(args, selected)
        bind_json(args.output / "run_config.json", config)
        environment = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        try:
            for phase in ("latents", "answers"):
                write_json(args.output / "status.json", {"status": "running", "phase": phase, "gpus": args.gpus})
                launch_phase(args, phase, environment)
            subprocess.run([str(args.score_python.resolve()), str(HERE / "score.py"),
                            "--output", str(args.output)], env=environment, cwd=ROOT, check=True)
            write_json(args.output / "status.json", {"status": "complete", "scope": config["scope"],
                       "generations": len(selected) * len(MODES)})
        except BaseException as error:
            write_json(args.output / "status.json", {"status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT / "src"))
    main()
