#!/usr/bin/env python3
"""Train the original+crops teacher and the student's two prediction phases."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("teacher_alignment", "teacher_joint", "warmup", "joint")
STEPS = dict(teacher_alignment=961, teacher_joint=961, warmup=200, joint=961)


def merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def recipe(phase):
    directory = ROOT / "configs/latent_prediction"
    first = "teacher_alignment" if phase.startswith("teacher") else "warmup"
    config = yaml.safe_load((directory / f"{first}.yaml").read_text())
    if phase != first:
        merge(config, yaml.safe_load((directory / f"{phase}.yaml").read_text()))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "teacher", "prediction", *PHASES))
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/latent_prediction/captions.jsonl.gz")
    parser.add_argument("--source-checkpoint", type=Path, help="Existing original+crops teacher for prediction stages")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latent_prediction")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete phases from their latest full checkpoint")
    parser.add_argument("--preflight", action="store_true", help="Two optimizer updates per selected stage in an independent output")
    parser.add_argument("--dry-run", action="store_true", help="Print fully resolved recipes and commands without training")
    parser.add_argument("--evaluate", action="store_true", help="Run all four benchmark conditions after joint training")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, help="Override per-GPU batch while preserving global batch 128")
    args = parser.parse_args()
    if args.evaluate and args.preflight:
        parser.error("Preflight checkpoints are not formal benchmark checkpoints")
    phases = PHASES if args.stage == "all" else PHASES[:2] if args.stage == "teacher" else PHASES[2:] if args.stage == "prediction" else (args.stage,)
    if args.evaluate and "joint" not in phases:
        parser.error("--evaluate requires all, prediction, or joint")
    output = args.output.resolve()
    if args.preflight:
        output = output / "preflight"
    teacher = (args.source_checkpoint or output / "teacher_joint/checkpoints/last.ckpt").resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        lock = (output / "workflow.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    for key, value in {"OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                       "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1"}.items():
        env.setdefault(key, value)
    for phase in phases:
        run = output / phase
        config = recipe(phase)
        config.update(exp_name=str(output), version=phase)
        config["data"].update(processor_name_or_path=str(args.base_model.resolve()),
                              image_root=str(args.image_root.resolve()), caption_path=str(args.captions.resolve()))
        for key, relative in {"train_path": "sft/train", "val_path": "sft/test",
                              "train_boxes_path": "align/train", "val_boxes_path": "align/test"}.items():
            config["data"][key] = str(ROOT / "data" / relative)
        if phase.startswith("teacher"):
            config["model"].update(encoder_model_name_or_path=str(args.base_model.resolve()),
                                    qa_model_name_or_path=str(args.base_model.resolve()))
        else:
            config["model"].update(source_checkpoint=str(teacher), base_model_path=str(args.base_model.resolve()))
        batch = config["data"]["batch_size"] if args.batch_size is None else args.batch_size
        if args.gpus < 1 or batch < 1 or 128 % (args.gpus * batch):
            parser.error("gpus × batch-size must be a positive divisor of 128")
        config["data"]["batch_size"] = batch
        accumulation = 128 // (args.gpus * batch)
        config["trainer"].update(devices=args.gpus, accumulate_grad_batches=accumulation)
        if phase == "warmup":
            config["trainer"]["limit_train_batches"] = 200 * accumulation
        if phase == "teacher_joint":
            config["init_checkpoint"] = str(output / "teacher_alignment/checkpoints/last.ckpt")
        elif phase == "joint":
            config["init_checkpoint"] = str(output / "warmup/checkpoints/last.ckpt")
        if args.preflight:
            config["trainer"].update(max_steps=2, limit_train_batches=2 * accumulation,
                                      limit_val_batches=2, val_check_interval=1.0, num_sanity_val_steps=0)
        config_file = output / "recipes" / f"{phase}.yaml"
        module = "lvr.stage1" if phase.startswith("teacher") else "lvr.predict_latents"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc_per_node={args.gpus}", "--module", module, "fit", "--config", str(config_file)]
        if args.dry_run:
            print(yaml.safe_dump(config, sort_keys=False))
            print(shlex.join(command))
            continue
        recipe_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        started = run / "training_started.json"
        if started.exists() and json.loads(started.read_text())["recipe_sha256"] != recipe_hash:
            parser.error(f"Configuration changed for existing {phase}; use a new --output")
        for path in (args.base_model, args.image_root, args.captions):
            if not path.exists():
                parser.error(f"Missing input: {path}; follow the preparation steps in README.md")
        marker = run / "train_exit_code"
        if marker.exists() and marker.read_text().strip() == "0" and (run / "checkpoints/last.ckpt").is_file():
            print(f"Already complete: {phase}", flush=True)
            continue
        if (run / "training_started.json").exists():
            if not args.resume:
                parser.error(f"Incomplete phase {phase}; use --resume or a new --output")
            paths = list((run / "checkpoints").glob("*.ckpt"))
            if not paths:
                parser.error(f"No recoverable checkpoint for {phase}; use a new --output")
            config.pop("init_checkpoint", None)
            config["ckpt_path"] = str(max(paths, key=lambda path: path.stat().st_mtime).resolve())
        for key in ("init_checkpoint", "ckpt_path"):
            if config.get(key) and not Path(config[key]).is_file():
                parser.error(f"Missing {key}: {config[key]}")
        if not phase.startswith("teacher") and not teacher.is_file():
            parser.error(f"Missing teacher: {teacher}; train the teacher or pass --source-checkpoint")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        run.mkdir(parents=True, exist_ok=True)
        started.write_text(json.dumps({"command": command, "recipe": str(config_file),
                                       "recipe_sha256": recipe_hash}, indent=2) + "\n")
        print(shlex.join(command), flush=True)
        with (run / "train.log").open("a") as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            marker.write_text(str(code) + "\n")
            raise SystemExit(f"{phase} failed ({code}); inspect {run / 'train.log'}")
        for rank in range(args.gpus):
            state = json.loads((run / f"runtime_rank{rank}.json").read_text())
            if state["status"] != "training_complete" or state["optimizer_steps"] != (2 if args.preflight else STEPS[phase]):
                raise RuntimeError(f"Incomplete {phase} rank {rank}: {state}")
        if not (run / "checkpoints/last.ckpt").is_file():
            raise RuntimeError(f"Missing completed checkpoint: {run}")
        marker.write_text("0\n")
        print(f"Completed: {phase}", flush=True)
    if args.evaluate and not args.dry_run:
        subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
                        "--checkpoint", str(output / "joint/checkpoints/last.ckpt"),
                        "--source-checkpoint", str(teacher), "--base-model", str(args.base_model.resolve()),
                        "--output", str(output / "benchmark"), "--gpus", str(args.gpus)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
