#!/usr/bin/env python3
"""Train the original+crops teacher and the student's two prediction phases."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

from lvr.training.diagnostics import diagnostic_environment
from lvr.data.defaults import split_records

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("teacher_alignment", "teacher_joint", "warmup", "joint")
CUDA_OOM_EXIT_CODE = 75  # The preflight shell may retry a smaller batch only for this code.


def merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def recipe(phase):
    directory = ROOT / "configs/latent_prediction"
    first = "teacher_alignment" if phase.startswith("teacher") or phase == "plain_sft" else "warmup"
    config = yaml.safe_load((directory / f"{first}.yaml").read_text())
    if phase != first:
        merge(config, yaml.safe_load((directory / f"{phase}.yaml").read_text()))
    return config


def training_failure_exit_code(log_path, offset=0):
    """Distinguish CUDA OOM from code/configuration failures in this invocation only."""
    pattern = re.compile(r"(?:torch\.(?:cuda\.)?OutOfMemoryError|RuntimeError): CUDA out of memory")
    with Path(log_path).open(errors="replace") as log:
        log.seek(offset)
        if any(pattern.search(line) for line in log):
            return CUDA_OOM_EXIT_CODE
    return 1


def is_deepspeed_checkpoint(path):
    """A Lightning DeepSpeed checkpoint is a directory with a tag in ``latest``."""
    path = Path(path)
    return path.is_dir() and (path / "latest").is_file()


def checkpoint_path(output, phase, deepspeed=False):
    name = "last_full.ckpt" if deepspeed else "last.ckpt"
    return Path(output) / phase / "checkpoints" / name


def latest_resume_checkpoint(directory, deepspeed=False):
    directory = Path(directory)
    preferred = directory / "last.ckpt"
    valid = is_deepspeed_checkpoint if deepspeed else Path.is_file
    if valid(preferred):
        return preferred
    candidates = [path for path in directory.glob("*.ckpt") if valid(path)]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def consolidate_deepspeed_checkpoint(source, destination, force=False):
    """Atomically create a normal Lightning checkpoint for phase transitions/evaluation."""
    source, destination = Path(source), Path(destination)
    if not is_deepspeed_checkpoint(source):
        raise FileNotFoundError(f"Missing DeepSpeed checkpoint: {source}")
    if destination.is_file() and not force:
        return destination
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        convert_zero_checkpoint_to_fp32_state_dict(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "teacher", "prediction", "plain_sft", *PHASES))
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data")
    parser.add_argument("--training-data-root", type=Path, default=ROOT / "data/default_training",
                        help="QA/bbox release root; explicit data retains historical training inputs")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/latent_prediction/captions.jsonl.gz")
    parser.add_argument("--source-checkpoint", type=Path, help="Existing original+crops teacher for prediction stages")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latent_prediction")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete phases from their latest full checkpoint")
    parser.add_argument("--preflight", action="store_true", help="Two optimizer updates per selected stage in an independent output")
    parser.add_argument("--preflight-eval-samples-per-gpu", type=int, default=2,
                        help="VStar questions per GPU/mode after a preflight ending in joint (default: 2)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Record per-rank events/stacks and enable distributed communication checks")
    parser.add_argument("--dry-run", action="store_true", help="Print fully resolved recipes and commands without training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate joint ablations or plain SFT on the five fixed benchmarks")
    parser.add_argument("--eval-processor-default-resolution", action="store_true",
                        help="Use the base processor's image limits for formal evaluation only")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, help="Override per-GPU batch while preserving global batch 128")
    parser.add_argument("--reconstruction-weight", type=float, default=0.0,
                        help="Add auxiliary feature reconstruction to both teacher phases (default: disabled)")
    parser.add_argument("--deepspeed-config", type=Path,
                        help="Enable Lightning DeepSpeedStrategy with this ZeRO configuration")
    args = parser.parse_args()
    args.training_data_root = args.training_data_root.resolve()
    train_records = split_records(args.training_data_root / "sft/train")
    if not math.isfinite(args.reconstruction_weight) or args.reconstruction_weight < 0:
        parser.error("--reconstruction-weight must be finite and nonnegative")
    if args.preflight_eval_samples_per_gpu < 1:
        parser.error("--preflight-eval-samples-per-gpu must be positive")
    diagnostics = args.diagnostics or (args.preflight and os.environ.get("LVR_PREFLIGHT_DIAGNOSTICS") == "1")
    if args.evaluate and args.preflight:
        parser.error("Preflight checkpoints are not formal benchmark checkpoints")
    phases = PHASES if args.stage == "all" else PHASES[:2] if args.stage == "teacher" else PHASES[2:] if args.stage == "prediction" else (args.stage,)
    if args.reconstruction_weight and not any(phase.startswith("teacher") for phase in phases):
        parser.error("--reconstruction-weight applies to teacher training; select teacher or all")
    if args.preflight and "joint" in phases:
        protocol = json.loads((ROOT / "benchmark/latent_prediction/data_config.json").read_text())
        if args.gpus * args.preflight_eval_samples_per_gpu > len(protocol["VStarBench"]["indices"]):
            parser.error("VStar preflight requests more questions than the recorded protocol contains")
    if args.evaluate and not {"joint", "plain_sft"}.intersection(phases):
        parser.error("--evaluate requires all, prediction, joint, or plain_sft")
    output = args.output.resolve()
    if args.preflight:
        output = output / "preflight"
    deepspeed = args.deepspeed_config is not None
    if deepspeed:
        args.deepspeed_config = args.deepspeed_config.resolve()
        if not args.deepspeed_config.is_file():
            parser.error(f"Missing DeepSpeed config: {args.deepspeed_config}")
    teacher = (args.source_checkpoint or checkpoint_path(output, "teacher_joint", deepspeed)).resolve()
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
        phase_env = diagnostic_environment(env, run, diagnostics)
        config = recipe(phase)
        if phase.startswith("teacher") and args.reconstruction_weight:
            config["model"]["reconstruction_weight"] = args.reconstruction_weight
        config.update(exp_name=str(output), version=phase)
        if deepspeed:
            config["trainer"]["strategy"] = {
                "class_path": "lightning.pytorch.strategies.DeepSpeedStrategy",
                "init_args": {"config": str(args.deepspeed_config), "exclude_frozen_parameters": False},
            }
        config["data"].update(processor_name_or_path=str(args.base_model.resolve()),
                              image_root=str(args.image_root.resolve()), caption_path=str(args.captions.resolve()))
        if phase == "plain_sft":
            config["data"]["caption_path"] = None
        for key, relative in {"train_path": "sft/train", "val_path": "sft/test",
                              "train_boxes_path": "align/train", "val_boxes_path": "align/test"}.items():
            config["data"][key] = str(args.training_data_root / relative)
        if phase.startswith("teacher") or phase == "plain_sft":
            config["model"].update(encoder_model_name_or_path=str(args.base_model.resolve()),
                                    qa_model_name_or_path=str(args.base_model.resolve()))
        else:
            config["model"].update(source_checkpoint=str(teacher), base_model_path=str(args.base_model.resolve()))
        batch = config["data"]["batch_size"] if args.batch_size is None else args.batch_size
        if args.gpus < 1 or batch < 1 or 128 % (args.gpus * batch):
            parser.error("gpus × batch-size must be a positive divisor of 128")
        config["data"]["batch_size"] = batch
        accumulation = 128 // (args.gpus * batch)
        steps_per_epoch = math.ceil(math.ceil(math.ceil(train_records / args.gpus) / batch) / accumulation)
        expected_steps = min(200, steps_per_epoch) if phase == "warmup" else steps_per_epoch
        config["trainer"].update(devices=args.gpus, accumulate_grad_batches=accumulation, max_steps=expected_steps)
        if phase == "warmup":
            config["trainer"]["limit_train_batches"] = expected_steps * accumulation
        if phase == "teacher_joint":
            config["init_checkpoint"] = str(checkpoint_path(output, "teacher_alignment", deepspeed))
        elif phase == "joint":
            config["init_checkpoint"] = str(checkpoint_path(output, "warmup", deepspeed))
        if args.preflight:
            config["trainer"].update(max_steps=2, limit_train_batches=2 * accumulation,
                                      limit_val_batches=2, val_check_interval=1.0, num_sanity_val_steps=0)
        config_file = output / "recipes" / f"{phase}.yaml"
        module = "lvr.stage1" if phase.startswith("teacher") or phase == "plain_sft" else "lvr.predict_latents"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc_per_node={args.gpus}", "--module", module, "fit", "--config", str(config_file)]
        if args.dry_run:
            print(yaml.safe_dump(config, sort_keys=False))
            if diagnostics:
                print(f"[diagnostics] directory={phase_env['LVR_DIAGNOSTICS_DIR']} "
                      f"interval={phase_env['LVR_DIAGNOSTICS_INTERVAL_SECONDS']}s "
                      f"TORCH_DISTRIBUTED_DEBUG={phase_env['TORCH_DISTRIBUTED_DEBUG']}")
            print(shlex.join(command))
            continue
        recipe_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        started = run / "training_started.json"
        if started.exists() and json.loads(started.read_text())["recipe_sha256"] != recipe_hash:
            parser.error(f"Configuration changed for existing {phase}; use a new --output")
        inputs = (args.base_model, args.image_root) if phase == "plain_sft" else (args.base_model, args.image_root, args.captions)
        for path in inputs:
            if not path.exists():
                parser.error(f"Missing input: {path}; follow the preparation steps in README.md")
        marker = run / "train_exit_code"
        native_checkpoint = run / "checkpoints/last.ckpt"
        native_exists = is_deepspeed_checkpoint(native_checkpoint) if deepspeed else native_checkpoint.is_file()
        if marker.exists() and marker.read_text().strip() == "0" and native_exists:
            if deepspeed:
                consolidate_deepspeed_checkpoint(native_checkpoint, checkpoint_path(output, phase, True))
            print(f"Already complete: {phase}", flush=True)
            continue
        if (run / "training_started.json").exists():
            if not args.resume:
                parser.error(f"Incomplete phase {phase}; use --resume or a new --output")
            resume_checkpoint = latest_resume_checkpoint(run / "checkpoints", deepspeed)
            if resume_checkpoint is None:
                parser.error(f"No recoverable checkpoint for {phase}; use a new --output")
            config.pop("init_checkpoint", None)
            config["ckpt_path"] = str(resume_checkpoint.resolve())
        for key in ("init_checkpoint", "ckpt_path"):
            if not config.get(key):
                continue
            path = Path(config[key])
            valid = is_deepspeed_checkpoint(path) if key == "ckpt_path" and deepspeed else path.is_file()
            if not valid:
                parser.error(f"Missing {key}: {path}")
        if phase in ("warmup", "joint") and not teacher.is_file():
            parser.error(f"Missing teacher: {teacher}; train the teacher or pass --source-checkpoint")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        run.mkdir(parents=True, exist_ok=True)
        if diagnostics:
            Path(phase_env["LVR_DIAGNOSTICS_DIR"]).mkdir(parents=True, exist_ok=True)
            print(f"[diagnostics] events/stacks: {phase_env['LVR_DIAGNOSTICS_DIR']} "
                  f"(every {phase_env['LVR_DIAGNOSTICS_INTERVAL_SECONDS']}s); "
                  "Python events do not imply CUDA completion", flush=True)
        started.write_text(json.dumps({"command": command, "recipe": str(config_file),
                                       "recipe_sha256": recipe_hash}, indent=2) + "\n")
        print(shlex.join(command), flush=True)
        with (run / "train.log").open("a") as log:
            log_offset = log.tell()
            code = subprocess.run(command, cwd=ROOT, env=phase_env, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            marker.write_text(str(code) + "\n")
            print(f"{phase} failed ({code}); inspect {run / 'train.log'}", file=sys.stderr, flush=True)
            raise SystemExit(training_failure_exit_code(run / "train.log", log_offset))
        for rank in range(args.gpus):
            state = json.loads((run / f"runtime_rank{rank}.json").read_text())
            if state["status"] != "training_complete" or state["optimizer_steps"] != (2 if args.preflight else expected_steps):
                raise RuntimeError(f"Incomplete {phase} rank {rank}: {state}")
        if not (is_deepspeed_checkpoint(native_checkpoint) if deepspeed else native_checkpoint.is_file()):
            raise RuntimeError(f"Missing completed checkpoint: {run}")
        if deepspeed:
            consolidate_deepspeed_checkpoint(native_checkpoint, checkpoint_path(output, phase, True), force=True)
        marker.write_text("0\n")
        print(f"Completed: {phase}", flush=True)
    if args.preflight and "joint" in phases:
        evaluation = output / "benchmark_preflight"
        command = [sys.executable, str(Path(__file__).with_name("preflight_evaluate.py")),
                   "--checkpoint", str(checkpoint_path(output, "joint", deepspeed)),
                   "--source-checkpoint", str(teacher), "--base-model", str(args.base_model.resolve()),
                   "--output", str(evaluation), "--gpus", str(args.gpus),
                   "--samples-per-gpu", str(args.preflight_eval_samples_per_gpu),
                   "--reserved-limit-gib", os.environ.get("PREFLIGHT_RESERVED_LIMIT_GIB", "75")]
        if os.environ.get("ALLOW_TIGHT_MEMORY") == "1":
            command.append("--allow-tight-memory")
        if args.dry_run:
            print(shlex.join([*command, "--dry-run"]))
        else:
            evaluation.mkdir(parents=True, exist_ok=True)
            log_path = evaluation / "preflight.log"
            print(f"[vstar-preflight] checking final inference on {args.gpus} GPUs; log={log_path}", flush=True)
            with log_path.open("a") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            (evaluation / "runner_exit_code").write_text(str(code) + "\n")
            if code:
                print(f"Benchmark preflight failed ({code}); inspect {log_path}. "
                      "Inference uses batch 1; training batch fallback is not applicable.", file=sys.stderr, flush=True)
                raise SystemExit(1)
            summary = json.loads((evaluation / "summary.json").read_text())
            if (summary.get("status") != "complete" or summary.get("world_size") != args.gpus
                    or summary.get("samples_per_gpu") != args.preflight_eval_samples_per_gpu
                    or summary.get("modes_completed") != ["predicted", "teacher", "zero", "shuffled"]):
                raise RuntimeError(f"Incomplete benchmark preflight: {evaluation}")
            print(f"Completed: VStar benchmark preflight; {evaluation / 'summary.json'}", flush=True)
    evaluation_output = output / ("benchmark_processor_default" if args.eval_processor_default_resolution else "benchmark")
    resolution_args = ["--processor-default-resolution"] if args.eval_processor_default_resolution else []
    if args.evaluate and "plain_sft" in phases:
        command = [sys.executable, str(Path(__file__).with_name("evaluate_plain_sft.py")),
                   "--checkpoint", str(checkpoint_path(output, "plain_sft", deepspeed)),
                   "--config", str(output / "recipes/plain_sft.yaml"),
                   "--base-model", str(args.base_model.resolve()),
                   "--output", str(evaluation_output), "--gpus", str(args.gpus), *resolution_args]
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    elif args.evaluate:
        command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                   "--checkpoint", str(checkpoint_path(output, "joint", deepspeed)),
                   "--source-checkpoint", str(teacher), "--base-model", str(args.base_model.resolve()),
                   "--output", str(evaluation_output), "--gpus", str(args.gpus), *resolution_args]
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
