#!/usr/bin/env python3
"""Train a matched plain SFT baseline, optionally after a supervised LVR experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import copy
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml
from lvr.data.defaults import training_data_root

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("plain_sft_shared", SCRIPT.with_name("train_and_evaluate.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
ROOT = shared.ROOT


class DependencyFailed(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def status(output, state, **details):
    shared.write_json(output / "workflow_status.json", {"status": state, "at_utc": now(), **details})


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def bind_dependency(directory):
    if directory is None:
        return None
    plan = json.loads((directory / "workflow_plan.json").read_text())
    processes = json.loads((directory / "processes.json").read_text())
    if plan["preflight"]:
        raise ValueError("A preflight run cannot be the preceding formal experiment")
    return {"output": str(directory), "plan_sha256": shared.digest(plan),
            "processes": {str(processes[key]): process_identity(processes[key])
                          for key in ("workflow_pid", "supervisor_pid")}}


def dependency_ready(binding):
    """A dead process or completed training alone is never sufficient to launch."""
    directory = Path(binding["output"])
    plan = json.loads((directory / "workflow_plan.json").read_text())
    if shared.digest(plan) != binding["plan_sha256"]:
        raise DependencyFailed("The preceding experiment's plan changed")
    alive = any(token is not None and process_identity(pid) == token
                for pid, token in binding["processes"].items())
    exit_file = directory / "workflow_exit_code"
    if not exit_file.exists():
        if alive:
            return False
        raise DependencyFailed("The preceding experiment stopped without a successful supervised exit")
    code = int(exit_file.read_text().strip())
    if code != 0:
        raise DependencyFailed(f"The preceding experiment exited {code}; plain SFT was not started")
    complete = json.loads((directory / "workflow_status.json").read_text())
    if (complete.get("status") != "complete" or complete.get("preflight") is not False
            or complete.get("plan_sha256") != binding["plan_sha256"]):
        raise DependencyFailed("The preceding experiment has no matching complete workflow status")
    for phase in shared.PHASES:
        stage = json.loads((directory / phase / "complete.json").read_text())
        if (stage.get("status") != "complete"
                or stage.get("optimizer_steps") != plan["optimizer_steps"][phase]
                or stage.get("recipe_sha256") != shared.digest(plan["recipes"][phase])):
            raise DependencyFailed(f"The preceding stage is incomplete or inconsistent: {phase}")
    report = json.loads((directory / "benchmark/summary.json").read_text())
    modes = report.get("modes", [])
    expected_samples = sum(plan["inputs"]["benchmark_counts"].values())
    if report.get("status") != "complete" or len(modes) != len(shared.MODES):
        raise DependencyFailed("The preceding benchmark is incomplete")
    if {r.get("mode") for r in modes} != set(shared.MODES):
        raise DependencyFailed("The preceding benchmark is missing modes")
    for mode in modes:
        count = plan["inputs"]["teacher_records"] if mode["mode"] == "teacher" else expected_samples
        if mode.get("status") != "complete" or mode.get("samples") != count:
            raise DependencyFailed(f"The preceding benchmark is incomplete: {mode['mode']}")
    return not alive


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def wait_for_slot(args, binding):
    """Keep the predecessor's workflow lock once it completes, through this experiment."""
    while True:
        lock = None
        if binding is not None and not dependency_ready(binding):
            reason = "waiting_for_preceding_training_and_benchmark"
        else:
            if binding is not None:
                lock = (Path(binding["output"]) / "workflow.lock").open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    status(args.output, "waiting", reason="preceding_workflow_lock_is_held")
                    time.sleep(args.poll_seconds)
                    continue
            try:
                busy = gpu_pids()
            except Exception:
                if lock is not None:
                    lock.close()
                raise
            if not busy:
                return lock
            if lock is not None:
                lock.close()
            reason = "waiting_for_gpu_processes_to_exit"
        status(args.output, "waiting", reason=reason, after=binding["output"] if binding else None)
        print(f"WAIT {now()} {reason}", flush=True)
        time.sleep(args.poll_seconds)


def build_recipe(args, inputs):
    mixed = args.dataset == "mixed"
    data_root = training_data_root(args, ROOT)
    records = inputs["train_records"] if mixed else inputs["counts"]["reasoning_v1/train"]
    global_batch = getattr(args, "global_batch_size", 128)
    epochs = getattr(args, "epochs", 1.0)
    if (args.gpus < 1 or args.batch_size < 1 or global_batch < 1
            or global_batch % (args.gpus * args.batch_size)):
        raise ValueError("gpus × batch-size must divide the positive global batch")
    if not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("epochs must be positive and finite")
    accumulation = global_batch // (args.gpus * args.batch_size)
    steps_per_epoch = math.ceil(math.ceil(math.ceil(records / args.gpus) / args.batch_size) / accumulation)
    steps = math.ceil(epochs * steps_per_epoch)
    config = shared.legacy.recipe("plain_sft")
    config.update(exp_name=str(args.output), version="plain_sft")
    preflight_steps = getattr(args, "preflight_steps", 2)
    config["trainer"].update(devices=args.gpus, max_epochs=math.ceil(epochs), max_steps=preflight_steps if args.preflight else steps,
                              accumulate_grad_batches=accumulation, limit_train_batches=1.0,
                              strategy={"class_path": "lightning.pytorch.strategies.DeepSpeedStrategy",
                                        "init_args": {"config": str(args.deepspeed_config),
                                                      "exclude_frozen_parameters": False}})
    config["trainer"]["callbacks"].append({"class_path": "lvr.training.audit.HostMemoryAudit"})
    config["model"].update(encoder_model_name_or_path=str(args.model), qa_model_name_or_path=str(args.model))
    config["data"].update(processor_name_or_path=str(args.model), image_root=str(args.image_root),
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           val_num_workers=args.val_num_workers, prefetch_factor=args.prefetch_factor,
                           pin_memory=False, caption_path=None,
                           train_path=str(data_root / "sft/train" if mixed else args.release / "train"),
                           val_path=str(data_root / "sft/test" if mixed else args.release / "val"),
                           train_boxes_path=str(data_root / "align/train") if mixed else None,
                           val_boxes_path=str(data_root / "align/test") if mixed else None,
                           additional_train_path=str(args.release / "train") if mixed else None,
                           additional_val_path=str(args.release / "val") if mixed else None)
    for key in ("original_pixels", "max_length"):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            config["data"][key] = value
    if getattr(args, "train_length_manifest_path", None):
        config["data"]["train_length_manifest_path"] = str(args.train_length_manifest_path)
    if getattr(args, "preflight_sample_ids", None):
        if not args.preflight:
            raise ValueError("Stress sample manifests are only valid for preflight runs")
        config["data"].update(train_sample_ids_path=str(args.preflight_sample_ids), train_length_manifest_path=None)
    if hasattr(args, "epochs"):
        # Fix the schedule before training; a resumed run must retain this horizon.
        config["model"]["lr_schedule_steps"] = config["trainer"]["max_steps"]
    cadence = getattr(args, "checkpoint_every_epochs", None)
    if cadence is not None:
        candidates = checkpoint_schedule(steps_per_epoch, epochs, cadence)
        if args.preflight:
            candidates = [{"epoch": preflight_steps / steps_per_epoch, "step": preflight_steps}]
        config["trainer"]["callbacks"] = [
            {"class_path": "lvr.training.checkpoints.OptimizerStepModelCheckpoint", "init_args": {
                "steps": [row["step"] for row in candidates], "filename": "candidate-{step:06d}",
                "save_top_k": -1, "save_last": "link", "auto_insert_metric_name": False}},
            {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {
                "filename": "recovery-{step:06d}", "every_n_train_steps": 400, "save_top_k": 1,
                "save_last": False, "save_on_train_epoch_end": False, "auto_insert_metric_name": False}},
            *[cb for cb in config["trainer"]["callbacks"] if not cb["class_path"].endswith("ModelCheckpoint")],
        ]
    if args.preflight:
        config["trainer"].update(limit_train_batches=preflight_steps * accumulation, limit_val_batches=2,
                                  val_check_interval=1.0, num_sanity_val_steps=0)
    elif getattr(args, "epoch_validation_only", False):
        # Keep whole-epoch training diagnostics. Fractional-epoch candidate scores
        # are produced separately from the saved checkpoints, not batch intervals.
        config["trainer"]["val_check_interval"] = 1.0
    if getattr(args, "memory_probe", False):
        if not args.preflight or not getattr(args, "preflight_sample_ids", None):
            raise ValueError("Memory probes require preflight and a deterministic sample manifest")
        config["trainer"].update(enable_checkpointing=False, limit_val_batches=0)
        config["trainer"]["callbacks"] = [cb for cb in config["trainer"]["callbacks"]
                                           if "ModelCheckpoint" not in cb["class_path"]]
        config["trainer"]["callbacks"].append({"class_path": "lvr.training.batch_capacity.GPUCapacityAudit",
            "init_args": {"sample_ids_path": str(args.preflight_sample_ids)}})
    run = args.output / "plain_sft"
    config["trainer"]["default_root_dir"] = str(run)
    for logger in config["trainer"].get("logger", []):
        logger["init_args"]["save_dir"] = str(run)
    for callback in config["trainer"]["callbacks"]:
        if "ModelCheckpoint" in callback["class_path"]:
            callback["init_args"]["dirpath"] = str(run / "checkpoints")
    return config, config["trainer"]["max_steps"]


def checkpoint_schedule(steps_per_epoch, epochs, cadence):
    """Map fractional epochs to updates, including the exact planned final update."""
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError("checkpoint_every_epochs must be positive and finite")
    fractions = [i * cadence for i in range(1, math.floor(epochs / cadence) + 1)]
    if not fractions or not math.isclose(fractions[-1], epochs):
        fractions.append(epochs)
    return [{"epoch": value, "step": math.ceil(value * steps_per_epoch)} for value in fractions]


def inspect_training_inputs(args, *, check_images=False):
    """Audit original-image QA inputs without imposing teacher or benchmark prerequisites."""
    from lvr.data.crop_qa import CropQADataset

    release = json.loads((args.release / "release_manifest.json").read_text())
    if release["status"] != "accepted":
        raise ValueError("Training release is not accepted")
    for name, expected in release["files_sha256"].items():
        if shared.file_sha(args.release / name) != expected:
            raise ValueError(f"Training release file changed: {name}")
    counts, bindings, ids, paths = {}, {}, set(), set()
    splits = [("reasoning_v1/train", args.release / "train", None),
              ("reasoning_v1/val", args.release / "val", None)]
    if args.dataset == "mixed":
        data_root = training_data_root(args, ROOT)
        splits += [("old/train", data_root / "sft/train", data_root / "align/train"),
                   ("old/test", data_root / "sft/test", data_root / "align/test")]
    for name, path, boxes in splits:
        bindings[name] = shared.split_binding(path)
        if boxes is not None:
            bindings[name + "/boxes"] = shared.split_binding(boxes)
        dataset = CropQADataset(path, boxes)
        counts[name] = len(dataset)
        for item in dataset.items:
            if item["id"] in ids:
                raise ValueError(f"Repeated train/validation sample ID: {item['id']}")
            ids.add(item["id"])
            for image in item.get("images") or [{"path": item["question_image"]}]:
                paths.add(args.image_root / image["path"])
        del dataset
    for split in ("train", "val"):
        if counts["reasoning_v1/" + split] != release["splits"][split]["records"]:
            raise ValueError(f"Training count differs from release manifest: {split}")
    if check_images:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    model_config = args.model / "config.json"
    if not model_config.is_file():
        if not args.dry_run:
            raise FileNotFoundError(f"Missing base model config: {model_config}")
        model_type = None
    else:
        model_type = json.loads(model_config.read_text())["model_type"]
        if args.expected_model_type and model_type != args.expected_model_type:
            raise ValueError(f"Expected {args.expected_model_type}, found {model_type}")
    weight_index = args.model / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(weight_index.read_text())["weight_map"].values()))
              if weight_index.is_file() else ["model.safetensors"])
    missing = [name for name in shards if not (args.model / name).is_file() or not (args.model / name).stat().st_size]
    if missing and not args.dry_run:
        raise FileNotFoundError(f"Missing base model shards: {missing}")
    model_files = sorted(args.model.iterdir()) if args.model.is_dir() else []
    return {"release_manifest_sha256": shared.file_sha(args.release / "release_manifest.json"),
            "counts": counts, "train_records": counts.get("old/train", 0) + counts["reasoning_v1/train"],
            "val_records": counts.get("old/test", 0) + counts["reasoning_v1/val"],
            "training_bindings": bindings, "benchmark_counts": {}, "model_type": model_type,
            "model_configuration": {p.name: shared.file_sha(p) for p in model_files
                                    if p.is_file() and p.suffix in (".json", ".jinja")},
            "model_weights": {name: (args.model / name).stat().st_size for name in shards if name not in missing},
            "missing_model_weights": missing,
            "checked": {"unique_sample_ids": len(ids), "original_image_paths": len(paths),
                        "image_existence_checked": check_images, "auxiliary_images_required": False}}


def training_command(args, recipe):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={args.gpus}", "--module", "lvr.stage1", "fit", "--config", str(recipe)]


def run_stage(args, config, steps, env):
    run = args.output / "plain_sft"
    run.mkdir(parents=True, exist_ok=True)
    started, finished = run / "training_started.json", run / "complete.json"
    identity = shared.digest(config)
    for marker in (started, finished):
        if marker.exists() and json.loads(marker.read_text())["recipe_sha256"] != identity:
            raise ValueError("Changed configuration for existing plain SFT stage")
    native, merged = run / "checkpoints/last.ckpt", run / "checkpoints/last_full.ckpt"
    if finished.exists():
        shared.rank_states(run, args.gpus, steps)
        if not getattr(args, "memory_probe", False):
            shared.legacy.consolidate_deepspeed_checkpoint(native, merged)
        print("ALREADY_COMPLETE plain_sft", flush=True)
        return
    current = copy.deepcopy(config)
    if started.exists():
        if getattr(args, "memory_probe", False):
            raise RuntimeError("An interrupted memory probe requires a fresh output directory")
        checkpoint = shared.legacy.latest_resume_checkpoint(run / "checkpoints", deepspeed=True)
        if getattr(args, "checkpoint_every_epochs", None) is not None:
            # `last` links to a retained candidate; a newer rolling recovery
            # checkpoint may exist between those candidates.
            candidates = [p for p in (run / "checkpoints").glob("*.ckpt")
                          if shared.legacy.is_deepspeed_checkpoint(p)]
            checkpoint = max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None
        if checkpoint is None:
            raise RuntimeError("Plain SFT stopped before a recoverable checkpoint; use a fresh --output")
        current["ckpt_path"] = str(checkpoint.resolve())
    recipe = args.output / "recipes/plain_sft.yaml"
    recipe.write_text(yaml.safe_dump(current, sort_keys=False))
    command = training_command(args, recipe)
    shared.write_json(started, {"recipe_sha256": identity, "at_utc": now(), "command": command})
    print("TRAIN " + shlex.join(command), flush=True)
    with (run / "train.log").open("a") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (run / "train_exit_code").write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"Plain SFT exited {code}; inspect {run / 'train.log'}")
    states = shared.rank_states(run, args.gpus, steps)
    if not getattr(args, "memory_probe", False):
        shared.legacy.consolidate_deepspeed_checkpoint(native, merged, force=True)
    shared.write_json(finished, {"status": "complete", "recipe_sha256": identity, "optimizer_steps": steps,
                                 "max_allocated_gib": max(s.get("peak_allocated_gib", 0) for s in states),
                                 "max_reserved_gib": max(s.get("peak_reserved_gib", 0) for s in states)})
    print(f"TRAINING_COMPLETE plain_sft {steps}", flush=True)


def evaluation_command(args, shard):
    command = [sys.executable, str(ROOT / "scripts/reasoning_benchmarks/evaluate.py"),
               "--suite", str(args.suite), "--selection", args.selection, "--mode", "plain",
               "--checkpoint", str(args.output / "plain_sft/checkpoints/last_full.ckpt"),
               "--model", str(args.model), "--device", f"cuda:{shard}",
               "--output", str(args.output / "benchmark/plain" / f"shard_{shard:02d}"),
               "--shard-count", str(args.eval_gpus), "--shard-index", str(shard),
               "--max-new-tokens", str(args.max_new_tokens)]
    if args.preflight:
        command.extend(["--per-task", "1"])
    return command


def run_evaluation(args, env):
    directory = args.output / "benchmark/plain"
    directory.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for shard in range(args.eval_gpus):
            command = evaluation_command(args, shard)
            with (directory / f"shard_{shard:02d}.log").open("a") as log:
                children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(f"Plain benchmark failed; inspect {directory}")
            time.sleep(1)
        if any(p.returncode for p in children):
            raise RuntimeError(f"Plain benchmark failed; inspect {directory}")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    report = shared.merge_shards(directory, [r["id"] for r in shared.selected_records(args, "plain")], args.eval_gpus)
    shared.write_json(args.output / "benchmark/summary.json", {"status": "complete", "modes": [report]})
    with (args.output / "benchmark/results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "benchmark", "n", "correct", "accuracy", "parse_failures", "truncated"])
        writer.writeheader()
        for metric in report["metrics"]:
            if metric["task"] == "all":
                writer.writerow({key: "plain" if key == "mode" else metric[key] for key in writer.fieldnames})
    print(f"EVALUATION_COMPLETE plain {report['samples']}", flush=True)


def make_plan(args, inputs, recipe, steps, binding):
    files = [SCRIPT, SCRIPT.with_name("train_and_evaluate.py"), ROOT / "scripts/latent_prediction/run.py",
             ROOT / "scripts/reasoning_benchmarks/evaluate.py", ROOT / "configs/latent_prediction/teacher_alignment.yaml",
             ROOT / "configs/latent_prediction/plain_sft.yaml", *sorted((ROOT / "src/lvr").rglob("*.py"))]
    return {"format": "lvr-reasoning-plain-sft-workflow-v1", "dataset": args.dataset,
            "inputs": {key: value for key, value in inputs.items() if key != "checked"},
            "recipe": recipe, "optimizer_steps": steps, "selection": args.selection,
            "training_only": bool(getattr(args, "train_only", False)),
            "preflight": args.preflight, "eval_gpus": args.eval_gpus, "max_new_tokens": args.max_new_tokens,
            "dependency": binding, "code": {str(p.relative_to(ROOT)): shared.file_sha(p) for p in files},
            "deepspeed_sha256": shared.file_sha(args.deepspeed_config),
            "data_sidecars": {name: {"path": str(getattr(args, name)), "sha256": shared.file_sha(getattr(args, name))}
                              for name in ("train_length_manifest_path", "preflight_sample_ids")
                              if getattr(args, name, None)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "data/processed/reasoning_v1")
    parser.add_argument("--training-data-root", type=Path, default=ROOT / "data/default_training",
                        help="Old-data release root; pass data explicitly only for historical reproduction")
    parser.add_argument("--suite", type=Path, default=shared.DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reasoning_plain_sft_mixed_v1_8gpu_20261003")
    parser.add_argument("--model", type=Path, default=Path("/workspace/wmz/.cache/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/wmz/lvr_pipeline_new/data"))
    parser.add_argument("--captions", type=Path, default=ROOT / "data/latent_prediction/captions.jsonl.gz")
    parser.add_argument("--deepspeed-config", type=Path, default=ROOT / "configs/latent_prediction/deepspeed_zero3_80gb.json")
    parser.add_argument("--dataset", choices=("mixed", "new-only"), default="mixed")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--original-pixels", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--checkpoint-every-epochs", type=float)
    parser.add_argument("--expected-model-type", choices=("qwen3_vl", "qwen2_5_vl"))
    parser.add_argument("--epoch-validation-only", action="store_true",
                        help="Whole-epoch in-loop diagnostics; validate fractional candidates separately")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--eval-gpus", type=int, default=8)
    parser.add_argument("--selection", choices=("clean", "full"), default="clean")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--after", type=Path, help="Wait for this supervised four-stage experiment and its complete benchmark")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=2)
    parser.add_argument("--preflight-sample-ids", type=Path)
    parser.add_argument("--train-length-manifest-path", type=Path)
    parser.add_argument("--memory-probe", action="store_true", help="Stress training without saving large disposable checkpoints")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true",
                        help="Audit/train original-image QA only; may combine with dry-run or prepare-only")
    args = parser.parse_args(argv)
    if args.train_only and args.eval_only:
        parser.error("--train-only and --eval-only are mutually exclusive")
    if args.preflight_steps < 1 or ((args.memory_probe or args.preflight_sample_ids) and not args.preflight):
        parser.error("Memory stress settings require --preflight and positive --preflight-steps")
    if (min(args.num_workers, args.val_num_workers) < 0 or args.prefetch_factor < 1
            or not 1 <= args.eval_gpus <= 8 or args.max_new_tokens < 1 or args.poll_seconds <= 0):
        parser.error("Invalid worker, prefetch, evaluation or polling settings")
    for name in ("release", "training_data_root", "suite", "output", "model", "image_root", "captions", "deepspeed_config", "after",
                 "train_length_manifest_path", "preflight_sample_ids"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.after == args.output:
        parser.error("--after and --output must differ")
    if args.preflight:
        args.output = args.output / "preflight"
    inspector = inspect_training_inputs if args.train_only else shared.inspect_inputs
    inputs = inspector(args, check_images=not args.dry_run)
    recipe, steps = build_recipe(args, inputs)
    previous = args.output / "workflow_plan.json"
    binding = (json.loads(previous.read_text())["dependency"] if previous.exists() else bind_dependency(args.after))
    if (binding["output"] if binding else None) != (str(args.after) if args.after else None):
        raise ValueError("The queued predecessor changed; use a fresh --output")
    plan = make_plan(args, inputs, recipe, steps, binding)
    n = inputs["train_records"] if args.dataset == "mixed" else inputs["counts"]["reasoning_v1/train"]
    val = inputs["val_records"] if args.dataset == "mixed" else inputs["counts"]["reasoning_v1/val"]
    print(json.dumps({"train_records": n, "val_records": val, "optimizer_steps": steps,
                      "benchmark_counts": inputs["benchmark_counts"], "after": str(args.after) if args.after else None,
                      "output": str(args.output)}, indent=2), flush=True)
    if args.dry_run:
        print(yaml.safe_dump(recipe, sort_keys=False))
        print(shlex.join(training_command(args, args.output / "recipes/plain_sft.yaml")))
        if not args.train_only:
            for shard in range(args.eval_gpus):
                print(shlex.join(evaluation_command(args, shard)))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if previous.exists() and json.loads(previous.read_text()) != plan:
            raise ValueError("Existing data/config/code changed; use a new --output")
        shared.write_json(previous, plan)
        shared.write_json(args.output / "input_check.json", {"status": "passed", **inputs})
        path = args.output / "recipes/plain_sft.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        if args.prepare_only:
            status(args.output, "prepared", dataset=args.dataset, train_records=n, val_records=val)
            return 0
        dependency_lock = None
        try:
            dependency_lock = wait_for_slot(args, binding)
            fresh_inputs = inspector(args, check_images=False)
            if make_plan(args, fresh_inputs, recipe, steps, binding) != plan:
                raise ValueError("Queued input/code bindings changed while waiting; GPU work was not started")
            import torch
            required_gpus = args.gpus if args.train_only else max(args.gpus, args.eval_gpus)
            if torch.cuda.device_count() < required_gpus:
                raise RuntimeError("Insufficient visible CUDA devices")
            env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LVR_PREFLIGHT_DIAGNOSTICS="0")
            env.pop("LVR_DIAGNOSTICS_DIR", None)
            for key, value in {"OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                               "NCCL_P2P_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo",
                               "NCCL_IB_DISABLE": "1"}.items():
                env.setdefault(key, value)
            if not args.eval_only:
                status(args.output, "training", optimizer_steps=steps)
                run_stage(args, recipe, steps, env)
            else:
                marker = json.loads((args.output / "plain_sft/complete.json").read_text())
                if marker["recipe_sha256"] != shared.digest(recipe):
                    raise ValueError("Unmatched completed plain SFT stage")
                shared.rank_states(args.output / "plain_sft", args.gpus, steps)
            if not args.train_only:
                status(args.output, "evaluating", selection=args.selection)
                run_evaluation(args, env)
            status(args.output, "training_complete" if args.train_only else "complete",
                   preflight=args.preflight, plan_sha256=shared.digest(plan))
        except Exception as exc:
            status(args.output, "dependency_failed" if isinstance(exc, DependencyFailed) else "failed",
                   error=str(exc))
            raise
        finally:
            if dependency_lock is not None:
                dependency_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
