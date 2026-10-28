#!/usr/bin/env python3
"""Continue no-SAT step 696 through four epochs, evaluating all 11 benchmarks each epoch."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE / "lvr_pipeline_new"
SOURCE = ROOT.parent / "2026-10-26-qwen25-no-sat-ablation"
SOURCE_RUN = SOURCE / "runs/main/plain_sft"
SOURCE_CHECKPOINT = SOURCE_RUN / "checkpoints/candidate-000696.ckpt"
SUITE = ROOT.parent / "2026-10-23-unified-benchmark/data"
TRAIN_RECORDS = 172224
GPUS = 4
BATCH_SIZE = 16
ACCUMULATION = 2
START_STEP = 696
STEPS_PER_EPOCH = math.ceil(TRAIN_RECORDS / (GPUS * BATCH_SIZE * ACCUMULATION))
EPOCH_STEPS = [STEPS_PER_EPOCH * epoch for epoch in range(1, 5)]
HALF_STEPS = [STEPS_PER_EPOCH * index // 2 for index in range(1, 9)
              if STEPS_PER_EPOCH * index // 2 > START_STEP]


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Bound inputs changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run_dir(epoch):
    return ROOT / f"runs/epoch_{epoch:02d}/plain_sft"


def epoch_checkpoint(epoch):
    return run_dir(epoch) / "checkpoints" / f"candidate-{EPOCH_STEPS[epoch-1]:06d}.ckpt"


def export_path(epoch):
    return ROOT / "runs/exports" / f"epoch-{epoch:02d}-step-{EPOCH_STEPS[epoch-1]:06d}_full.ckpt"


def benchmark_dir(epoch):
    return ROOT / f"results/benchmark/epoch_{epoch:02d}"


def native_info(path):
    """Read ZeRO client metadata; tensor storage is mmap'ed, not copied."""
    import torch
    from continuation_runtime import native_state, original_lr
    path = Path(path).resolve()
    saved = native_state(path)
    step = int(saved["global_step"])
    if saved["dp_world_size"] != GPUS or int(saved["global_steps"]) != step:
        raise ValueError(f"Expected consistent four-rank ZeRO optimizer state: {path}")
    if saved["hyper_parameters"]["lr_schedule_steps"] != 2784:
        raise ValueError("Checkpoint LR horizon changed")
    schedule = saved["lr_schedulers"][0]
    if schedule["last_epoch"] != step or not math.isclose(schedule["_last_lr"][0], original_lr(step), rel_tol=1e-9):
        raise ValueError("Checkpoint step and original LR schedule disagree")
    tag = (path / "latest").read_text().strip()
    partition_info = []
    for rank in range(GPUS):
        states_path = path / tag / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        optimizer_path = path / tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        client = torch.load(states_path, map_location="cpu", mmap=True, weights_only=False)
        if int(client["global_step"]) != step or not optimizer_path.is_file() or optimizer_path.stat().st_size < 1:
            raise ValueError(f"Incomplete checkpoint partition {rank}: {path}")
        partition_info.append({"rank": rank, "optimizer_bytes": optimizer_path.stat().st_size})
        del client
    fit = saved["loops"]["fit_loop"]
    return {"path": str(path), "step": step, "epoch": saved["epoch"],
            "learning_rate": schedule["_last_lr"][0], "partitions": partition_info,
            "processed_batches_current_epoch": fit["epoch_loop.batch_progress"]["current"]["processed"],
            "is_last_batch": fit["epoch_loop.batch_progress"]["is_last_batch"],
            "files": {str(p.relative_to(path)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in sorted(path.rglob("*")) if p.is_file()}}


def build_recipe(epoch, resume, args):
    recipe = yaml.safe_load((SOURCE / "configs/plain_sft.yaml").read_text())
    recipe = copy.deepcopy(recipe)
    target = EPOCH_STEPS[epoch-1]
    directory = run_dir(epoch)
    recipe.update(exp_name=str(directory.parent), version="plain_sft", ckpt_path=str(resume))
    recipe.pop("init_checkpoint", None)
    recipe["trainer"].update(max_epochs=epoch, max_steps=target, num_sanity_val_steps=0,
                             use_distributed_sampler=False, default_root_dir=str(directory))
    recipe["data"].update(num_workers=args.num_workers, val_num_workers=2,
                          prefetch_factor=args.prefetch_factor, persistent_workers=True)
    for logger in recipe["trainer"]["logger"]:
        logger["init_args"]["save_dir"] = str(directory)
    recipe["trainer"]["callbacks"] = [
        {"class_path": "continuation_runtime.ContinuationAudit", "init_args": {
            "resume_checkpoint": str(resume), "baseline_run": str(SOURCE_RUN), "end_step": target}},
        {"class_path": "lvr.training.audit.HostMemoryAudit"},
        {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor", "init_args": {"logging_interval": "step"}},
        {"class_path": "continuation_runtime.HalfEpochModelCheckpoint", "init_args": {
            "steps": [step for step in HALF_STEPS if step % STEPS_PER_EPOCH],
            "filename": "candidate-{step:06d}", "save_top_k": -1, "save_last": False,
            "dirpath": str(directory / "checkpoints"), "auto_insert_metric_name": False,
            "enable_version_counter": False}},
        {"class_path": "continuation_runtime.EpochModelCheckpoint", "init_args": {
            "steps_per_epoch": STEPS_PER_EPOCH, "dirpath": str(directory / "checkpoints")}},
        {"class_path": "continuation_runtime.RecoveryModelCheckpoint", "init_args": {
            "dirpath": str(directory / "checkpoints")}},
    ]
    return recipe


def environment():
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'script'}:{PROJECT / 'src'}", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               LVR_PREFLIGHT_DIAGNOSTICS="0")
    cache = ROOT / "runs/cache"
    hf = ROOT.parent / "models/.cache/huggingface"
    for name, path in {"HF_HOME": hf, "HF_HUB_CACHE": hf / "hub", "HUGGINGFACE_HUB_CACHE": hf / "hub",
                       "HF_DATASETS_CACHE": hf / "datasets", "TRANSFORMERS_CACHE": hf / "hub",
                       "TMPDIR": ROOT.parent / ".tmp", "TORCH_HOME": cache / "torch",
                       "TORCH_EXTENSIONS_DIR": cache / "torch_extensions", "TRITON_CACHE_DIR": cache / "triton",
                       "XDG_CACHE_HOME": cache / "xdg", "MPLCONFIGDIR": cache / "matplotlib"}.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def prepare(args, bind):
    old = module("no_sat_source", SOURCE / "script/experiment.py")
    old.verify()
    info = native_info(SOURCE_CHECKPOINT)
    if info["step"] != START_STEP or info["processed_batches_current_epoch"] != START_STEP * ACCUMULATION:
        raise ValueError("Unexpected half-epoch source cursor")
    from lvr.data.crop_qa import CropQADataset
    from lvr.data.length_manifest import load_plain_qa_lengths
    from lvr.eval.paper_benchmark import load_suite
    from continuation_runtime import original_lr
    recipe = yaml.safe_load((SOURCE / "configs/plain_sft.yaml").read_text())
    data = recipe["data"]
    items = CropQADataset(data["train_path"], data["train_boxes_path"]).items
    items += CropQADataset(data["additional_train_path"]).items
    if len(items) != TRAIN_RECORDS or any(item.get("source") == "sat_perspective" for item in items):
        raise ValueError("Expected the existing no-SAT training data")
    lengths = load_plain_qa_lengths(data["train_length_manifest_path"], items, data["processor_name_or_path"],
                                   data["original_pixels"], data["max_length"])
    from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
    sampler_hashes = {}
    for rank in range(GPUS):
        sampler = DistributedLengthGroupedSampler(batch_size=BATCH_SIZE * GPUS, num_replicas=GPUS,
                       rank=rank, seed=data["seed"], lengths=lengths)
        sampler_hashes[str(rank)] = digest(list(sampler))
        if sampler_hashes[str(rank)] != read(SOURCE_RUN / f"runtime_rank{rank}.json")["sampler_indices_sha256"]:
            raise ValueError(f"Original sample order changed on rank {rank}")
    rows, protocol, _ = load_suite(SUITE)
    if len(protocol["benchmarks"]) != 11 or len(rows) != 10107 or protocol["total_records"] != 9957:
        raise ValueError("Expected the full eleven-benchmark suite")
    files = [*sorted((ROOT / "script").glob("*.py")), ROOT / "script/run.sh",
             SOURCE / "configs/plain_sft.yaml", Path(data["additional_train_path"]),
             Path(data["train_length_manifest_path"]), SUITE / "manifest.json", SUITE / "protocol.json",
             PROJECT / "scripts/latent_prediction/run.py",
             PROJECT / "scripts/reasoning_data/validate_plain_sft_candidates.py",
             *sorted((PROJECT / "scripts/paper_benchmarks").glob("*.py")),
             *sorted((PROJECT / "src/lvr").rglob("*.py"))]
    plan = {"source_checkpoint": info, "train_records": len(items), "start_step": START_STEP,
            "train_gpus": GPUS, "microbatch": BATCH_SIZE, "accumulation": ACCUMULATION,
            "global_batch": GPUS * BATCH_SIZE * ACCUMULATION, "steps_per_epoch": STEPS_PER_EPOCH,
            "target_epochs": 4, "target_step": EPOCH_STEPS[-1], "half_epoch_checkpoints": HALF_STEPS,
            "benchmark_steps": EPOCH_STEPS, "benchmarks": list(protocol["benchmarks"]),
            "benchmark_records": 9957, "benchmark_generations_per_epoch": len(rows),
            "num_workers_per_gpu": args.num_workers, "prefetch_factor": args.prefetch_factor,
            "original_sampler_indices_sha256": sampler_hashes,
            "lr_schedule_steps": 2784, "lr_policy": "preserve_original_warmup_cosine_then_hold_1e-6",
            "milestone_learning_rates": {str(s): original_lr(s) for s in [START_STEP, *EPOCH_STEPS]},
            "epoch_tail": "Each rank's last 16 samples are split into 8+8 for accumulation 2; no duplicates or drops. "
                          "The last optimizer update in each epoch has 64 samples globally; others have 128.",
            "test_benchmarks_used_for_selection": False,
            "files_sha256": {str(p): file_sha(p) for p in files}}
    for epoch in range(1, 5):
        resume = SOURCE_CHECKPOINT if epoch == 1 else epoch_checkpoint(epoch-1)
        config = build_recipe(epoch, resume, args)
        path = ROOT / "configs" / f"epoch_{epoch:02d}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != payload and (ROOT / "configs/plan.json").exists():
            raise ValueError(f"Bound recipe changed: {path}")
        if not path.exists() or path.read_text() != payload:
            path.write_text(payload)
    if bind:
        save(ROOT / "configs/plan.json", plan, immutable=True)
    else:
        save(ROOT / "runs/checks/plan_preview.json", plan)
    return plan


def training_command(recipe):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4",
            str(ROOT / "script/train.py"), "fit", "--config", str(recipe)]


def benchmark_command(epoch):
    model = yaml.safe_load((SOURCE / "configs/plain_sft.yaml").read_text())["model"]["qa_model_name_or_path"]
    return [sys.executable, str(PROJECT / "scripts/paper_benchmarks/evaluate.py"), "--suite", str(SUITE),
            "--model", model, "--mode", "plain", "--checkpoint", str(export_path(epoch)),
            "--expected-step", str(EPOCH_STEPS[epoch-1]), "--gpus", str(GPUS), "--output", str(benchmark_dir(epoch))]


def latest_resume(epoch):
    previous = SOURCE_CHECKPOINT if epoch == 1 else epoch_checkpoint(epoch-1)
    candidates = [previous]
    directory = run_dir(epoch) / "checkpoints"
    if directory.exists():
        candidates += [p for p in directory.glob("*.ckpt") if p.is_dir() and not p.is_symlink()]
    valid = []
    for path in candidates:
        try:
            info = native_info(path)
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            if path == previous:
                raise
            print(f"Incomplete checkpoint ignored: {path}: {error}", flush=True)
            continue
        if START_STEP <= info["step"] <= EPOCH_STEPS[epoch-1]:
            valid.append((info["step"], path))
    if not valid:
        raise RuntimeError("No complete native checkpoint is available for full-state resume")
    return max(valid, key=lambda item: item[0])


def gpu_check():
    import torch
    if torch.cuda.device_count() < GPUS:
        raise RuntimeError("This workflow needs four visible GPUs; use --dry-run on a CPU terminal")
    check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    if check.stdout.strip():
        raise RuntimeError("GPUs already have compute processes; launch on an idle four-GPU task")


def status(state, **details):
    save(ROOT / "runs/status.json", {"status": state, "updated_at_unix": time.time(), **details})


def train_epoch(epoch, args, env):
    target = EPOCH_STEPS[epoch-1]
    step, resume = latest_resume(epoch)
    if step < target:
        recipe = build_recipe(epoch, resume, args)
        # Keep earlier train/val curves when restarting within the same epoch.
        attempt = f"from_step_{step:06d}_{time.time_ns()}"
        for logger in recipe["trainer"]["logger"]:
            logger["init_args"]["version"] += "_" + attempt
        directory = run_dir(epoch)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory.parent / "resume_recipe.yaml"
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        save(directory / "training_started.json", {"resume_checkpoint": str(resume), "start_step": step,
             "target_step": target, "recipe_sha256": digest(recipe), "at_unix": time.time()})
        status("training", epoch=epoch, start_step=step, target_step=target)
        print(f"TRAIN epoch={epoch} step={step}->{target} log={directory / 'train.log'}", flush=True)
        with (directory / "train.log").open("a") as log:
            result = subprocess.run(training_command(path), cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (directory / "train_exit_code").write_text(f"{result.returncode}\n")
        if result.returncode:
            raise RuntimeError(f"Training failed: inspect {directory / 'train.log'}")
        for rank in range(GPUS):
            audit = read(directory / f"continuation_rank{rank}.json")
            if audit["status"] != "training_complete" or audit["optimizer_steps"] != target:
                raise RuntimeError("Not all ranks reached the requested epoch boundary")
    info = native_info(epoch_checkpoint(epoch))
    if info["step"] != target or not info["is_last_batch"]:
        raise RuntimeError("Missing a complete epoch-boundary checkpoint")
    save(run_dir(epoch) / "complete.json", {"status": "complete", "epoch": epoch,
         "optimizer_steps": target, "checkpoint": str(epoch_checkpoint(epoch)), "learning_rate": info["learning_rate"]})


def evaluate_epoch(epoch, env):
    status("exporting", epoch=epoch, step=EPOCH_STEPS[epoch-1])
    validator = module("continuation_export", PROJECT / "scripts/reasoning_data/validate_plain_sft_candidates.py")
    legacy = module("continuation_legacy", PROJECT / "scripts/latent_prediction/run.py")
    validator.prepare_export(epoch_checkpoint(epoch), export_path(epoch), legacy.consolidate_deepspeed_checkpoint)
    status("benchmark", epoch=epoch, step=EPOCH_STEPS[epoch-1])
    print(f"BENCHMARK epoch={epoch} step={EPOCH_STEPS[epoch-1]} output={benchmark_dir(epoch)}", flush=True)
    subprocess.run(benchmark_command(epoch), cwd=PROJECT, env=env, check=True)
    summary = read(benchmark_dir(epoch) / "summary.json")
    config = read(benchmark_dir(epoch) / "run_config.json")
    if (summary["status"] != "complete" or summary["generations"] != 10107
            or len(summary["metrics"]) != 11 or config["expected_step"] != EPOCH_STEPS[epoch-1]
            or Path(config["checkpoint"]) != export_path(epoch).resolve()):
        raise RuntimeError("The full benchmark is incomplete or refers to the wrong checkpoint")


def summarize():
    rows = []
    for epoch in range(1, 5):
        path = benchmark_dir(epoch) / "summary.json"
        if not path.exists():
            continue
        report = read(path)
        if report["status"] != "complete":
            continue
        for metric in report["metrics"]:
            rows.append({"epoch": epoch, "optimizer_steps": EPOCH_STEPS[epoch-1],
                         "benchmark": metric["benchmark"], "accuracy_pct": 100 * metric["accuracy"],
                         "records": metric["records"], "parse_failures": metric["parse_failures"]})
    if rows:
        target = ROOT / "results/benchmark_by_epoch.csv"
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save(ROOT / "results/benchmark_by_epoch.json", {"rows": rows, "test_benchmarks_used_for_selection": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "prepare", "summarize"), nargs="?", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    args = parser.parse_args()
    if args.num_workers < 1 or args.prefetch_factor < 1:
        parser.error("Worker count and prefetch factor must be positive")
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "runs/workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "summarize":
            summarize()
            return
        plan = prepare(args, bind=not args.dry_run)
        print(json.dumps({key: plan[key] for key in ("start_step", "target_step", "steps_per_epoch",
              "half_epoch_checkpoints", "benchmark_steps", "num_workers_per_gpu", "lr_policy")}, indent=2), flush=True)
        if args.dry_run:
            for epoch in range(1, 5):
                print(shlex.join(training_command(ROOT / "configs" / f"epoch_{epoch:02d}.yaml")), flush=True)
                print(shlex.join(benchmark_command(epoch)), flush=True)
            return
        if args.action == "prepare":
            return
        env = environment()
        gpu_check()
        try:
            for epoch in range(1, 5):
                train_epoch(epoch, args, env)
                evaluate_epoch(epoch, env)
                summarize()
            status("complete", optimizer_steps=EPOCH_STEPS[-1], epochs=4)
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
