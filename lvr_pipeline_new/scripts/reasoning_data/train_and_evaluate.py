#!/usr/bin/env python3
"""Train four LVR stages on old + reasoning_v1 data, then evaluate new benchmarks."""
from __future__ import annotations

import argparse
from collections import Counter
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
import signal
import subprocess
import sys
import time

import yaml

from lvr.data.records import read_records
from lvr.data.defaults import training_data_root
from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT, file_sha, load_suite


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = import_script("four_stage_runner", ROOT / "scripts/latent_prediction/run.py")
PHASES = legacy.PHASES
MODES = ("predicted", "teacher", "zero", "shuffled")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def split_binding(path):
    path = Path(path)
    index = json.loads((path / "index.json").read_text())
    return {"index_sha256": file_sha(path / "index.json"),
            "shards": {s["file"]: file_sha(path / s["file"]) for s in index["shards"]}}


def verify_overlap_binding(name, path, binding, audited):
    """A filtered QA split can reuse conservative exclusions from its audited superset."""
    if binding == audited:
        return None
    if name not in ("old/train", "old/test"):
        raise ValueError(f"Benchmark overlap audit does not cover current data: {name}")
    original = ROOT / "data/sft" / name.rsplit("/", 1)[1]
    if split_binding(original) != audited:
        raise ValueError(f"Audited source changed; cannot prove a decontaminated subset: {name}")
    before, after = read_records(original), read_records(path)
    source_hashes = Counter(digest(row) for row in before)
    selected_hashes = Counter(digest(row) for row in after)
    if len(after) >= len(before) or selected_hashes - source_hashes:
        raise ValueError(f"Current QA is not an unchanged strict subset of the audited source: {name}")
    return {"status": "verified_strict_subset", "source_path": str(original),
            "source_binding": audited, "current_binding": binding,
            "source_records": len(before), "retained_records": len(after),
            "benchmark_exclusions": "Conservative original audit retained without rewriting its cache"}


def inspect_inputs(args, *, check_images=False):
    """Bind the actual mixed inputs to the benchmark's existing overlap audit."""
    release = json.loads((args.release / "release_manifest.json").read_text())
    if release["status"] != "accepted":
        raise ValueError("Training release is not accepted")
    for name, expected in release["files_sha256"].items():
        if file_sha(args.release / name) != expected:
            raise ValueError(f"Training release file changed: {name}")
    records, manifest = load_suite(args.suite, selection=args.selection)
    overlap = json.loads((args.suite / "overlap_report.json").read_text())
    bindings, counts, subset_checks = {}, {}, {}
    data_root = training_data_root(args, ROOT)
    for name, path in {
        "old/train": data_root / "sft/train", "old/test": data_root / "sft/test",
        "reasoning_v1/train": args.release / "train", "reasoning_v1/val": args.release / "val",
    }.items():
        bindings[name] = split_binding(path)
        subset = verify_overlap_binding(name, path, bindings[name], overlap["bindings"][name])
        if subset:
            subset_checks[name] = subset
        counts[name] = len(read_records(path))
    for split in ("train", "val"):
        if counts["reasoning_v1/" + split] != release["splits"][split]["records"]:
            raise ValueError(f"Training count differs from release manifest: {split}")
    bindings["old/boxes_train"] = split_binding(data_root / "align/train")
    bindings["old/boxes_test"] = split_binding(data_root / "align/test")
    bindings["old/captions"] = file_sha(args.captions)
    if not args.model.is_dir() or not args.image_root.is_dir():
        raise FileNotFoundError("Missing local model or training image root")
    if not (ROOT / ".venv-mentis/bin/python").is_file():
        raise FileNotFoundError("Missing CPU geometry scoring environment: .venv-mentis")
    result = {
        "release_manifest_sha256": file_sha(args.release / "release_manifest.json"),
        "suite_manifest_sha256": file_sha(args.suite / "manifest.json"),
        "counts": counts, "train_records": counts["old/train"] + counts["reasoning_v1/train"],
        "val_records": counts["old/test"] + counts["reasoning_v1/val"],
        "training_bindings": bindings, "benchmark_counts": dict(Counter(r["benchmark"] for r in records)),
        "teacher_records": sum(bool(r["helper_images"]) for r in records),
        "overlap_subset_checks": subset_checks,
        "model_configuration": {p.name: file_sha(p) for p in args.model.iterdir()
                                if p.is_file() and p.suffix in (".json", ".jinja")},
    }
    if check_images:
        from lvr.data.crop_qa import CropQADataset
        paths, ids = set(), set()
        for split, old_split in (("train", "train"), ("val", "test")):
            old = CropQADataset(data_root / "sft" / old_split, data_root / "align" / old_split,
                                caption_path=args.captions)
            new = CropQADataset(args.release / split)
            for item in old.items + new.items:
                if item["id"] in ids:
                    raise ValueError(f"Repeated train/val sample ID: {item['id']}")
                ids.add(item["id"])
                if not item.get("description", "").strip():
                    raise ValueError(f"Missing Description: {item['id']}")
                images = item.get("images") or [{"path": item["question_image"]}]
                for im in images + item.get("auxiliary_images", []):
                    paths.add(args.image_root / im["path"])
            del old, new
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        for name, expected in manifest["asset_sha256"].items():
            if file_sha(args.suite / name) != expected:
                raise ValueError(f"Benchmark image changed: {name}")
        result["checked"] = {"training_ids_and_descriptions": len(ids), "training_image_paths": len(paths),
                             "benchmark_image_hashes": len(manifest["asset_sha256"]),
                             "training_image_scope": "Existence only; full decode/hash evidence belongs to the frozen release."}
    return result


def build_recipes(args, train_records):
    """Preserve the last formal recipe, changing only data, paths and epoch length."""
    if args.gpus < 1 or args.batch_size < 1 or 128 % (args.gpus * args.batch_size):
        raise ValueError("gpus × batch-size must divide global batch 128")
    accumulation = 128 // (args.gpus * args.batch_size)
    steps_per_epoch = math.ceil(math.ceil(math.ceil(train_records / args.gpus) / args.batch_size) / accumulation)
    configs, steps = {}, {}
    data_root = training_data_root(args, ROOT)
    for phase in PHASES:
        config = legacy.recipe(phase)
        config.update(exp_name=str(args.output), version=phase)
        steps[phase] = 2 if args.preflight else min(200, steps_per_epoch) if phase == "warmup" else steps_per_epoch
        config["trainer"].update(devices=args.gpus, accumulate_grad_batches=accumulation,
                                 max_steps=steps[phase], max_epochs=1,
                                 limit_train_batches=steps[phase] * accumulation if phase == "warmup" or args.preflight else 1.0,
                                 strategy={"class_path": "lightning.pytorch.strategies.DeepSpeedStrategy",
                                           "init_args": {"config": str(args.deepspeed_config), "exclude_frozen_parameters": False}})
        config["data"].update(processor_name_or_path=str(args.model), image_root=str(args.image_root),
                              batch_size=args.batch_size, caption_path=str(args.captions),
                              train_path=str(data_root / "sft/train"), val_path=str(data_root / "sft/test"),
                              train_boxes_path=str(data_root / "align/train"), val_boxes_path=str(data_root / "align/test"),
                              additional_train_path=str(args.release / "train"), additional_val_path=str(args.release / "val"))
        for key in ("num_workers", "val_num_workers", "prefetch_factor", "pin_memory"):
            value = getattr(args, key, None)
            if value is not None:
                config["data"][key] = value
        if getattr(args, "host_memory_audit", False):
            config["trainer"]["callbacks"].append({"class_path": "lvr.training.audit.HostMemoryAudit"})
        if phase.startswith("teacher"):
            config["model"].update(encoder_model_name_or_path=str(args.model), qa_model_name_or_path=str(args.model))
        else:
            config["model"].update(source_checkpoint=str(args.output / "teacher_joint/checkpoints/last_full.ckpt"),
                                   base_model_path=str(args.model))
        previous = {"teacher_joint": "teacher_alignment", "joint": "warmup"}.get(phase)
        if previous:
            config["init_checkpoint"] = str(args.output / previous / "checkpoints/last_full.ckpt")
        if args.preflight:
            config["trainer"].update(limit_val_batches=2, val_check_interval=1.0, num_sanity_val_steps=0)
            config["trainer"]["callbacks"].append({"class_path": "lvr.training.reasoning_audit.ReasoningAcceptance"})
        configs[phase] = config
    return configs, steps


def training_command(args, phase, config_file):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={args.gpus}", "--module",
            "lvr.stage1" if phase.startswith("teacher") else "lvr.predict_latents",
            "fit", "--config", str(config_file)]


def rank_states(run, gpus, steps):
    states = [json.loads((run / f"runtime_rank{i}.json").read_text()) for i in range(gpus)]
    if any(s["status"] != "training_complete" or s["optimizer_steps"] != steps for s in states):
        raise RuntimeError(f"Not all ranks completed {steps} updates: {run}")
    return states


def run_stage(args, phase, config, steps, environment):
    run = args.output / phase
    run.mkdir(parents=True, exist_ok=True)
    started, finished = run / "training_started.json", run / "complete.json"
    identity = digest(config)
    if started.exists() and json.loads(started.read_text())["recipe_sha256"] != identity:
        raise ValueError(f"Changed configuration for existing stage: {phase}")
    native = run / "checkpoints/last.ckpt"
    merged = run / "checkpoints/last_full.ckpt"
    if finished.exists():
        if json.loads(finished.read_text())["recipe_sha256"] != identity:
            raise ValueError(f"Changed completed recipe: {phase}")
        rank_states(run, args.gpus, steps)
        legacy.consolidate_deepspeed_checkpoint(native, merged)
        print(f"ALREADY_COMPLETE {phase}", flush=True)
        return
    current = copy.deepcopy(config)
    if started.exists():
        checkpoint = legacy.latest_resume_checkpoint(run / "checkpoints", deepspeed=True)
        if checkpoint is None:
            raise RuntimeError(f"{phase} was interrupted before any recoverable checkpoint. Use a fresh --output; existing logs are retained.")
        current.pop("init_checkpoint", None)
        current["ckpt_path"] = str(checkpoint.resolve())
    for key in ("init_checkpoint",):
        if current.get(key) and not Path(current[key]).is_file():
            raise FileNotFoundError(current[key])
    if phase in ("warmup", "joint") and not Path(current["model"]["source_checkpoint"]).is_file():
        raise FileNotFoundError(current["model"]["source_checkpoint"])
    recipe_path = args.output / "recipes" / f"{phase}.yaml"
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(yaml.safe_dump(current, sort_keys=False))
    command = training_command(args, phase, recipe_path)
    write_json(started, {"recipe_sha256": identity, "command": command})
    print("TRAIN " + shlex.join(command), flush=True)
    with (run / "train.log").open("a") as log:
        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    (run / "train_exit_code").write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"{phase} exited {code}; inspect {run / 'train.log'}. No automatic batch or precision change.")
    states = rank_states(run, args.gpus, steps)
    legacy.consolidate_deepspeed_checkpoint(native, merged, force=True)
    write_json(finished, {"status": "complete", "recipe_sha256": identity, "optimizer_steps": steps,
                          "max_allocated_gib": max(s.get("peak_allocated_gib", 0) for s in states),
                          "max_reserved_gib": max(s.get("peak_reserved_gib", 0) for s in states)})
    print(f"TRAINING_COMPLETE {phase} {steps}", flush=True)


def selected_records(args, mode):
    rows, _ = load_suite(args.suite, selection=args.selection)
    if mode == "teacher":
        rows = [r for r in rows if r["helper_images"]]
    if args.preflight:
        by_task = {}
        for row in rows:
            by_task.setdefault((row["benchmark"], row["task"]), []).append(row)
        rows = [min(group, key=lambda r: r["id"]) for _, group in sorted(by_task.items())]
    return rows


def evaluation_command(args, mode, shard):
    command = [sys.executable, str(ROOT / "scripts/reasoning_benchmarks/evaluate.py"),
               "--suite", str(args.suite), "--selection", args.selection, "--mode", mode,
               "--checkpoint", str(args.output / "joint/checkpoints/last_full.ckpt"),
               "--source-checkpoint", str(getattr(args, "source_checkpoint", None)
                                          or args.output / "teacher_joint/checkpoints/last_full.ckpt"),
               "--model", str(args.model), "--device", f"cuda:{shard}",
               "--output", str(args.output / "benchmark" / mode / f"shard_{shard:02d}"),
               "--shard-count", str(args.eval_gpus), "--shard-index", str(shard),
               "--max-new-tokens", str(args.max_new_tokens)]
    if args.preflight:
        command.extend(["--per-task", "1"])
    if getattr(args, "num_loops", None) is not None:
        command.extend(["--num-loops", str(args.num_loops)])
    return command


def merge_shards(directory, expected_ids, shard_count):
    """Reject missing/duplicate IDs and mixed configurations before aggregating."""
    rows, reference = [], None
    for shard in range(shard_count):
        path = directory / f"shard_{shard:02d}"
        report = json.loads((path / "summary.json").read_text())
        plan = report["plan"]
        if report["status"] != "complete" or plan["shard"] != [shard, shard_count]:
            raise ValueError(f"Incomplete/wrong evaluation shard: {path}")
        shared = {k: v for k, v in plan.items() if k not in ("ids", "samples", "shard", "benchmarks")}
        if reference is not None and shared != reference:
            raise ValueError("Evaluation shards used different model/data/code configurations")
        reference = shared
        part = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        if (len(part) != report["samples"] or len(part) != plan["samples"]
                or [r["id"] for r in part] != plan["ids"]):
            # Resumed journals may have a different order, but still need exact set equality.
            if (len(part) != report["samples"] or len(part) != plan["samples"]
                    or {r["id"] for r in part} != set(plan["ids"])):
                raise ValueError(f"Prediction/plan ID mismatch: {path}")
        rows.extend(part)
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected_ids):
        raise ValueError("Missing, duplicate or unexpected evaluation IDs across shards")
    rows = [by_id[i] for i in expected_ids]
    # Reuse the evaluator's aggregation; it imports only local utilities until main().
    directory_scripts = str(ROOT / "scripts/reasoning_benchmarks")
    sys.path.insert(0, directory_scripts)
    try:
        evaluator = import_script("reasoning_evaluator", Path(directory_scripts) / "evaluate.py")
    finally:
        sys.path.remove(directory_scripts)
    report = {"status": "complete", "mode": reference["mode"], "selection": reference["selection"],
              "scope": "acceptance_subset" if reference["per_task"] else "full_selected_protocol",
              "samples": len(rows), "metrics": evaluator.summary(rows), "shared_plan": reference,
              "shard_count": shard_count}
    (directory / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_json(directory / "summary.json", report)
    return report


def run_evaluation(args, environment, *, modes=MODES):
    reports = []
    for mode in modes:
        directory = args.output / "benchmark" / mode
        directory.mkdir(parents=True, exist_ok=True)
        children = []
        try:
            for shard in range(args.eval_gpus):
                command = evaluation_command(args, mode, shard)
                log = (directory / f"shard_{shard:02d}.log").open("a")
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    log.close()
                children.append(child)
            while any(p.poll() is None for p in children):
                failed = [p.returncode for p in children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Evaluation {mode} failed {failed}; inspect {directory}")
                time.sleep(1)
            if any(p.returncode for p in children):
                raise RuntimeError(f"Evaluation {mode} failed; inspect {directory}")
        finally:
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in children:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        report = merge_shards(directory, [r["id"] for r in selected_records(args, mode)], args.eval_gpus)
        reports.append(report)
        print(f"EVALUATION_COMPLETE {mode} {report['samples']}", flush=True)
    write_json(args.output / "benchmark/summary.json", {"status": "complete", "modes": reports})
    with (args.output / "benchmark/results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "benchmark", "n", "correct", "accuracy", "parse_failures", "truncated"])
        writer.writeheader()
        for report in reports:
            for metric in report["metrics"]:
                if metric["task"] == "all":
                    writer.writerow({key: report["mode"] if key == "mode" else metric[key] for key in writer.fieldnames})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "data/processed/reasoning_v1")
    parser.add_argument("--training-data-root", type=Path, default=ROOT / "data/default_training",
                        help="Old-data release root; explicit data preserves historical reproduction")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reasoning_mixed_v1_8gpu")
    parser.add_argument("--model", type=Path, default=Path("/workspace/wmz/.cache/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/wmz/lvr_pipeline_new/data"))
    parser.add_argument("--captions", type=Path, default=ROOT / "data/latent_prediction/captions.jsonl.gz")
    parser.add_argument("--deepspeed-config", type=Path, default=ROOT / "configs/latent_prediction/deepspeed_zero3_80gb.json")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, help="Training DataLoader workers per rank")
    parser.add_argument("--val-num-workers", type=int, help="Validation workers per rank; 0 avoids a second worker pool")
    parser.add_argument("--prefetch-factor", type=int, help="Batches prefetched per worker")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable pinned host batch storage")
    parser.add_argument("--host-memory-audit", action="store_true",
                        help="Record per-rank host memory at training, validation and checkpoint boundaries")
    parser.add_argument("--eval-gpus", type=int, default=8)
    parser.add_argument("--selection", choices=("clean", "full"), default="clean")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--preflight", action="store_true", help="Isolated 2 updates/stage plus small benchmark; output/preflight")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="Inspect recipes/commands without writes or GPU use")
    actions.add_argument("--prepare-only", action="store_true", help="Validate inputs and save recipes/plan without GPU use")
    actions.add_argument("--check-data-only", action="store_true", help="Check mixed labels/images and benchmark hashes without training")
    actions.add_argument("--eval-only", action="store_true", help="Evaluate this output's completed final checkpoints")
    actions.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    if any(value is not None and value < 0 for value in (args.num_workers, args.val_num_workers)):
        parser.error("worker counts must be nonnegative")
    if args.prefetch_factor is not None and args.prefetch_factor < 1:
        parser.error("prefetch-factor must be positive")
    for key in ("release", "training_data_root", "suite", "output", "model", "image_root", "captions", "deepspeed_config"):
        setattr(args, key, getattr(args, key).resolve())
    if args.preflight:
        args.output = args.output / "preflight"
    if not 1 <= args.eval_gpus <= 8 or args.max_new_tokens < 1:
        parser.error("eval-gpus must be 1..8 and max-new-tokens positive")
    inputs = inspect_inputs(args, check_images=not args.dry_run)
    configs, steps = build_recipes(args, inputs["train_records"])
    plan = {"format": "lvr-mixed-reasoning-workflow-v1", "inputs": {k: v for k, v in inputs.items() if k != "checked"},
            "recipes": configs, "optimizer_steps": steps, "selection": args.selection,
            "preflight": args.preflight, "eval_gpus": args.eval_gpus, "max_new_tokens": args.max_new_tokens,
            "code": {str(p.relative_to(ROOT)): file_sha(p) for p in
                     [Path(__file__), *sorted((ROOT / "src/lvr").rglob("*.py")),
                      ROOT / "scripts/latent_prediction/run.py", ROOT / "scripts/reasoning_benchmarks/evaluate.py"]},
            "deepspeed_sha256": file_sha(args.deepspeed_config)}
    print(json.dumps({"train_records": inputs["train_records"], "val_records": inputs["val_records"],
                      "optimizer_steps": steps, "benchmark_counts": inputs["benchmark_counts"],
                      "teacher_records": inputs["teacher_records"], "output": str(args.output)}, indent=2), flush=True)
    if args.dry_run:
        for phase, config in configs.items():
            print(f"--- {phase} ---\n" + yaml.safe_dump(config, sort_keys=False))
            print(shlex.join(training_command(args, phase, args.output / "recipes" / f"{phase}.yaml")))
        for mode in MODES:
            for shard in range(args.eval_gpus):
                print(shlex.join(evaluation_command(args, mode, shard)))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = args.output / "workflow_plan.json"
        if binding.exists() and json.loads(binding.read_text()) != plan:
            raise ValueError("Existing workflow data/config/code changed; use a new --output")
        write_json(binding, plan)
        write_json(args.output / "input_check.json", {"status": "passed", **inputs})
        for phase, config in configs.items():
            path = args.output / "recipes" / f"{phase}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            # A resumed recipe may contain ckpt_path. Never rewrite it for a read-only preparation.
            if not path.exists():
                path.write_text(yaml.safe_dump(config, sort_keys=False))
        if args.prepare_only or args.check_data_only:
            return 0
        import torch
        if torch.cuda.device_count() < (args.eval_gpus if args.eval_only else max(args.gpus, args.eval_gpus)):
            raise RuntimeError("Insufficient visible CUDA devices for this workflow")
        environment = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           LVR_PREFLIGHT_DIAGNOSTICS="0")
        environment.pop("LVR_DIAGNOSTICS_DIR", None)
        for key, value in {"OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                           "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1"}.items():
            environment.setdefault(key, value)
        if not args.eval_only:
            for phase in PHASES:
                run_stage(args, phase, configs[phase], steps[phase], environment)
        else:
            for phase in PHASES:
                run = args.output / phase
                complete = json.loads((run / "complete.json").read_text())
                if complete["recipe_sha256"] != digest(configs[phase]):
                    raise ValueError(f"Unmatched completed stage: {phase}")
                rank_states(run, args.gpus, steps[phase])
        if not args.train_only:
            run_evaluation(args, environment)
        write_json(args.output / "workflow_status.json", {"status": "training_complete" if args.train_only else "complete",
                                                          "preflight": args.preflight, "plan_sha256": digest(plan)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
