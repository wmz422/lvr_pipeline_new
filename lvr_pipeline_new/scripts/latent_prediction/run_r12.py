#!/usr/bin/env python3
"""R12, weight 0.1: three training stages (student includes warmup), then 11 benchmarks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("r12_mixed_workflow", ROOT / "scripts/reasoning_data/train_and_evaluate.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)

STAGES = {
    "1_teacher_alignment": ["teacher_alignment"],
    "2_teacher_joint": ["teacher_joint"],
    "3_student": ["warmup", "joint"],
}
ANSWER_POLICY = "structured_v1"
MIRA_POLICY = "task_normalized_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/r12_reconstruction_0p1")
    parser.add_argument("--model", type=Path, default=Path("/workspace/wmz/.cache/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/wmz/lvr_pipeline_new/data"))
    parser.add_argument("--release", type=Path, default=ROOT / "data/processed/reasoning_v1")
    parser.add_argument("--training-data-root", type=Path, default=ROOT / "data/default_training")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/latent_prediction/captions.jsonl.gz")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmark/reasoning_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--bench-root", type=Path,
                        default=Path("/workspace/wmz/lvr_pipeline_new/benchmark/bench"))
    parser.add_argument("--vlmeval-root", type=Path, default=ROOT / "third_party/VLMEvalKit")
    parser.add_argument("--deepspeed-config", type=Path,
                        default=ROOT / "configs/latent_prediction/deepspeed_zero3_80gb.json")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--eval-gpus", type=int, help="Defaults to --gpus")
    parser.add_argument("--batch-size", type=int, default=16, help="Per GPU; accumulation keeps global batch 128")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="Print recipes/commands without writes or GPU use")
    actions.add_argument("--prepare-only", action="store_true", help="Validate local inputs and save recipes; CPU only")
    actions.add_argument("--eval-only", action="store_true", help="Evaluate completed R12 training in the same output")
    actions.add_argument("--train-only", action="store_true", help="Stop after student joint; evaluate later with --eval-only")
    args = parser.parse_args(argv)
    args.eval_gpus = args.gpus if args.eval_gpus is None else args.eval_gpus
    if args.gpus < 1 or args.batch_size < 1 or 128 % (args.gpus * args.batch_size):
        parser.error("gpus × batch-size must divide global batch 128")
    if not 1 <= args.eval_gpus <= 8:
        parser.error("eval-gpus must be 1..8")
    if min(args.num_workers, args.val_num_workers) < 0 or args.prefetch_factor < 1:
        parser.error("worker counts must be nonnegative; prefetch-factor must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    # Match the latest report's clean protocol and generation settings.
    args.selection, args.max_new_tokens = "clean", 512
    args.preflight, args.host_memory_audit = False, True
    return args


def build_recipes(args, train_records):
    configs, steps = workflow.build_recipes(args, train_records)
    for phase in ("teacher_alignment", "teacher_joint"):
        configs[phase]["model"].update(
            reconstruction_weight=0.1, reconstruction_normalize=True,
            reconstruction_hidden_size=512, reconstruction_num_layers=2, reconstruction_num_heads=8,
        )
    # Student recipes, including joint's initialization from warmup, stay intact.
    # stage1.py derives data.include_reconstruction from model.reconstruction_weight.
    return configs, steps


def inspect_old_benchmarks(args):
    from lvr.eval.local_protocol import SUPPORTED, source_files

    datasets = json.loads((args.protocol / "datasets.json").read_text())
    if set(datasets) != set(SUPPORTED):
        raise ValueError("R12 requires the complete old five-benchmark protocol")
    files = {p for name in datasets for p in source_files(args.bench_root, name)}
    if not (args.vlmeval_root / "run.py").is_file():
        raise FileNotFoundError(args.vlmeval_root / "run.py")
    return {
        "counts": {name: entry["samples"] for name, entry in datasets.items()},
        "protocol": str(args.protocol), "bench_root": str(args.bench_root),
        "vlmeval_root": str(args.vlmeval_root),
        "protocol_sha256": {name: workflow.file_sha(args.protocol / name)
                            for name in ("datasets.json", "data_config.json", "crop_annotations.json")},
        "source_files": {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                         for p in sorted(files)},
    }


def old_evaluation_command(args):
    command = [sys.executable, str(ROOT / "scripts/latent_prediction/evaluate.py"),
            "--checkpoint", str(args.output / "joint/checkpoints/last_full.ckpt"),
            "--source-checkpoint", str(getattr(args, "source_checkpoint", None)
                                       or args.output / "teacher_joint/checkpoints/last_full.ckpt"),
            "--base-model", str(args.model), "--output", str(args.output / "benchmark_old"),
            "--protocol", str(args.protocol), "--bench-root", str(args.bench_root),
            "--vlmeval-root", str(args.vlmeval_root), "--gpus", str(args.eval_gpus),
            "--processor-default-resolution"]
    if getattr(args, "num_loops", None) is not None:
        command.extend(["--num-loops", str(args.num_loops)])
    return command


def rescore_command(args, mode, output):
    return [str(ROOT / ".venv-mentis/bin/python"), str(ROOT / "scripts/reasoning_benchmarks/rescore.py"),
            "--input", str(args.output / "benchmark" / mode / "predictions.jsonl"),
            "--output", str(output), "--suite", str(args.suite), "--selection", args.selection,
            "--answer-policy", ANSWER_POLICY, "--mira-policy", MIRA_POLICY]


def run_logged(command, log_path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + shlex.join(command), flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log_path}")


def validate_rescore(args, mode, directory):
    """Only reuse complete scores tied to the exact strict predictions."""
    report = json.loads((directory / "summary.json").read_text())
    source = args.output / "benchmark" / mode / "predictions.jsonl"
    if (report["status"] != "passed" or report["input"] != str(source)
            or report["input_sha256"] != workflow.file_sha(source)
            or report["answer_policy"] != ANSWER_POLICY or report["mira_policy"] != MIRA_POLICY):
        raise ValueError(f"Changed or incomplete rescore: {directory}")
    for name, expected in report["code_sha256"].items():
        if workflow.file_sha(ROOT / name) != expected:
            raise ValueError(f"Rescore code changed: {name}; use a fresh --output")
    records = workflow.selected_records(args, mode)
    rows = [json.loads(line) for line in (directory / "predictions_rescored.jsonl").read_text().splitlines()]
    if (len(rows) != report["samples"] or len(rows) != len(records)
            or {r["id"] for r in rows} != {r["id"] for r in records}
            or any(r["mode"] != mode for r in rows)):
        raise ValueError(f"Missing, duplicate or unexpected rescored IDs: {directory}")
    expected_counts = Counter(r["benchmark"] for r in records)
    metrics = report["metrics"]
    if (len(metrics) != len(expected_counts)
            or {m["benchmark"]: m["n"] for m in metrics} != dict(expected_counts)):
        raise ValueError(f"Incomplete rescore metrics: {directory}")
    return report


def run_rescoring(args, environment, *, modes=workflow.MODES):
    parent = args.output / "benchmark_latest"
    parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in modes:
        output = parent / mode
        if not output.exists():
            # Promote only validated results. Interrupted attempts remain separate for diagnosis.
            temporary = Path(tempfile.mkdtemp(prefix=f".{mode}-", dir=parent))
            run_logged(rescore_command(args, mode, temporary), parent / f"{mode}.log", environment)
            validate_rescore(args, mode, temporary)
            temporary.rename(output)
        reports[mode] = validate_rescore(args, mode, output)
    return reports


def collect_results(args, inputs, latest, *, experiment="R12", stages=None, modes=workflow.MODES):
    """Keep old/new aggregates separate; teacher's helper subset is not a six-task score."""
    rows, native = [], {}
    for mode in modes:
        old = json.loads((args.output / "benchmark_old" / f"summary_{mode}.json").read_text())["results"]
        if set(old) != set(inputs["old_benchmarks"]["counts"]):
            raise ValueError(f"Incomplete old benchmark results: {mode}")
        native[mode] = old
        for name, n in inputs["old_benchmarks"]["counts"].items():
            entry = old[name]
            if entry["samples"] != n:
                raise ValueError(f"Old benchmark sample count changed: {mode}/{name}")
            correct = entry["correct_questions"]
            rows.append(dict(mode=mode, group="old5", benchmark=name, status="complete", n=n,
                             correct=correct, accuracy_percent=100 * correct / n,
                             strict_correct=correct, strict_accuracy_percent=100 * correct / n,
                             policy="original", parse_failures=entry["unmatched_answers"],
                             truncated=entry["truncated_predictions"]))
        new = {m["benchmark"]: m for m in latest[mode]["metrics"]}
        teacher_counts = Counter(r["benchmark"] for r in workflow.selected_records(args, "teacher"))
        expected = teacher_counts if mode == "teacher" else inputs["benchmark_counts"]
        if set(new) != set(expected):
            raise ValueError(f"Incomplete new benchmark results: {mode}")
        for name in inputs["benchmark_counts"]:
            if name not in new:
                rows.append(dict(mode=mode, group="new6", benchmark=name, status="teacher_unavailable",
                                 n=0, policy=f"{ANSWER_POLICY}+{MIRA_POLICY}"))
                continue
            m = new[name]
            if m["n"] != expected[name]:
                raise ValueError(f"New benchmark sample count changed: {mode}/{name}")
            rows.append(dict(mode=mode, group="new6", benchmark=name, status="complete", n=m["n"],
                             correct=m["rescored_correct"], accuracy_percent=m["rescored_accuracy_percent"],
                             strict_correct=m["strict_correct"], strict_accuracy_percent=m["strict_accuracy_percent"],
                             policy=f"{ANSWER_POLICY}+{MIRA_POLICY}",
                             parse_failures=m["rescored_parse_failures"], truncated=None))
    aggregates = []
    for mode in modes:
        for group in ("old5", "new6"):
            subset = [r for r in rows if r["mode"] == mode and r["group"] == group and r["status"] == "complete"]
            n = sum(r["n"] for r in subset)
            correct, strict = (sum(r[key] for r in subset) for key in ("correct", "strict_correct"))
            aggregates.append(dict(mode=mode, group="new_helper_subset" if mode == "teacher" and group == "new6" else group,
                                   n=n, correct=correct, accuracy_percent=100 * correct / n,
                                   strict_correct=strict, strict_accuracy_percent=100 * strict / n))
    report = dict(status="complete", experiment=experiment, stages=STAGES if stages is None else stages,
                  results=rows, aggregates=aggregates,
                  old_native_metrics=native,
                  scoring=dict(old="original", answer_policy=ANSWER_POLICY, mira_policy=MIRA_POLICY,
                               note="New policies are non-official format normalization; strict scores are retained."))
    if getattr(args, "num_loops", None) is not None:
        report["prediction_num_loops"] = args.num_loops
    workflow.write_json(args.output / "benchmark_summary.json", report)
    with (args.output / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "group", "benchmark", "status", "n", "correct",
                                "accuracy_percent", "strict_correct", "strict_accuracy_percent", "policy",
                                "parse_failures", "truncated"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def execute(args, inputs, configs, steps, environment):
    if not args.eval_only:
        for phase in workflow.PHASES:
            workflow.run_stage(args, phase, configs[phase], steps[phase], environment)
    else:
        for phase in workflow.PHASES:
            run = args.output / phase
            complete = json.loads((run / "complete.json").read_text())
            if complete["recipe_sha256"] != workflow.digest(configs[phase]):
                raise ValueError(f"Unmatched completed stage: {phase}")
            workflow.rank_states(run, args.gpus, steps[phase])
        for phase in ("teacher_joint", "joint"):
            checkpoint = args.output / phase / "checkpoints/last_full.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    if not args.train_only:
        run_logged(old_evaluation_command(args), args.output / "benchmark_old/evaluate.log", environment)
        # Generate once under the frozen strict protocol, then rescore the SAME texts.
        workflow.run_evaluation(args, environment)
        collect_results(args, inputs, run_rescoring(args, environment))


def main(argv=None):
    args = parse_args(argv)
    print("R12: teacher_alignment -> teacher_joint -> student (warmup -> joint)", flush=True)
    print("Validating local data and evaluation protocols...", flush=True)
    inputs = workflow.inspect_inputs(args, check_images=not args.dry_run)
    inputs["old_benchmarks"] = inspect_old_benchmarks(args)
    configs, steps = build_recipes(args, inputs["train_records"])
    code_paths = sorted({Path(__file__), ROOT / "scripts/run_r12.sh",
                         ROOT / "scripts/reasoning_data/train_and_evaluate.py",
                         *ROOT.glob("src/lvr/**/*.py"), *ROOT.glob("scripts/latent_prediction/*.py"),
                         *ROOT.glob("scripts/reasoning_benchmarks/*.py")})
    plan = dict(format="lvr-r12-workflow-v1", stages=STAGES,
                inputs={k: v for k, v in inputs.items() if k != "checked"},
                recipes=configs, optimizer_steps=steps, eval_gpus=args.eval_gpus,
                evaluation=dict(modes=list(workflow.MODES), selection=args.selection,
                                old_scoring="original", answer_policy=ANSWER_POLICY, mira_policy=MIRA_POLICY,
                                original_pixels="processor_default", helper_pixels=262144,
                                old_max_new_tokens=256, new_max_new_tokens=args.max_new_tokens,
                                system_prompt="You are a helpful assistant."),
                code={str(p.relative_to(ROOT)): workflow.file_sha(p) for p in code_paths},
                deepspeed_sha256=workflow.file_sha(args.deepspeed_config))
    print(json.dumps(dict(output=str(args.output), train_records=inputs["train_records"],
                          val_records=inputs["val_records"], optimizer_steps=steps,
                          old_benchmarks=inputs["old_benchmarks"]["counts"],
                          new_benchmarks=inputs["benchmark_counts"], teacher_helper_records=inputs["teacher_records"],
                          reconstruction_weight=0.1, reconstruction_normalize=True), indent=2), flush=True)
    if args.dry_run:
        for phase, config in configs.items():
            print(f"--- {phase} ---\n" + yaml.safe_dump(config, sort_keys=False))
            print(shlex.join(workflow.training_command(args, phase, args.output / "recipes" / f"{phase}.yaml")))
        print(shlex.join(old_evaluation_command(args)))
        for mode in workflow.MODES:
            for shard in range(args.eval_gpus):
                print(shlex.join(workflow.evaluation_command(args, mode, shard)))
            print(shlex.join(rescore_command(args, mode, args.output / "benchmark_latest" / mode)))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = args.output / "workflow_plan.json"
        if binding.exists() and json.loads(binding.read_text()) != plan:
            raise ValueError("Existing R12 data/config/code changed; use a new --output")
        workflow.write_json(binding, plan)
        workflow.write_json(args.output / "input_check.json", dict(status="passed", **inputs))
        for phase, config in configs.items():
            path = args.output / "recipes" / f"{phase}.yaml"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False))
        if args.prepare_only:
            print(f"PREPARED {args.output} (no training or evaluation launched)", flush=True)
            return 0
        import torch

        required = args.eval_gpus if args.eval_only else args.gpus if args.train_only else max(args.gpus, args.eval_gpus)
        if torch.cuda.device_count() < required:
            raise RuntimeError(f"Need {required} visible CUDA devices. Use --prepare-only for CPU validation.")
        environment = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           LVR_PREFLIGHT_DIAGNOSTICS="0", LVR_REASONING_SUITE=str(args.suite), MPLBACKEND="Agg")
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment.pop("LVR_DIAGNOSTICS_DIR", None)
        for key, value in {"OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                           "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1"}.items():
            environment.setdefault(key, value)
        status = dict(status="running", plan_sha256=workflow.digest(plan))
        workflow.write_json(args.output / "workflow_status.json", status)
        try:
            execute(args, inputs, configs, steps, environment)
        except BaseException as error:
            workflow.write_json(args.output / "workflow_status.json", dict(status, status="failed", error=str(error)))
            raise
        workflow.write_json(args.output / "workflow_status.json",
                            dict(status, status="training_complete" if args.train_only else "complete"))
        print(f"R12 COMPLETE: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
