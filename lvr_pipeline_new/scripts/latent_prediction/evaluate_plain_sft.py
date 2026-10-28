#!/usr/bin/env python3
"""Evaluate a zero-slot QA checkpoint on the same 3,662-question protocol."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def export_table(summary_path):
    summary = json.loads(summary_path.read_text())
    rows = []
    for name, value in summary["results"].items():
        rows.append(dict(benchmark=name, samples=value["samples"], correct=value["correct_questions"],
                         accuracy_percent=100 * value["correct_questions"] / value["samples"],
                         empty_predictions=value["empty_predictions"],
                         truncated_predictions=value["truncated_predictions"]))
    total = dict(benchmark="ALL_WEIGHTED", samples=sum(r["samples"] for r in rows),
                 correct=sum(r["correct"] for r in rows),
                 empty_predictions=sum(r["empty_predictions"] for r in rows),
                 truncated_predictions=sum(r["truncated_predictions"] for r in rows)
                 if all(r["truncated_predictions"] is not None for r in rows) else None)
    total["accuracy_percent"] = total["correct"] / total["samples"] * 100
    rows.append(total)
    with (summary_path.parent / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Plain SFT benchmark results", "", "| Benchmark | Correct / N | Accuracy % |", "| --- | ---: | ---: |"]
    lines.extend(f"| {r['benchmark']} | {r['correct']} / {r['samples']} | {r['accuracy_percent']:.2f} |" for r in rows)
    (summary_path.parent / "results.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Resolved plain SFT training recipe")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/plain_sft/benchmark")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--bench-root", type=Path,
                        default=os.environ.get("LVR_BENCH_ROOT", "/workspace/wmz/lvr_pipeline_new/benchmark/bench"))
    parser.add_argument("--vlmeval-root", type=Path, default=ROOT / "third_party/VLMEvalKit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--processor-default-resolution", action="store_true",
                        help="Use base processor image limits on every benchmark, bypassing the training pixel cap")
    args = parser.parse_args(argv)
    if args.gpus < 1:
        parser.error("gpus must be positive")
    output, protocol = args.output.resolve(), args.protocol.resolve()
    datasets = list(json.loads((protocol / "datasets.json").read_text()))
    work = output / "plain_sft"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
               str(SCRIPTS / "run_lvr_vlmeval.py"), "--checkpoint", str(args.checkpoint.resolve()),
               "--bottleneck-config", str(args.config.resolve()), "--base-model", str(args.base_model.resolve()),
               "--latent-mode", "none", "--model", "Bottleneck-none", "--data", *datasets,
               "--crop-annotations", str(protocol / "crop_annotations.json"),
               "--data-config-file", str(protocol / "data_config.json"),
               "--max-sequence-tokens", "65536", "--max-new-tokens", "256",
               "--judge", "exact_matching", "--api-nproc", "4", "--work-dir", str(work),
               "--vlmeval-root", str(args.vlmeval_root.resolve()), "--reuse", "--reuse-aux", "infer"]
    if args.processor_default_resolution:
        command.append("--processor-default-resolution")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    for path in (args.checkpoint, args.config, args.base_model, args.vlmeval_root / "run.py"):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if config["model"]["num_latents"] != 0 or config["data"].get("qa_crops"):
        parser.error("Expected a zero-slot, original-image plain SFT recipe")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = {name: str(path.resolve()) for name, path in (
        ("checkpoint", args.checkpoint), ("config", args.config), ("base_model", args.base_model),
        ("protocol", protocol), ("bench_root", args.bench_root))}
    inputs.update(checkpoint_size=args.checkpoint.stat().st_size, checkpoint_mtime_ns=args.checkpoint.stat().st_mtime_ns,
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  protocol_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (protocol / n for n in ("datasets.json", "data_config.json", "crop_annotations.json"))})
    if args.processor_default_resolution:
        inputs["image_resolution"] = {
            "policy": "processor_default",
            "preprocessor_sha256": hashlib.sha256((args.base_model / "preprocessor_config.json").read_bytes()).hexdigest(),
        }
    manifest = output / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != inputs:
        parser.error("Evaluation inputs changed; use a new --output")
    manifest.write_text(json.dumps(inputs, indent=2) + "\n")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PRED_FORMAT="tsv",
               LVR_BENCH_ROOT=str(args.bench_root.resolve()), LVR_BENCH_PROTOCOL=str(protocol),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("LMUData", str(ROOT / "benchmark/LMUData-local"))
    env.setdefault("OMP_NUM_THREADS", "8")
    summary = output / "summary_plain_sft.json"
    code = 1
    try:
        (output / "phase").write_text("checking_local_data\n")
        subprocess.run([sys.executable, "-m", "lvr.eval.local_protocol"], cwd=ROOT, env=env, check=True)
        if not summary.exists():
            (output / "phase").write_text("plain_sft\n")
            print(f"[plain-sft eval] five benchmarks / 3662 questions; log={output / 'evaluate.log'}", flush=True)
            with (output / "evaluate.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            candidates = list((work / "Bottleneck-none").glob("T*/status.json"))
            if not candidates:
                raise FileNotFoundError(f"No evaluation status under {work}")
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent
        else:
            run_dir = work / "Bottleneck-none" / json.loads(summary.read_text())["eval_id"]
        (output / "phase").write_text("summarizing\n")
        subprocess.run([sys.executable, str(SCRIPTS / "summarize_lvr_vlmeval.py"), str(run_dir),
                        "--datasets", str(protocol / "datasets.json"), "--output", str(summary)],
                       cwd=ROOT, env=env, check=True)
        export_table(summary)
        (output / "phase").write_text("complete\n")
        code = 0
        print(f"[plain-sft eval] complete: {output / 'results.md'}", flush=True)
    finally:
        (output / "exit_code").write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
