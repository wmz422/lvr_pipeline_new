#!/usr/bin/env python3
"""Evaluate teacher/predicted/zero/shuffled on the recorded 3,662-question protocol."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latent_prediction/benchmark")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--vlmeval-root", type=Path, default=ROOT / "third_party/VLMEvalKit")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("gpus must be positive")
    for path in (args.checkpoint, args.source_checkpoint, args.base_model, args.vlmeval_root / "run.py"):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    output, protocol = args.output.resolve(), args.protocol.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = {name: str(path.resolve()) for name, path in
              (("checkpoint", args.checkpoint), ("source_checkpoint", args.source_checkpoint),
               ("base_model", args.base_model), ("protocol", protocol))}
    inputs["checkpoint_size"] = args.checkpoint.stat().st_size
    inputs["checkpoint_mtime_ns"] = args.checkpoint.stat().st_mtime_ns
    manifest = output / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != inputs:
        parser.error("Evaluation inputs changed; use a new --output")
    manifest.write_text(json.dumps(inputs, indent=2) + "\n")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PRED_FORMAT="tsv")
    env.setdefault("LMUData", str(ROOT / "benchmark/LMUData"))
    env.setdefault("OMP_NUM_THREADS", "8")
    datasets = list(json.loads((protocol / "datasets.json").read_text()))
    adapter = ["--checkpoint", str(args.checkpoint.resolve()), "--prediction-checkpoint",
               "--source-checkpoint", str(args.source_checkpoint.resolve()), "--base-model", str(args.base_model.resolve()),
               "--bottleneck-config", str(ROOT / "configs/latent_prediction/warmup.yaml"),
               "--crop-annotations", str(protocol / "crop_annotations.json"), "--max-sequence-tokens", "65536"]
    code = 1
    try:
        # Predict first: all original images are materialized before shuffled donors are read.
        for mode in ("predicted", "teacher", "zero", "shuffled"):
            summary = output / f"summary_{mode}.json"
            if not summary.exists():
                (output / "phase").write_text(mode + "\n")
                subprocess.run([sys.executable, str(SCRIPTS / "smoke_lvr_vlmeval.py"), *adapter,
                                "--latent-mode", mode, "--data", *datasets, "--data-config", str(protocol / "data_config.json"),
                                "--output", str(output / f"smoke_{mode}.jsonl")], cwd=ROOT, env=env, check=True)
                subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
                                str(SCRIPTS / "run_lvr_vlmeval.py"), *adapter, "--vlmeval-root", str(args.vlmeval_root.resolve()),
                                "--latent-mode", mode, "--model", f"Bottleneck-{mode}", "--data", *datasets,
                                "--data-config-file", str(protocol / "data_config.json"), "--max-new-tokens", "256",
                                "--judge", "exact_matching", "--api-nproc", "4", "--work-dir", str(output / mode)],
                               cwd=ROOT, env=env, check=True)
            subprocess.run([sys.executable, str(SCRIPTS / "summarize_lvr_vlmeval.py"), str(output / mode),
                            "--datasets", str(protocol / "datasets.json"), "--output", str(summary)],
                           cwd=ROOT, env=env, check=True)
        (output / "phase").write_text("complete\n")
        code = 0
    finally:
        (output / "exit_code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
