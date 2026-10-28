#!/usr/bin/env python3
"""Evaluate teacher/predicted/zero/shuffled on the recorded 3,662-question protocol."""

import argparse
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
MODES = ("predicted", "teacher", "zero", "shuffled")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latent_prediction/benchmark")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--vlmeval-root", type=Path, default=ROOT / "third_party/VLMEvalKit")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--num-loops", type=int, help="Latent prediction rounds; default reads the checkpoint")
    parser.add_argument("--bench-root", type=Path,
                        default=os.environ.get("LVR_BENCH_ROOT", "/workspace/wmz/lvr_pipeline_new/benchmark/bench"))
    parser.add_argument("--processor-default-resolution", action="store_true",
                        help="Use base processor image limits on every benchmark, bypassing the training pixel cap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpus < 1:
        parser.error("gpus must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must not repeat")
    if args.num_loops is not None and args.num_loops < 1:
        parser.error("num-loops must be positive")
    output, protocol = args.output.resolve(), args.protocol.resolve()
    datasets = list(json.loads((protocol / "datasets.json").read_text()))
    adapter = ["--checkpoint", str(args.checkpoint.resolve()), "--prediction-checkpoint",
               "--source-checkpoint", str(args.source_checkpoint.resolve()), "--base-model", str(args.base_model.resolve()),
               "--bottleneck-config", str(ROOT / "configs/latent_prediction/warmup.yaml"),
               "--crop-annotations", str(protocol / "crop_annotations.json"), "--max-sequence-tokens", "65536"]
    if args.processor_default_resolution:
        adapter.append("--processor-default-resolution")
    if args.num_loops is not None:
        adapter.extend(["--num-loops", str(args.num_loops)])

    def commands(mode):
        smoke = [sys.executable, str(SCRIPTS / "smoke_lvr_vlmeval.py"), *adapter,
                 "--latent-mode", mode, "--data", *datasets, "--data-config", str(protocol / "data_config.json"),
                 "--output", str(output / f"smoke_{mode}.jsonl")]
        evaluate = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
                    str(SCRIPTS / "run_lvr_vlmeval.py"), *adapter, "--vlmeval-root", str(args.vlmeval_root.resolve()),
                    "--latent-mode", mode, "--model", f"Bottleneck-{mode}", "--data", *datasets,
                    "--data-config-file", str(protocol / "data_config.json"), "--max-new-tokens", "256",
                    "--judge", "exact_matching", "--api-nproc", "4", "--work-dir", str(output / mode),
                    "--reuse", "--reuse-aux", "infer"]
        return smoke, evaluate

    if args.dry_run:
        for mode in args.modes:
            for command in commands(mode):
                print(shlex.join(command))
        return 0
    for path in (args.checkpoint, args.source_checkpoint, args.base_model, args.vlmeval_root / "run.py"):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = {name: str(path.resolve()) for name, path in
              (("checkpoint", args.checkpoint), ("source_checkpoint", args.source_checkpoint),
               ("base_model", args.base_model), ("protocol", protocol), ("bench_root", args.bench_root))}
    inputs["checkpoint_size"] = args.checkpoint.stat().st_size
    inputs["checkpoint_mtime_ns"] = args.checkpoint.stat().st_mtime_ns
    if args.modes != list(MODES):
        inputs["modes"] = args.modes
    if args.num_loops is not None:
        inputs["num_loops"] = args.num_loops
    if args.processor_default_resolution:
        inputs["image_resolution"] = {
            "policy": "processor_default",
            "preprocessor_sha256": hashlib.sha256((args.base_model / "preprocessor_config.json").read_bytes()).hexdigest(),
            "bottleneck_config_sha256": hashlib.sha256((ROOT / "configs/latent_prediction/warmup.yaml").read_bytes()).hexdigest(),
            "protocol_sha256": {name: hashlib.sha256((protocol / name).read_bytes()).hexdigest()
                                for name in ("datasets.json", "data_config.json", "crop_annotations.json")},
        }
    manifest = output / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != inputs:
        parser.error("Evaluation inputs changed; use a new --output")
    manifest.write_text(json.dumps(inputs, indent=2) + "\n")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PRED_FORMAT="tsv")
    env.setdefault("LMUData", str(ROOT / "benchmark/LMUData-local"))
    env.update(LVR_BENCH_ROOT=str(args.bench_root.resolve()), LVR_BENCH_PROTOCOL=str(protocol),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("OMP_NUM_THREADS", "8")
    code = 1
    try:
        (output / "phase").write_text("checking_local_data\n")
        subprocess.run([sys.executable, "-m", "lvr.eval.local_protocol"], cwd=ROOT, env=env, check=True)
        # Predict first: all original images are materialized before shuffled donors are read.
        for mode in args.modes:
            summary = output / f"summary_{mode}.json"
            if not summary.exists():
                (output / "phase").write_text(mode + "\n")
                for command in commands(mode):
                    subprocess.run(command, cwd=ROOT, env=env, check=True)
                candidates = list((output / mode / f"Bottleneck-{mode}").glob("T*/status.json"))
                if not candidates:
                    raise FileNotFoundError(f"No evaluation status under {output / mode}")
                run_dir = max(candidates, key=lambda path: path.stat().st_mtime_ns).parent
            else:
                run_dir = output / mode / f"Bottleneck-{mode}" / json.loads(summary.read_text())["eval_id"]
            subprocess.run([sys.executable, str(SCRIPTS / "summarize_lvr_vlmeval.py"), str(run_dir),
                            "--datasets", str(protocol / "datasets.json"), "--output", str(summary)],
                           cwd=ROOT, env=env, check=True)
        (output / "phase").write_text("complete\n")
        code = 0
    finally:
        (output / "exit_code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
