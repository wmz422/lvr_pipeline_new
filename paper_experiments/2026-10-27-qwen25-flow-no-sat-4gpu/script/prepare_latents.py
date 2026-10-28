#!/usr/bin/env python3
"""Prepare reusable FP32 VAE latents with independent CPU or GPU workers."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT), str(WORKSPACE / "doc/flow_reconstruction_parallel_20261015"),
               str(WORKSPACE / "lvr_pipeline_new/src")]
from reconstruction_experiment.latent_cache import (
    LatentStore, bind_plan, build_plan, finalize, read_plan, run_worker, shard_metadata)
from reconstruction_experiment.prepare import write_json


def validate_equivalent_layout(expected, stored):
    """Permit different shard boundaries only when every bound tensor is identical."""
    omit = {"shards", "plan_sha256"}
    if ({k: v for k, v in expected.items() if k not in omit}
            != {k: v for k, v in stored.items() if k not in omit}):
        raise ValueError("VAE/data/encoding settings differ from the stored cache")
    required = {item["key"]: {k: v for k, v in item.items() if k != "offset"}
                for shard in expected["shards"] for item in shard["items"]}
    seen = set()
    for index, shard in enumerate(stored["shards"]):
        if (set(shard) != {"items", "bytes", "file"} or not shard["items"]
                or shard["file"] != f"shards/{index:06d}.f32"):
            raise ValueError("Invalid alternate cache shard layout")
        offset = 0
        for item in shard["items"]:
            key = item["key"]
            if (key in seen or key not in required or item["offset"] != offset
                    or {k: v for k, v in item.items() if k != "offset"} != required[key]):
                raise ValueError("Alternate cache tensor inventory or layout differs")
            seen.add(key)
            offset += item["bytes"]
        if offset != shard["bytes"]:
            raise ValueError("Alternate cache shard byte count differs")
        if shard["bytes"] > expected["encoding"]["shard_bytes"] and len(shard["items"]) != 1:
            raise ValueError("Alternate cache shard exceeds the bound size")
    if seen != set(required):
        raise ValueError("Alternate cache is missing tensors")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-mib", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.workers, args.batch_size, args.shard_mib, args.threads) < 1:
        parser.error("Workers, batch size, shard size and CPU threads must be positive")
    args.flow_data, args.output, args.model = (path.expanduser().resolve() for path in
                                             (args.flow_data, args.output, args.model))
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    if args.worker_rank is not None:
        device = "cpu" if args.device == "cpu" else f"cuda:{args.worker_rank}"
        run_worker(args.output, args.worker_rank, args.workers, device)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan(args.flow_data, args.model, batch_size=args.batch_size,
                          shard_bytes=args.shard_mib * 2**20, device_type=args.device)
        if (args.output / "plan.json").exists():
            stored = read_plan(args.output)
            validate_equivalent_layout(plan, stored)
            plan = stored
        bind_plan(args.output, plan)
        print(json.dumps({"unique_images": plan["unique_images"], "image_occurrences": plan["image_occurrences"],
                          "cache_gib": plan["bytes"] / 2**30, "shards": len(plan["shards"]),
                          "dtype": "float32", "device": args.device, "workers": args.workers}), flush=True)
        if args.plan_only:
            return
        # Verify committed shards before reuse; partially written shards are rebuilt.
        pending = [i for i in range(len(plan["shards"])) if shard_metadata(args.output, plan, i) is None]
        if not pending:
            if (args.output / "status.json").exists():
                LatentStore(args.output, expected_gt=plan["gt_protocol_sha256"], expected_encoder=plan["encoder"])
            else:
                finalize(args.output)
            print("VAE_CACHE_ALREADY_COMPLETE", args.output, flush=True)
            return
        if args.device == "cuda" and torch.cuda.device_count() < args.workers:
            raise RuntimeError(f"Need {args.workers} visible GPUs for encoding; found {torch.cuda.device_count()}")
        remaining = sum(plan["shards"][i]["bytes"] for i in pending)
        if shutil.disk_usage(args.output).free < remaining + 64 * 2**20:
            raise RuntimeError(f"Insufficient cache disk space; need approximately {remaining / 2**30:.2f} GiB")
        started, processes = time.monotonic(), []
        write_json(args.output / "preparation.json", {"status": "encoding", "workers": args.workers,
            "device": args.device, "remaining_shards": len(pending), "plan_sha256": plan["plan_sha256"]})
        try:
            for rank in range(args.workers):
                command = [sys.executable, str(Path(__file__).resolve()), "--flow-data", str(args.flow_data),
                    "--output", str(args.output), "--model", str(args.model), "--device", args.device,
                    "--workers", str(args.workers), "--batch-size", str(args.batch_size),
                    "--shard-mib", str(args.shard_mib), "--threads", str(args.threads), "--worker-rank", str(rank)]
                processes.append(subprocess.Popen(command))
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None, 0) for process in processes):
                    raise RuntimeError("VAE worker failed; completed shards are retained for resume")
                time.sleep(.2)
            if any(process.returncode for process in processes):
                raise RuntimeError("VAE worker failed; inspect the encoding log")
            finalize(args.output)
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            write_json(args.output / "preparation.json", {"status": "interrupted", "resume_supported": True,
                                                         "plan_sha256": plan["plan_sha256"]})
            raise
        write_json(args.output / "preparation.json", {"status": "complete", "workers": args.workers,
            "device": args.device, "seconds_this_invocation": time.monotonic() - started,
            "plan_sha256": plan["plan_sha256"], "new_shards": len(pending)})
        print("VAE_CACHE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
