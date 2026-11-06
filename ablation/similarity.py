#!/usr/bin/env python3
"""Cache first-eight native latent positions for cross-case cosine diagnostics."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from common import HERE, PROJECT, bind_json, digest, load_data, select_rows, sha256, write_json
from download_models import verify_asset
from similarity_cache import read_trace, save_trace, trace_path

METHOD_ASSETS = {"ours": ["ours", "qwen3-base"], "lvr": ["lvr", "qwen25-processor"],
                 "monet": ["monet", "monet-teacher", "qwen25-processor"]}


def config_for(args, rows):
    lock = json.loads((HERE / "models.lock.json").read_text())
    required = list(METHOD_ASSETS[args.method])
    if args.method == "monet":
        required.remove("monet" if args.kind == "teacher" else "monet-teacher")
    assets = {}
    for name in required:
        spec = lock["models"][name]
        stats = verify_asset(args.assets / "models" / name, spec, hash_files=args.verify_assets)
        assets[name] = {"spec": spec, "stats": stats}
    sources = {}
    if args.method != "ours":
        spec = lock["sources"][args.method]
        stats = verify_asset(args.assets / "sources" / args.method, spec, hash_files=True)
        sources[args.method] = {"spec": spec, "stats": stats}
    code = {p.name: sha256(p) for p in HERE.glob("*.py") if not p.name.startswith("test_")}
    own_code = {str(p.relative_to(PROJECT)): sha256(p) for p in (PROJECT / "src/lvr").rglob("*.py")} if args.method == "ours" else {}
    packages = {}
    for name in ("torch", "transformers", "safetensors", "qwen-vl-utils", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema": "cross-case-latent-run-v1", "method": args.method, "kind": args.kind,
            "suite": str(args.benchmark), "suite_manifest_sha256": sha256(args.benchmark / "manifest.json"),
            "assets_root": str(args.assets), "models": assets, "sources": sources,
            "scope": "smoke" if args.smoke_per_benchmark else "full", "selected_ids": [r["id"] for r in rows],
            "retained_positions": 8, "code": code, "own_code": own_code, "packages": packages,
            "seed": 42 if args.method != "monet" else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_ASSETS, required=True)
    parser.add_argument("--kind", choices=["teacher", "predicted"], required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=HERE.parent / ".local/ablation-assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--smoke-per-benchmark", type=int, default=0)
    parser.add_argument("--verify-assets", action="store_true", help="Rehash weights instead of checking the download receipt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.smoke_per_benchmark < 0:
        parser.error("Invalid shard or smoke selection")
    for key in ("assets", "benchmark", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    # Monet must be patched before interpreter startup, including its engine children.
    if args.method == "monet" and args.kind == "predicted" and not args.dry_run and not os.environ.get("MONET_ABLATION_BOOTSTRAPPED"):
        from monet_observer import prepare_runtime
        source_spec = json.loads((HERE / "models.lock.json").read_text())["sources"]["monet"]
        verify_asset(args.assets / "sources/monet", source_spec, hash_files=True)
        parent = args.output / "monet/predicted/runtime"
        parent.mkdir(parents=True, exist_ok=True)
        runtime = prepare_runtime(args.assets / "sources/monet", Path(tempfile.mkdtemp(prefix="worker_", dir=parent)))
        trace_dir = runtime / "traces"
        trace_dir.mkdir()
        env = dict(os.environ, MONET_ABLATION_BOOTSTRAPPED="1", LATENT_SIZE="10",
                   MONET_ABLATION_TRACE_DIR=str(trace_dir),
                   PYTHONPATH=str(runtime) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   VLLM_WORKER_MULTIPROC_METHOD="spawn")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    rows, protocol, _, _ = load_data(args.benchmark)
    selected = select_rows(rows, args.smoke_per_benchmark)
    config = config_for(args, selected)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "method": args.method, "kind": args.kind,
                          "scope": config["scope"], "records": len(selected), "binding": digest(config)}, indent=2))
        return
    branch_dir = args.output / args.method / args.kind
    branch_dir.mkdir(parents=True, exist_ok=True)
    # Brief config lock permits simultaneous shards, with one consistent provenance binding.
    with (branch_dir / "config.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        bind_json(branch_dir / "run_config.json", config)
    binding = digest(config)
    lock = (branch_dir / f"worker_{args.shard_index}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = []
    for row in selected[args.shard_index::args.num_shards]:
        path = trace_path(args.output, args.method, args.kind, row["id"])
        if path.exists():
            read_trace(path, row["id"], binding)
        else:
            pending.append(row)
    if not pending:
        print("All assigned traces already verified")
        return
    import torch
    from similarity_models import load_extractor
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("Trained-model extraction requires a CUDA device; use --dry-run for CPU input checks")
    if args.method == "monet" and args.kind == "predicted" and args.device != "cuda:0":
        raise ValueError("For native vLLM select one GPU with CUDA_VISIBLE_DEVICES and use --device cuda:0")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.cuda.set_device(args.device)
    extractor = load_extractor(args.method, args.kind, assets=args.assets, suite=args.benchmark,
                               protocol=protocol, device=args.device, runtime=branch_dir / "runtime")
    write_json(branch_dir / f"extractor_{args.shard_index}.json", extractor.info)
    for index, row in enumerate(pending, 1):
        started = time.monotonic()
        value, record = extractor.extract(row)
        save_trace(trace_path(args.output, args.method, args.kind, row["id"]), value, row["id"], binding,
                   dict(record, elapsed_seconds=time.monotonic() - started))
        print(f"{args.method} {args.kind} shard={args.shard_index} {index}/{len(pending)} {row['id']} positions={min(len(value), 8)}", flush=True)
    write_json(branch_dir / f"complete_{args.shard_index}.json",
               {"run_sha256": binding, "num_shards": args.num_shards, "shard_index": args.shard_index})


if __name__ == "__main__":
    main()
