#!/usr/bin/env python3
"""Small, fail-fast VStar generation check using the formal benchmark adapter."""

# Keep imports CPU/stdlib-only until a worker has selected its one physical GPU.
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MODES = ("predicted", "teacher", "zero", "shuffled")
DATASET = "VStarBench"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def select_indices(config, annotations, count):
    """Deterministic protocol subset, including its highest crop-count example."""
    entry = config[DATASET]
    if entry["class"] != "DatasetSubset" or entry["dataset"] != DATASET:
        raise ValueError("VStar preflight requires the recorded DatasetSubset protocol")
    indices = [str(index) for index in entry["indices"]]
    if count < 1 or count > len(indices) or len(set(indices)) != len(indices):
        raise ValueError(f"Need 1..{len(indices)} distinct VStar questions, requested {count}")
    rows = {str(row["index"]): row for row in annotations["samples"].values()
            if row["benchmark"] == DATASET}
    if not set(indices) <= rows.keys():
        raise ValueError("Missing VStar crop annotations")
    largest = max(indices, key=lambda index: sum(len(image["boxes"]) for image in rows[index]["images"]))
    selected = indices[:count]
    if largest not in selected:
        selected[-1] = largest
    return selected


def bind_worker_gpu(environment):
    rank = int(environment["RANK"])
    local_rank = int(environment["LOCAL_RANK"])
    world_size = int(environment["WORLD_SIZE"])
    local_world_size = int(environment["LOCAL_WORLD_SIZE"])
    if not (0 <= rank < world_size == local_world_size and 0 <= local_rank < local_world_size):
        raise ValueError("Preflight expects single-node torchrun ranks")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    devices = [item.strip() for item in visible.split(",")] if visible is not None else None
    if devices is not None and (len(devices) < local_world_size or any(not item or item == "-1" for item in devices)):
        raise ValueError("Not enough visible GPUs for benchmark preflight")
    environment["CUDA_VISIBLE_DEVICES"] = devices[local_rank] if devices else str(local_rank)
    return rank, world_size


def check_records(directory, run_id, mode, indices, world_size, limit, allow_tight):
    records = []
    for rank in range(world_size):
        record = json.loads((directory / f"runtime_rank{rank}.json").read_text())
        expected = indices[rank::world_size]
        if (record.get("run_id") != run_id or record.get("status") != "complete"
                or record.get("rank") != rank or record.get("world_size") != world_size
                or record.get("mode") != mode or record.get("completed_indices") != expected):
            raise RuntimeError(f"Incomplete or stale VStar preflight record: {mode} rank {rank}")
        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            value = float(record[key])
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"Invalid {key}: {mode} rank {rank}")
        if record["peak_reserved_gib"] > limit and not allow_tight:
            raise RuntimeError(f"Benchmark reserved memory exceeds {limit:g} GiB: {mode} rank {rank}; "
                               "reducing training micro-batch does not change inference batch 1")
        records.append(record)
    return records


def run_worker(args):
    rank, world_size = bind_worker_gpu(os.environ)
    directory = args.output / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.output / "inputs.json").read_text())
    indices = manifest["indices"][rank::world_size]
    if len(indices) != args.samples_per_gpu or manifest["run_id"] != args.run_id:
        raise ValueError("Worker assignment does not match the preflight manifest")
    state = dict(run_id=args.run_id, rank=rank, world_size=world_size, mode=args.mode,
                 dataset=DATASET, inference_batch_size=1, indices=indices, completed_indices=[],
                 pid=os.getpid(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                 status="starting", started_at_unix=time.time())

    def update(status, **fields):
        state.update(status=status, updated_at_unix=time.time(),
                     peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                     **fields)
        write_json(directory / f"runtime_rank{rank}.json", state)

    update("loading_dependencies")
    try:
        import torch
        import torch.distributed as dist
        from lvr.eval.latent_prediction import PredictedLatentModel, shuffled_sources
        from lvr.eval.vlmeval import DatasetSubset

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        update("loading_dataset")
        annotations = json.loads((args.protocol / "crop_annotations.json").read_text())["samples"]
        by_index = {str(row["index"]): identity for identity, row in annotations.items()
                    if row["benchmark"] == DATASET}
        donors = shuffled_sources(annotations) if args.mode == "shuffled" else {}
        # Serialize local dataset materialization (never download). Keep the
        # full recorded dataset so shuffled donors retain their formal identities.
        with (args.output / "dataset.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            config = json.loads((args.protocol / "data_config.json").read_text())[DATASET]
            dataset = DatasetSubset(dataset=DATASET, indices=config["indices"])
            positions = {str(index): position for position, index in enumerate(dataset.data["index"])}
            messages = []
            for index in indices:
                messages.append(dataset.build_prompt(dataset.data.iloc[positions[index]]))
                if donors:
                    donor = annotations[donors[by_index[index]]]
                    dataset.build_prompt(dataset.data.iloc[positions[str(donor["index"])]])

        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        update("loading_model")
        dist.barrier(device_ids=[0])
        model = PredictedLatentModel(
            checkpoint=str(args.checkpoint), config=str(ROOT / "configs/latent_prediction/warmup.yaml"),
            latent_mode=args.mode, max_new_tokens=256, max_sequence_tokens=65536,
            crop_annotations=args.protocol / "crop_annotations.json",
            base_model_path=str(args.base_model), source_checkpoint=str(args.source_checkpoint),
            num_loops=getattr(args, "num_loops", None),
            processor_default_resolution=getattr(args, "processor_default_resolution", False),
        )
        torch.cuda.synchronize(0)
        update("model_loaded")
        # Require all replicas to coexist, rather than letting faster workers exit
        # before slower workers load their models (important for host-memory checks).
        dist.barrier(device_ids=[0])
        with (directory / f"predictions_rank{rank}.jsonl").open("w", buffering=1) as stream:
            for index, message in zip(indices, messages):
                update("generating", current_index=index)
                result = model.generate(message, dataset=DATASET)
                if (not isinstance(result, dict) or not isinstance(result.get("prediction"), str)
                        or not result["prediction"].strip()):
                    raise RuntimeError(f"Empty/invalid generation: {DATASET}/{index}")
                torch.cuda.synchronize(0)
                stream.write(json.dumps(dict(index=index, **result), ensure_ascii=False) + "\n")
                state["completed_indices"].append(index)
                update("generating", peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3)
                print(f"[vstar-preflight] mode={args.mode} rank={rank} "
                      f"completed={len(state['completed_indices'])}/{len(indices)} "
                      f"reserved={state['peak_reserved_gib']:.2f}GiB", flush=True)
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        update("complete")
    except BaseException as error:
        update("failed", exception_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/latent_prediction")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--num-loops", type=int)
    parser.add_argument("--processor-default-resolution", action="store_true")
    parser.add_argument("--samples-per-gpu", type=int, default=2)
    parser.add_argument("--reserved-limit-gib", type=float, default=75)
    parser.add_argument("--allow-tight-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_loops is not None and args.num_loops < 1:
        parser.error("num-loops must be positive")
    if args.gpus < 1 or args.samples_per_gpu < 1 or not math.isfinite(args.reserved_limit_gib) or args.reserved_limit_gib <= 0:
        parser.error("GPU count, samples per GPU and memory limit must be positive")
    for name in ("checkpoint", "source_checkpoint", "base_model", "output", "protocol"):
        setattr(args, name, getattr(args, name).resolve())
    os.environ["LVR_BENCH_PROTOCOL"] = str(args.protocol)
    if args.worker:
        run_worker(args)
        return
    indices = select_indices(json.loads((args.protocol / "data_config.json").read_text()),
                             json.loads((args.protocol / "crop_annotations.json").read_text()),
                             args.gpus * args.samples_per_gpu)
    run_id = uuid.uuid4().hex
    commands = {}
    for mode in MODES:
        commands[mode] = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                          f"--nproc_per_node={args.gpus}", str(Path(__file__).resolve()), "--worker",
                          "--mode", mode, "--run-id", run_id, "--checkpoint", str(args.checkpoint),
                          "--source-checkpoint", str(args.source_checkpoint), "--base-model", str(args.base_model),
                          "--protocol", str(args.protocol), "--output", str(args.output),
                          "--samples-per-gpu", str(args.samples_per_gpu)]
        if args.num_loops is not None:
            commands[mode].extend(["--num-loops", str(args.num_loops)])
        if args.processor_default_resolution:
            commands[mode].append("--processor-default-resolution")
    if args.dry_run:
        print(f"[vstar-preflight] {len(indices)} VStar questions, inference batch 1, indices={indices}")
        for command in commands.values():
            print(shlex.join(command))
        return
    for path in (args.checkpoint, args.source_checkpoint, args.base_model / "config.json"):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(run_id=run_id, dataset=DATASET, indices=indices, world_size=args.gpus,
                    samples_per_gpu=args.samples_per_gpu, inference_batch_size=1,
                    checkpoint=str(args.checkpoint), source_checkpoint=str(args.source_checkpoint),
                    base_model=str(args.base_model), protocol=str(args.protocol), modes=list(MODES))
    manifest.update(num_loops=args.num_loops, processor_default_resolution=args.processor_default_resolution)
    write_json(args.output / "inputs.json", manifest)
    summary = dict(manifest, status="running", modes_completed=[], started_at_unix=time.time())
    write_json(args.output / "summary.json", summary)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    environment.pop("LVR_DIAGNOSTICS_DIR", None)
    environment.setdefault("LMUData", str(ROOT / "benchmark/LMUData-local"))
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    code = 1
    try:
        for mode, command in commands.items():
            directory = args.output / mode
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "evaluate.log"
            print(f"[vstar-preflight] {mode}: {args.gpus} GPUs x {args.samples_per_gpu} questions; "
                  f"log={log_path}", flush=True)
            with log_path.open("a") as stream:
                stream.write(f"\nrun_id={run_id}\n{shlex.join(command)}\n")
                stream.flush()
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            (directory / "exit_code").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError(f"VStar {mode} failed ({result.returncode}); inspect {log_path}")
            records = check_records(directory, run_id, mode, indices, args.gpus,
                                    args.reserved_limit_gib, args.allow_tight_memory)
            summary[mode] = dict(questions=sum(len(row["completed_indices"]) for row in records),
                                 peak_allocated_gib=max(row["peak_allocated_gib"] for row in records),
                                 peak_reserved_gib=max(row["peak_reserved_gib"] for row in records),
                                 max_process_rss_gib=max(row["peak_process_rss_gib"] for row in records))
            summary["modes_completed"].append(mode)
            write_json(args.output / "summary.json", summary)
        summary.update(status="complete", finished_at_unix=time.time())
        code = 0
        print(f"[vstar-preflight] passed: {args.output / 'summary.json'}", flush=True)
    except BaseException as error:
        summary.update(status="failed", exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "summary.json", summary)
        (args.output / "exit_code").write_text(str(code) + "\n")
        lock.close()


if __name__ == "__main__":
    main()
