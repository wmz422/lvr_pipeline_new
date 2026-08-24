#!/usr/bin/env python3
"""Run reconstruction comparison shards concurrently, then merge them by dataset index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from compare_reconstruction import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_PROMPT,
    DEFAULT_QWEN_PATH,
    DEFAULT_TEST_JSON,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument(
        "--qwen-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from exc
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("--gpus must be unique non-negative integers")
    return gpus


def _selected_indices(test_json: Path, start_index: int, limit: int) -> list[int]:
    with test_json.open("r", encoding="utf-8") as handle:
        total = len(json.load(handle))
    if start_index < 0 or start_index >= total:
        raise ValueError(f"--start-index {start_index} is outside dataset of length {total}.")
    if limit == 0 or limit < -1:
        raise ValueError("--limit must be positive or -1 for all remaining records.")
    stop = total if limit == -1 else min(total, start_index + limit)
    return list(range(start_index, stop))


def _shard_path(output: Path, gpu: int) -> Path:
    return output.with_name(f"{output.stem}.part-gpu{gpu}{output.suffix}")


def _merge(shards: list[Path], output: Path, expected_indices: list[int]) -> None:
    rows: dict[int, dict] = {}
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if index in rows:
                    raise ValueError(f"Duplicate dataset index {index} in {shard}:{line_number}")
                rows[index] = row
    missing = sorted(set(expected_indices).difference(rows))
    unexpected = sorted(set(rows).difference(expected_indices))
    if missing or unexpected:
        raise ValueError(f"Incomplete merge; missing={missing}, unexpected={unexpected}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in expected_indices:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    temporary.replace(output)


def main() -> None:
    args = build_parser().parse_args()
    gpus = _parse_gpus(args.gpus)
    indices = _selected_indices(args.test_json, args.start_index, args.limit)
    base_size, extra = divmod(len(indices), len(gpus))
    assignments: list[tuple[int, list[int]]] = []
    cursor = 0
    for position, gpu in enumerate(gpus):
        shard_size = base_size + (1 if position < extra else 0)
        shard_indices = indices[cursor : cursor + shard_size]
        cursor += shard_size
        if shard_indices:
            assignments.append((gpu, shard_indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = [_shard_path(args.output, gpu) for gpu, _ in assignments]
    if args.overwrite:
        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)

    script = Path(__file__).with_name("compare_reconstruction.py")
    processes: list[subprocess.Popen[bytes]] = []
    for gpu, shard_indices in assignments:
        shard_path = _shard_path(args.output, gpu)
        command = [
            sys.executable,
            str(script),
            "--checkpoint", str(args.checkpoint),
            "--test-json", str(args.test_json),
            "--image-root", str(args.image_root),
            "--qwen-model-path", str(args.qwen_model_path),
            "--output", str(shard_path),
            "--start-index", str(shard_indices[0]),
            "--limit", str(len(shard_indices)),
            "--image-size", str(args.image_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--prompt", args.prompt,
            "--system", args.system,
            # The LAM positional encoder uses .cuda() internally.  Restricting each
            # worker to one physical GPU makes that call resolve to its logical cuda:0.
            "--device", "cuda:0",
            "--qwen-dtype", args.qwen_dtype,
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(
            f"GPU {gpu}: {len(shard_indices)} records "
            f"({shard_indices[0]}..{shard_indices[-1]}) -> {shard_path}",
            flush=True,
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes.append(subprocess.Popen(command, env=environment))

    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError(f"One or more GPU shards failed: return codes {failures}")
    _merge(shard_paths, args.output, indices)
    print(f"Finished {len(indices)} records. Merged results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
