#!/usr/bin/env python3
"""Wait for an align run to finish, then evaluate its best validation checkpoint.

The watcher is intentionally read-only with respect to the training run: until
the configured final global step is present in ``checkpoints/last.ckpt``, it
only reads that small checkpoint once per polling interval.  It never sends
signals, acquires CUDA, imports the model, or starts DataLoader workers while
training is active.

Once complete, it waits for a short checkpoint-settling period, selects the
lowest ``val/ce_loss_epoch`` checkpoint from Lightning ModelCheckpoint's saved
``best_k_models`` state, runs ``latent_ablation.py`` as a child process, records
its exit status, and exits.  Thus both watcher and evaluation process close
automatically when evaluation completes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "runs/align/align_full_lr1e4"
DEFAULT_ABLATION = Path(__file__).resolve().parent / "latent_ablation.py"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ablation-script", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--settle-seconds", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "latent_ablation.json")
    parser.add_argument("--status", type=Path, default=DEFAULT_RESULTS_DIR / "watch_status.json")
    parser.add_argument("--log", type=Path, default=DEFAULT_RESULTS_DIR / "latent_ablation.log")
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # The run-root config is ``{fit: {...}}`` while the logger copy is already
    # the inner mapping.  The watcher accepts either representation.
    if isinstance(config, dict) and isinstance(config.get("fit"), dict):
        config = config["fit"]
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return config


def expected_global_steps(config: dict[str, Any], train_json: Path) -> int:
    """Reproduce Lightning's DDP + gradient-accumulation step count."""
    trainer = config["trainer"]
    max_steps = trainer.get("max_steps", -1)
    if max_steps is not None and int(max_steps) > 0:
        return int(max_steps)

    data = config["data"]
    with train_json.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)
    num_examples = len(examples)
    devices = int(trainer.get("devices", 1))
    num_nodes = int(trainer.get("num_nodes", 1))
    world_size = devices * num_nodes
    batch_size = int(data["batch_size"])
    accumulate = int(trainer.get("accumulate_grad_batches", 1))
    max_epochs = int(trainer["max_epochs"])
    batches_per_rank = math.ceil(math.ceil(num_examples / world_size) / batch_size)
    return math.ceil(batches_per_rank / accumulate) * max_epochs


def load_last_checkpoint(path: Path) -> dict[str, Any]:
    # Checkpoints are only ~0.5 MB because this run saves trainable-only state.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint content: {path}")
    return checkpoint


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def select_best_checkpoint(last_checkpoint: dict[str, Any], checkpoint_dir: Path) -> tuple[Path, float]:
    """Select the minimum monitored score saved by the configured callback."""
    callbacks = last_checkpoint.get("callbacks", {})
    candidates: list[tuple[float, Path]] = []
    for state in callbacks.values():
        if not isinstance(state, dict) or state.get("monitor") != "val/ce_loss_epoch":
            continue
        for raw_path, raw_score in (state.get("best_k_models") or {}).items():
            path = Path(raw_path)
            if path.exists():
                candidates.append((_as_float(raw_score), path))
        best_path = state.get("best_model_path")
        best_score = state.get("best_model_score")
        if best_path and best_score is not None:
            path = Path(best_path)
            if path.exists():
                candidates.append((_as_float(best_score), path))
    if not candidates:
        raise RuntimeError(
            f"No usable val/ce_loss_epoch best checkpoint in {checkpoint_dir}/last.ckpt."
        )
    return min(candidates, key=lambda item: item[0])


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yaml"
    checkpoint_dir = run_dir / "checkpoints"
    last_path = checkpoint_dir / "last.ckpt"
    if not config_path.is_file() or not args.ablation_script.is_file():
        raise FileNotFoundError("Run config or ablation script does not exist.")
    if args.poll_seconds <= 0 or args.settle_seconds < 0:
        raise ValueError("--poll-seconds must be > 0 and --settle-seconds must be >= 0.")

    config = load_yaml(config_path)
    train_json = Path(config["data"]["train_path"])
    target_step = expected_global_steps(config, train_json)
    write_status(
        args.status,
        state="waiting_for_training",
        run_dir=str(run_dir),
        target_global_step=target_step,
        last_global_step=None,
    )

    while True:
        if not last_path.is_file():
            write_status(args.status, state="waiting_for_first_checkpoint", target_global_step=target_step)
            time.sleep(args.poll_seconds)
            continue
        try:
            last_checkpoint = load_last_checkpoint(last_path)
            global_step = int(last_checkpoint.get("global_step", -1))
        except Exception as error:  # A checkpoint may be between writes.
            write_status(args.status, state="waiting_for_readable_checkpoint", error=str(error))
            time.sleep(args.poll_seconds)
            continue
        if global_step < target_step:
            write_status(
                args.status,
                state="waiting_for_training",
                target_global_step=target_step,
                last_global_step=global_step,
            )
            time.sleep(args.poll_seconds)
            continue
        break

    write_status(
        args.status,
        state="training_complete_settling",
        target_global_step=target_step,
        last_global_step=global_step,
        settle_seconds=args.settle_seconds,
    )
    time.sleep(args.settle_seconds)
    final_checkpoint = load_last_checkpoint(last_path)
    final_step = int(final_checkpoint.get("global_step", -1))
    if final_step < target_step:
        # Training was resumed or the checkpoint changed while settling.
        write_status(args.status, state="training_resumed", last_global_step=final_step)
        return main()

    best_path, best_score = select_best_checkpoint(final_checkpoint, checkpoint_dir)
    command = [
        sys.executable,
        str(args.ablation_script.resolve()),
        "--config", str(config_path),
        "--checkpoint", str(best_path),
        "--test-json", str(Path(config["data"]["test_path"])),
        "--image-root", str(Path(config["data"]["image_root"])),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--output", str(args.output),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    write_status(
        args.status,
        state="running_ablation",
        selected_checkpoint=str(best_path),
        selected_val_ce_loss=best_score,
        command=command,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    write_status(
        args.status,
        state="completed" if result.returncode == 0 else "ablation_failed",
        selected_checkpoint=str(best_path),
        selected_val_ce_loss=best_score,
        ablation_exit_code=result.returncode,
        result_path=str(args.output),
        log_path=str(args.log),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
