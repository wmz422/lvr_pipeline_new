"""Audit a deterministic GPU stress run and distinguish capacity from code failures."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from lightning.pytorch import Callback
import torch


def cuda_oom(exception):
    # torch.cuda.OutOfMemoryError aliases torch.OutOfMemoryError, also used on
    # other devices. The text must identify CUDA even when the type matches.
    message = str(exception).lower()
    return isinstance(exception, RuntimeError) and (
        "cuda out of memory" in message or "cuda error: out of memory" in message)


class GPUCapacityAudit(Callback):
    """Keep per-rank completion, real Adam state and final allocator peaks."""

    def __init__(self, sample_ids_path: str):
        self.manifest_path = Path(sample_ids_path)

    def setup(self, trainer, module, stage):
        self.path = Path(trainer.default_root_dir) / f"capacity_rank{trainer.global_rank}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.seen = []
        self.last_step = -1
        self.record = {"rank": trainer.global_rank, "status": "starting", "steps": [],
                       "peak_nonallocator_bytes": 0}
        self.write()

    def write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n")
        temporary.replace(self.path)

    def snapshot(self, trainer):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        nonallocator = max(0, total - free - torch.cuda.memory_reserved())
        external = max(self.record["peak_nonallocator_bytes"], nonallocator)
        self.record.update(total_bytes=total, peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                           peak_nonallocator_bytes=external, estimated_headroom_bytes=total-reserved-external,
                           optimizer_steps=int(trainer.global_step), elapsed_seconds=time.time()-self.started,
                           optimizer_state_entries=len(trainer.optimizers[0].state),
                           seen_sample_ids=sorted(set(self.seen)), samples_processed=len(self.seen))

    def on_train_start(self, trainer, module):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU capacity probes require CUDA")
        self.record.update(status="training", world_size=trainer.world_size,
                           device_name=torch.cuda.get_device_name(),
                           batch_size=trainer.train_dataloader.batch_size,
                           accumulate_grad_batches=trainer.accumulate_grad_batches)
        self.snapshot(trainer)
        self.write()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.seen.extend(batch["ids"])
        self.snapshot(trainer)
        if trainer.global_step != self.last_step:
            self.record["steps"].append({key: self.record[key] for key in (
                "optimizer_steps", "peak_allocated_bytes", "peak_reserved_bytes", "estimated_headroom_bytes",
                "optimizer_state_entries", "samples_processed")})
            self.last_step = trainer.global_step
        self.write()

    def on_train_end(self, trainer, module):
        self.snapshot(trainer)
        manifest = json.loads(self.manifest_path.read_text())
        expected = manifest["sample_ids"][trainer.global_rank::trainer.world_size]
        if self.seen != expected:
            raise RuntimeError("GPU stress sample order/coverage differs from the recorded manifest")
        if not self.record["optimizer_state_entries"]:
            raise RuntimeError("GPU stress run did not materialize Adam optimizer state")
        self.record.update(status="complete", sample_order_verified=True)
        self.write()

    def on_exception(self, trainer, module, exception):
        if not hasattr(self, "record"):
            self.setup(trainer, module, "fit")
        self.record.update(status="failed", error_kind="cuda_oom" if cuda_oom(exception) else "other",
                           error_type=type(exception).__name__, error=str(exception))
        self.write()


def assess_probe(returncode, records, *, world_size, steps, headroom_gib, log_text=""):
    """Only explicit CUDA OOM or insufficient measured margin permits a smaller batch."""
    if not math.isfinite(headroom_gib) or headroom_gib < 0:
        raise ValueError("GPU headroom must be finite and nonnegative")
    failures = [row for row in records if row.get("status") == "failed"]
    if any(row.get("error_kind") != "cuda_oom" for row in failures):
        return {"status": "error", "reason": "non_oom_rank_failure", "failures": failures}
    if returncode:
        explicit = bool(failures) or "cuda out of memory" in log_text.lower() or "cuda error: out of memory" in log_text.lower()
        return {"status": "capacity" if explicit else "error", "reason": "cuda_oom" if explicit else "process_failure",
                "returncode": returncode, "failures": failures}
    if len(records) != world_size or sorted(row.get("rank", -1) for row in records) != list(range(world_size)):
        return {"status": "error", "reason": "missing_or_duplicate_rank_completion"}
    for row in records:
        if (row.get("status") != "complete" or row.get("optimizer_steps") != steps
                or not row.get("sample_order_verified") or not row.get("optimizer_state_entries")):
            return {"status": "error", "reason": "incomplete_stress_or_optimizer_state", "rank": row.get("rank")}
        if not 0 < row.get("peak_allocated_bytes", 0) <= row.get("peak_reserved_bytes", 0) <= row.get("total_bytes", 0):
            return {"status": "error", "reason": "invalid_gpu_memory_measurement", "rank": row.get("rank")}
    minimum = min(row["estimated_headroom_bytes"] for row in records) / 2**30
    return {"status": "passed" if minimum >= headroom_gib else "capacity",
            "reason": "all_ranks_passed" if minimum >= headroom_gib else "insufficient_headroom",
            "minimum_headroom_gib": minimum,
            "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in records) / 2**30,
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in records) / 2**30}


def select_largest(candidates, run_candidate):
    attempts = []
    for batch in candidates:
        result = dict(run_candidate(batch), batch_size=batch)
        attempts.append(result)
        if result["status"] == "passed":
            return batch, attempts
        if result["status"] != "capacity":
            raise RuntimeError(f"Batch {batch} failed for a non-capacity reason: {result}")
    raise RuntimeError(f"No candidate batch fits with the requested memory margin: {attempts}")
