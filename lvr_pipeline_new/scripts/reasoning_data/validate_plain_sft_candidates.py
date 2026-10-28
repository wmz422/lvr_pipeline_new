#!/usr/bin/env python3
"""Validate saved Plain SFT candidates on an identical split, with exact sample coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from lvr.data.crop_qa import CropQACollator, CropQADataset
from lvr.data.prompt import cap_qwen_pixels
from lvr.models.visual_bottleneck import VisualBottleneck, _read_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_identity(path):
    """Bind native rank shards (or a full checkpoint) without rereading tensor bytes."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        if not (path / "latest").is_file():
            raise ValueError(f"Not a native DeepSpeed checkpoint: {path}")
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        files = [path]
    return {"path": str(path), "latest": (path / "latest").read_text() if path.is_dir() else None,
            "files": {str(p.relative_to(path)) if path.is_dir() else p.name:
                      {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}}


def prepare_export(source, export, convert):
    """Reuse only exports with a matching native-source and destination identity."""
    identity = checkpoint_identity(source)
    binding_path = export.with_suffix(export.suffix + ".source.json")
    if export.exists():
        if not binding_path.is_file():
            raise ValueError(f"Existing export lacks a native-source binding: {export}")
        binding = json.loads(binding_path.read_text())
        if binding != {"source": identity, "export": checkpoint_identity(export)}:
            raise ValueError("Native checkpoint or its export changed; use new export and result directories")
        return identity
    if binding_path.exists():
        raise ValueError("Export binding exists without its checkpoint; use a fresh export directory")
    export.parent.mkdir(parents=True, exist_ok=True)
    convert(source, export)
    if checkpoint_identity(source) != identity:
        raise ValueError("Native checkpoint changed during export; do not use this export")
    write_json(binding_path, {"source": identity, "export": checkpoint_identity(export)})
    return identity


def validation_dataset(data):
    dataset = CropQADataset(data["val_path"], data.get("val_boxes_path"))
    if data.get("additional_val_path"):
        dataset.items.extend(CropQADataset(data["additional_val_path"]).items)
    if len({item["id"] for item in dataset.items}) != len(dataset):
        raise ValueError("Duplicate validation sample identifiers")
    return dataset


def item_source(item):
    if item.get("source"):
        return "reasoning", item["source"]
    return "old", Path(item["question_image"]).parts[-2]


def training_source_policy(data, validation=None):
    """Freeze source weights from training only, retaining every validation example."""
    train = CropQADataset(data["train_path"], data.get("train_boxes_path"))
    if data.get("additional_train_path"):
        train.items.extend(CropQADataset(data["additional_train_path"]).items)
    validation = validation if validation is not None else validation_dataset(data)
    train_counts = Counter(item_source(item)[1] for item in train.items)
    val_counts = Counter(item_source(item)[1] for item in validation.items)
    if not train_counts or set(train_counts) != set(val_counts):
        raise ValueError("Every training source must have validation coverage, without extra sources")
    train_ids = [item["id"] for item in train.items]
    val_ids = [item["id"] for item in validation.items]
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & set(val_ids):
        raise ValueError("Training IDs must be unique and disjoint from validation")
    return {"format": "plain-sft-train-source-weighted-ce-v1",
            "metric": "train_source_weighted_mean_sample_ce", "direction": "minimize",
            "tie_break": "earliest_checkpoint_step", "train_samples": len(train),
            "validation_samples": len(validation), "train_sample_ids_sha256": digest(train_ids),
            "validation_sample_ids_sha256": digest(val_ids),
            "train_source_counts": dict(sorted(train_counts.items())),
            "validation_source_counts": dict(sorted(val_counts.items())),
            "source_weights": {source: count / len(train) for source, count in sorted(train_counts.items())}}


def weighted_selection_score(metrics, policy):
    weights = policy["source_weights"]
    if (not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)):
        raise ValueError("Invalid fixed training source weights")
    if {key.removeprefix("source/") for key in metrics if key.startswith("source/")} != set(weights):
        raise ValueError("Validation source coverage differs from selection policy")
    score = 0.0
    for source, weight in weights.items():
        metric = metrics[f"source/{source}"]
        if metric["samples"] != policy["validation_source_counts"][source]:
            raise ValueError("Validation source sample count differs from frozen policy")
        value = metric["mean_sample_ce"]
        if not math.isfinite(value):
            raise ValueError("Non-finite candidate selection metric")
        score += weight * value
    return score


def select_candidate_summaries(summaries, expected_steps, policy):
    """Compare the complete predeclared checkpoint set under one frozen dev policy."""
    if (len(summaries) != len(expected_steps)
            or sorted(s["checkpoint_step"] for s in summaries) != sorted(expected_steps)):
        raise ValueError("All expected candidate checkpoints must be validated exactly once")
    scores = []
    for summary in summaries:
        if (summary.get("status") != "complete" or summary.get("selection_policy") != policy
                or summary.get("selection_policy_sha256") != digest(policy)):
            raise ValueError("Candidate is incomplete or uses a different frozen validation policy")
        score = weighted_selection_score(summary["metrics"], policy)
        if not math.isclose(score, summary["selection_score"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Stored candidate score does not match source metrics")
        scores.append({"checkpoint_step": summary["checkpoint_step"], "selection_score": score})
    best = min(scores, key=lambda row: (row["selection_score"], row["checkpoint_step"]))
    return {"status": "complete", "selection_policy": policy, "selection_policy_sha256": digest(policy),
            "candidates": sorted(scores, key=lambda row: row["checkpoint_step"]),
            "selected_checkpoint_step": best["checkpoint_step"], "selected_score": best["selection_score"],
            "test_benchmarks_used_for_selection": False}


@torch.inference_mode()
def score_batch(model, qa):
    """Return per-example NLL; padding and answer length never alter group weighting."""
    hidden = model._hidden(model.qa, qa)
    results = []
    for states, labels in zip(hidden[:, :-1], qa["labels"][:, 1:]):
        valid = labels != -100
        count = int(valid.sum())
        if not count:
            raise ValueError("Validation example has no answer tokens")
        ce = float(model.qa.lm_head(states[valid], labels[valid]).float())
        if not torch.isfinite(torch.tensor(ce)):
            raise ValueError("Non-finite validation CE")
        results.append({"sample_ce": ce, "target_tokens": count, "nll_sum": ce * count})
    return results


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in ("all", f"group/{row['group']}", f"source/{row['source']}"):
            groups[key].append(row)
    return {key: {"samples": len(values), "target_tokens": sum(v["target_tokens"] for v in values),
                  "mean_sample_ce": sum(v["sample_ce"] for v in values) / len(values),
                  "token_ce": sum(v["nll_sum"] for v in values) / sum(v["target_tokens"] for v in values)}
            for key, values in sorted(groups.items())}


def worker(args, config, dataset):
    from transformers import AutoProcessor

    directory = args.output
    plan = json.loads((directory / "plan.json").read_text())
    ids = list(range(args.shard_index, len(dataset), args.gpus))
    complete = directory / f"shard_{args.shard_index:02d}.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["plan_sha256"] == digest(plan) and [r["id"] for r in saved["rows"]] == [dataset[i]["id"] for i in ids]:
            return 0
        raise ValueError(f"Mismatched validation shard: {complete}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = config["data"]
    processor = AutoProcessor.from_pretrained(data["processor_name_or_path"], local_files_only=True)
    processor.tokenizer.padding_side = "right"
    cap_qwen_pixels(processor, data["original_pixels"])
    collator = CropQACollator(processor, data["image_root"], num_latents=0,
                             original_pixels=data["original_pixels"], max_length=data["max_length"],
                             include_encoder=False, include_descriptions=False)
    loader = DataLoader(Subset(dataset, ids), batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=collator)
    model = VisualBottleneck.from_checkpoint(args.checkpoint, "cpu",
                                             base_model_path=data["processor_name_or_path"])
    if model.hparams.num_latents != 0 or model.checkpoint_step != plan["checkpoint_step"]:
        raise ValueError("Expected the exact zero-latent candidate checkpoint")
    # Preserve checkpoint parameter dtypes and native FP32 rotary buffers, matching
    # the shared checkpoint-evaluation path; inference arithmetic uses BF16 autocast.
    model.to(device)
    index = {row["id"]: row for row in dataset.items}
    rows = []
    for batch in loader:
        qa = {key: value.to(device) for key, value in batch["qa"].items()}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            scores = score_batch(model, qa)
        for sample_id, score in zip(batch["ids"], scores):
            group, source = item_source(index[sample_id])
            rows.append({"id": sample_id, "group": group, "source": source,
                         "checkpoint_step": model.checkpoint_step, **score})
        if len(rows) % 100 < args.batch_size:
            print(f"VALIDATION rank={args.shard_index} samples={len(rows)}/{len(ids)}", flush=True)
    if [row["id"] for row in rows] != [dataset[i]["id"] for i in ids]:
        raise ValueError("Validation shard coverage differs from its assigned sample IDs")
    write_json(complete, {"status": "complete", "plan_sha256": digest(plan), "rows": rows})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native DeepSpeed directory or consolidated .ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--selection-policy", type=Path, help="Pretraining-frozen validation source weights and IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpus < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid worker or batch settings")
    args.config, args.checkpoint, args.output = args.config.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if not args.dry_run and not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing source candidate checkpoint: {args.checkpoint}")
    config = yaml.safe_load(args.config.read_text())
    if config["model"]["num_latents"] != 0 or config["model"].get("description_weight", 0):
        raise ValueError("Candidate validation requires Plain SFT without description supervision")
    dataset = validation_dataset(config["data"])
    if args.shard_index is not None:
        return worker(args, config, dataset)
    policy = training_source_policy(config["data"], dataset)
    if args.selection_policy is not None and json.loads(args.selection_policy.read_text()) != policy:
        raise ValueError("Training or validation data changed from the frozen selection policy")
    commands = []
    source = args.checkpoint
    export = args.checkpoint
    if source.is_dir() or source.suffix == ".ckpt" and not source.exists():
        if args.export_dir is None:
            parser.error("--export-dir is required for a native DeepSpeed checkpoint")
        export = args.export_dir.resolve() / (source.stem + "_full.ckpt")
    for rank in range(args.gpus):
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                   "--checkpoint", str(export), "--output", str(args.output), "--gpus", str(args.gpus),
                   "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                   "--device", f"cuda:{rank}", "--shard-index", str(rank)]
        commands.append(command)
    if args.dry_run:
        print(json.dumps({"validation_samples": len(dataset), "source_checkpoint": str(source),
                          "export": str(export), "expected_step": args.expected_step,
                          "original_pixels": config["data"]["original_pixels"],
                          "selection_policy": policy}, indent=2))
        for command in commands:
            print(shlex.join(command))
        return 0
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Insufficient visible CUDA devices for candidate validation")
    if source.is_dir():
        spec = importlib.util.spec_from_file_location("candidate_export", ROOT / "scripts/latent_prediction/run.py")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        identity = prepare_export(source, export, legacy.consolidate_deepspeed_checkpoint)
    else:
        identity = checkpoint_identity(source)
    checkpoint = _read_checkpoint(export)
    step = int(checkpoint["global_step"])
    if args.expected_step is not None and step != args.expected_step:
        raise ValueError(f"Expected checkpoint update {args.expected_step}, found {step}")
    del checkpoint
    plan = {"checkpoint": str(export), "checkpoint_step": step, "checkpoint_bytes": export.stat().st_size,
            "source_checkpoint_identity": identity,
            "checkpoint_mtime_ns": export.stat().st_mtime_ns, "config_sha256": digest(config),
            "samples_sha256": digest(dataset.items), "samples": len(dataset), "gpus": args.gpus,
            "selection_policy": policy, "selection_policy_sha256": digest(policy),
            "batch_size": args.batch_size, "workers": args.workers,
            "precision": "bf16 autocast; checkpoint parameter dtypes preserved",
            "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [Path(__file__).resolve(), *sorted((ROOT / "src/lvr").rglob("*.py"))]}}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Candidate validation inputs/code changed; use a fresh output directory")
    write_json(plan_path, plan)
    children = []
    try:
        for rank, command in enumerate(commands):
            with (args.output / f"shard_{rank:02d}.log").open("a") as handle:
                children.append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                 start_new_session=True))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f"Candidate validation failed; inspect {args.output}")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError(f"Candidate validation failed; inspect {args.output}")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    rows = []
    for rank in range(args.gpus):
        shard = json.loads((args.output / f"shard_{rank:02d}.json").read_text())
        if shard["status"] != "complete" or shard["plan_sha256"] != digest(plan):
            raise ValueError("Incomplete or mismatched validation shard")
        rows.extend(shard["rows"])
    if len(rows) != len(dataset) or {r["id"] for r in rows} != {r["id"] for r in dataset.items}:
        raise ValueError("Validation coverage must contain every sample exactly once, without sampler padding")
    metrics = summarize(rows)
    write_json(args.output / "summary.json", {"status": "complete", "checkpoint_step": step,
               "plan_sha256": digest(plan), "metrics": metrics, "selection_policy": policy,
               "selection_policy_sha256": digest(policy), "selection_score": weighted_selection_score(metrics, policy)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
