"""Evaluate crop-conditioned QA generation and latent ablations on a held-out split."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from vlmeval.dataset.utils.vqa_eval import process_answer
from vlmeval.utils.matching_util import can_infer
import yaml

from lvr.data.crop_qa import CropQACollator, CropQADataset
from lvr.data.prompt import cap_qwen_pixels, strip_scaffold_tags
from lvr.eval.scoring import parse_choices_from_text
from lvr.models.visual_bottleneck import VisualBottleneck


def source(item):
    return Path(item["question_image"]).parts[2]


def shuffled_donors(items, seed):
    """Derange within each source, excluding the original image as a donor."""
    rng = random.Random(seed)
    donors = {}
    for name in sorted({source(item) for item in items}):
        group = [item for item in items if source(item) == name]
        for _ in range(100):
            rng.shuffle(group)
            pairs = list(zip(group, group[1:] + group[:1]))
            if all(a["question_image"] != b["question_image"] for a, b in pairs):
                donors.update({a["id"]: b["id"] for a, b in pairs})
                break
        else:
            raise ValueError(f"Cannot construct different-image donors for {name}")
    return donors


def sample_latents(mu, logvar, ids, seed):
    """One draw per sample, invariant to batching and distributed sharding."""
    noise = []
    for sample_id in ids:
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        noise.append(torch.randn(mu.shape[1:], generator=generator))
    return mu + (0.5 * logvar).exp() * torch.stack(noise)


def score_prediction(prediction, item):
    choices = parse_choices_from_text(item["question"])
    reference = item["answer"].strip()
    if choices and reference in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(reference) == 1:
        extracted = can_infer(prediction, {chr(65 + i): text for i, text in enumerate(choices)}) or ""
        return {"choice": extracted, "mcq_correct": int(extracted == reference),
                "normalized_em": None, "token_f1": None}
    prediction = process_answer(strip_scaffold_tags(prediction))
    reference = process_answer(reference)
    pred_tokens, ref_tokens = prediction.split(), reference.split()
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    denominator = len(pred_tokens) + len(ref_tokens)
    f1 = 2 * overlap / denominator if denominator else 1.0
    return {"choice": None, "mcq_correct": None,
            "normalized_em": int(prediction == reference), "token_f1": f1}


def summarize(records):
    summary = {}
    metrics = ("mcq_correct", "normalized_em", "token_f1")
    for mode in sorted({row["mode"] for row in records}):
        rows = [row for row in records if row["mode"] == mode]
        summary[mode] = {}
        for name in ["all", *sorted({row["source"] for row in rows})]:
            subset = [row for row in rows if name == "all" or row["source"] == name]
            result = {"samples": len(subset), "empty": sum(not row["prediction"].strip() for row in subset),
                      "truncated": sum(row["truncated"] for row in subset)}
            for metric in metrics:
                values = [row[metric] for row in subset if row[metric] is not None]
                result[metric] = {"n": len(values), "mean": float(np.mean(values)) if values else None}
            summary[mode][name] = result
    baseline = {row["id"]: row for row in records if row["mode"] == "none"}
    deltas = {}
    for mode in summary:
        if mode == "none":
            continue
        deltas[mode] = {}
        for metric in metrics:
            differences = [row[metric] - baseline[row["id"]][metric] for row in records
                           if row["mode"] == mode and row[metric] is not None]
            if differences:
                values = np.asarray(differences)
                rng = np.random.default_rng(42)
                bootstrap = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
                deltas[mode][metric] = {"n": len(values), "mean": float(values.mean()),
                                       "paired_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    return {"metrics": summary, "deltas_vs_no_latent": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--qa-system-prompt", default="You are a helpful assistant.")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=2))

    def barrier():
        if world > 1:
            dist.barrier()

    def log(event, **fields):
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "rank": rank,
                          "event": event, **fields}), flush=True)

    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    data = config["data"]
    dataset = CropQADataset(data.get("val_path", "data/default_training/sft/test"), data.get("val_boxes_path", "data/default_training/align/test"))
    items = dataset.items
    if args.limit is not None:
        indices = np.linspace(0, len(items) - 1, min(args.limit, len(items)), dtype=int)
        items = [items[index] for index in indices]
    donors = {} if args.baseline_only else shuffled_donors(items, args.seed)
    by_id = {item["id"]: item for item in items}
    local_items = sorted(items, key=lambda item: (len(item["boxes"]), len(item["question"])))[rank::world]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "run.json").exists():
            raise FileExistsError(f"Use a new output directory: {args.output}")
    barrier()
    log("loading_checkpoint", checkpoint=str(args.checkpoint))
    model = VisualBottleneck.from_checkpoint(args.checkpoint, device)
    step = model.checkpoint_step
    modes = ["none", "mu", "shuffled_mu"]
    if model.hparams.gaussian:
        modes = ["none", "mu", "sample", "shuffled_mu", "shuffled_sample"]
    if args.baseline_only:
        modes = ["none"]
    processor = AutoProcessor.from_pretrained(data["processor_name_or_path"])
    processor.tokenizer.padding_side = "right"
    cap_qwen_pixels(processor, data["original_pixels"])
    kwargs = {key: data[key] for key in ("image_root", "num_latents", "original_pixels", "crop_pixels", "max_length")}

    def loader(encoder):
        return DataLoader(local_items, batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True, collate_fn=CropQACollator(
                              processor, **kwargs, include_answers=False, include_encoder=encoder,
                              encoder_uses_question=model.hparams.encoder_uses_question,
                              encoder_type=model.hparams.encoder_type,
                              encoder_includes_original=model.hparams.encoder_includes_original,
                              system_prompt="You are a helpful assistant." if encoder else args.qa_system_prompt))

    if rank == 0:
        files = [Path(__file__), Path(__file__).parents[1] / "models/visual_bottleneck.py",
                 Path(__file__).parents[1] / "data/crop_qa.py"]
        manifest = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_bytes": args.checkpoint.stat().st_size,
                    "checkpoint_step": step, "samples": len(items), "source_counts": dict(Counter(map(source, items))),
                    "world_size": world, "batch_size": args.batch_size, "seed": args.seed, "modes": modes,
                    "max_new_tokens": args.max_new_tokens, "do_sample_tokens": False,
                    "qa_system_prompt": args.qa_system_prompt, "mcq_scorer": "VLMEvalKit can_infer (exact matching)",
                    "data_config": data, "donors": donors,
                    "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "scope": "Held-out QA with supplied crops; open-answer EM/F1 are lexical metrics, not semantic accuracy."}
        (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("model_loaded", modes=modes, samples=len(local_items), checkpoint_step=step)
    mu = samples = None
    if not args.baseline_only:
        cache = {"ids": [], "mu": [], "logvar": []}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader(True):
                mu, logvar = model.encode({key: value.to(device) for key, value in batch["encoder"].items()})
                if not torch.isfinite(mu).all() or (logvar is not None and not torch.isfinite(logvar).all()):
                    raise ValueError("The encoder produced nonfinite latent parameters")
                cache["ids"].extend(batch["ids"])
                cache["mu"].append(mu.cpu())
                if logvar is not None:
                    cache["logvar"].append(logvar.cpu())
                log("encoding", completed=len(cache["ids"]), total=len(local_items))
        cache["mu"] = torch.cat(cache["mu"])
        cache["logvar"] = torch.cat(cache["logvar"]) if cache["logvar"] else None
        torch.save(cache, args.output / f"latents.rank{rank}.pt")
        barrier()
        caches = [torch.load(args.output / f"latents.rank{i}.pt", weights_only=True) for i in range(world)]
        ids = [sample_id for part in caches for sample_id in part["ids"]]
        assert len(ids) == len(set(ids)) == len(items)
        positions = {sample_id: i for i, sample_id in enumerate(ids)}
        mu = torch.cat([part["mu"] for part in caches])
        if model.hparams.gaussian:
            samples = sample_latents(mu, torch.cat([part["logvar"] for part in caches]), ids, args.seed)
        del caches, cache
    eos = model.qa.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    completed = 0
    with (args.output / f"predictions.rank{rank}.jsonl").open("x") as stream:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader(False):
                qa = {key: value.to(device) for key, value in batch["qa"].items()}
                for mode in modes:
                    latent_ids = [donors[sample_id] if mode.startswith("shuffled") else sample_id for sample_id in batch["ids"]]
                    bank = samples if "sample" in mode else mu
                    latent = None if mode == "none" else bank[[positions[sample_id] for sample_id in latent_ids]].to(device)
                    tokens = model.generate_answers(qa, latent, max_new_tokens=args.max_new_tokens,
                                                    pad_token_id=processor.tokenizer.pad_token_id).cpu().tolist()
                    predictions = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                    for sample_id, donor, prediction, generated in zip(batch["ids"], latent_ids, predictions, tokens):
                        item = by_id[sample_id]
                        length = next((i + 1 for i, token in enumerate(generated) if token in eos), len(generated))
                        row = {"id": sample_id, "source": source(item), "mode": mode,
                               "question": item["question"], "reference": item["answer"], "prediction": prediction,
                               "crop_count": len(item["boxes"]), "latent_source_id": donor if mode != "none" else None,
                               "generated_tokens": length, "truncated": not any(token in eos for token in generated),
                               **score_prediction(prediction, item)}
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                completed += len(batch["ids"])
                log("generation", completed=completed, total=len(local_items), elapsed_seconds=time.monotonic() - started)
    barrier()
    if rank == 0:
        records = [json.loads(line) for i in range(world)
                   for line in (args.output / f"predictions.rank{i}.jsonl").read_text().splitlines()]
        assert len(records) == len({(row["id"], row["mode"]) for row in records}) == len(items) * len(modes)
        result = {"samples": len(items), "prediction_count": len(records), "checkpoint_step": step,
                  "completed_at": datetime.now(timezone.utc).isoformat(), **summarize(records)}
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        log("complete", summary=str(args.output / "summary.json"))
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
