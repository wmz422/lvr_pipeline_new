"""Offline Plain/LVR evaluation on the frozen reasoning benchmark protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, EXPECTED, ROOT, file_sha, load_suite, model_item
from lvr.eval.teacher_shuffle import teacher_record
from lvr.eval.answer_extraction import ANSWER_POLICIES
from lvr.eval.mira_scoring import MIRA_POLICIES
from download import write_json


def summary(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["benchmark"], "all", "all")].append(r)
        groups[(r["benchmark"], r["task"], "all")].append(r)
        if r["difficulty"] is not None:
            groups[(r["benchmark"], r["task"], str(r["difficulty"]))].append(r)
    metrics = []
    for (benchmark, task, difficulty), rows in sorted(groups.items()):
        n = len(rows); correct = sum(r["correct"] for r in rows)
        entry = {"benchmark": benchmark, "task": task, "difficulty": difficulty, "n": n,
                 "correct": correct, "accuracy": correct/n,
                 "parse_failures": sum(not r["parsed"] for r in rows),
                 "truncated": sum(r["reached_token_limit"] for r in rows)}
        planning = [r for r in rows if "shortest" in r]
        if planning:
            entry.update(shortest_n=len(planning), shortest_correct=sum(r["shortest"] for r in planning))
        metrics.append(entry)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--selection", choices=("clean", "full"), default="clean")
    parser.add_argument("--benchmarks", choices=EXPECTED, nargs="+")
    parser.add_argument("--mode", choices=("base", "plain", "predicted", "zero", "shuffled", "teacher", "teacher_shuffled"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/workspace/wmz/.cache/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-loops", type=int, help="Latent prediction rounds; default reads the checkpoint")
    parser.add_argument("--answer-policy", choices=ANSWER_POLICIES, default="strict")
    parser.add_argument("--mira-policy", choices=MIRA_POLICIES, default="strict")
    parser.add_argument("--system-prompt", help="Explicit system instruction; default preserves the existing protocol")
    parser.add_argument("--per-task", type=int, help="Deterministic small acceptance subset; omit for complete evaluation")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_loops is not None and (args.num_loops < 1 or args.mode in ("base", "plain")):
        parser.error("--num-loops requires a latent prediction checkpoint and a positive value")
    if args.mode == "base" and (args.checkpoint or args.source_checkpoint):
        parser.error("Base evaluation loads only --model; checkpoint arguments are not allowed")
    if args.mode != "base" and args.checkpoint is None:
        parser.error("Trained model modes require --checkpoint")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    if args.max_new_tokens < 1 or args.per_task is not None and args.per_task < 1:
        parser.error("Token and sample limits must be positive")
    records, manifest = load_suite(args.suite, benchmarks=args.benchmarks, selection=args.selection)
    lookup = {r["id"]: r for r in records}
    donors = json.loads((args.suite/"donors.json").read_text())[args.selection]
    no_helper = [r["id"] for r in records if not r["helper_images"]]
    if args.mode in ("teacher", "teacher_shuffled"):
        records = [r for r in records if r["helper_images"]]
    if args.per_task is not None:
        buckets = defaultdict(list)
        for r in records:
            buckets[(r["benchmark"], r["task"])].append(r)
        records = [r for _, items in sorted(buckets.items()) for r in sorted(items, key=lambda x: x["id"])[:args.per_task]]
    records = records[args.shard_index::args.shard_count]
    if not records:
        parser.error("Selection/shard contains no samples")
    plan = {"mode": args.mode, "selection": args.selection, "samples": len(records),
            "benchmarks": sorted({r["benchmark"] for r in records}), "per_task": args.per_task,
            "teacher_unavailable_in_requested_selection": len(no_helper) if args.mode in ("teacher", "teacher_shuffled") else 0,
            "ids": [r["id"] for r in records], "shard": [args.shard_index, args.shard_count],
            "max_new_tokens": args.max_new_tokens, "suite_manifest_sha256": file_sha(args.suite/"manifest.json"),
            "original_pixels": "processor_default", "helper_pixels": 262144}
    if args.num_loops is not None:
        plan["num_loops"] = args.num_loops
    if args.answer_policy != "strict" or args.mira_policy != "strict":
        plan.update(answer_policy=args.answer_policy, mira_policy=args.mira_policy)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("checkpoint is missing")
    if args.mode not in ("base", "plain") and (args.source_checkpoint is None or not args.source_checkpoint.is_file()):
        parser.error("LVR modes require --source-checkpoint")
    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    from lvr.models.visual_bottleneck import VisualBottleneck
    from lvr.models.latent_prediction import LatentPrediction
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    common = dict(processor=processor, image_root=args.suite, original_pixels=None,
                  crop_pixels=262144, max_length=131072, include_answers=False)
    if args.system_prompt is not None:
        common["system_prompt"] = args.system_prompt
        plan["system_prompt"] = args.system_prompt
    student = CropQACollator(**common, include_encoder=False, num_latents=0 if args.mode in ("base", "plain") else 8)
    teacher = CropQACollator(**common, include_encoder=True) if args.mode in ("teacher", "teacher_shuffled") else None
    plan.update(checkpoint_sha256=file_sha(args.checkpoint) if args.checkpoint else None, source_checkpoint_sha256=file_sha(args.source_checkpoint) if args.source_checkpoint else None,
                processor_files={p.name: file_sha(p) for p in args.model.iterdir() if p.is_file() and p.suffix in (".json", ".jinja")},
                code={str(p.relative_to(ROOT)): file_sha(p) for p in [Path(__file__), ROOT/"src/lvr/eval/reasoning_benchmark.py", ROOT/"src/lvr/eval/reasoning_scoring.py", ROOT/"src/lvr/eval/answer_extraction.py", ROOT/"src/lvr/eval/mira_scoring.py", ROOT/"src/lvr/data/crop_qa.py", ROOT/"src/lvr/models/latent_prediction.py", ROOT/"src/lvr/models/visual_bottleneck.py"]})
    if args.mode == "base":
        plan["pretrained_model"] = {"path": str(args.model.resolve()), "weights": {
            p.name: file_sha(p) for p in sorted(args.model.glob("*.safetensors"))}}
        if not plan["pretrained_model"]["weights"]:
            parser.error("Base model safetensors weights are missing")
    if args.mode in ("teacher", "teacher_shuffled"):
        plan["code"]["src/lvr/eval/teacher_shuffle.py"] = file_sha(ROOT/"src/lvr/eval/teacher_shuffle.py")
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output/"run_config.json"
    if config.exists() and json.loads(config.read_text()) != plan:
        raise ValueError("Resume protocol/checkpoint/configuration differs; use a new output directory")
    write_json(config, plan)
    journal = args.output/"generations.jsonl"
    results = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    ids = {r["id"] for r in records}
    if len({r["id"] for r in results}) != len(results) or not {r["id"] for r in results} <= ids:
        raise ValueError("Duplicate/foreign resume records")
    done = {r["id"] for r in results}
    pending = [r for r in records if r["id"] not in done]
    if pending:
        if args.mode == "base":
            model = VisualBottleneck.from_pretrained(args.model, args.device)
        elif args.mode == "plain":
            model = VisualBottleneck.from_checkpoint(args.checkpoint, args.device, base_model_path=args.model)
        else:
            model = LatentPrediction.from_checkpoint(args.checkpoint, args.device, source_checkpoint=args.source_checkpoint, base_model_path=args.model)
        prediction_rounds = model.prediction_loop_count(args.num_loops) if args.mode in ("predicted", "shuffled") else 0
        def device(branch):
            return {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in branch.items()}
        with torch.inference_mode(), torch.autocast("cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            for row in pending:
                start = time.monotonic()
                qa = device(student([model_item(row, args.suite)])["qa"])
                latent, donor_id = None, None
                if args.mode == "predicted":
                    latent = model.predict_latents(qa, num_loops=prediction_rounds)
                elif args.mode == "zero":
                    latent = torch.zeros((1, model.hparams.num_latents, model.hparams.latent_dim), device=args.device)
                elif args.mode == "shuffled":
                    donor_id = donors[row["id"]]["id"]
                    if donor_id == row["id"] or donor_id not in lookup:
                        raise ValueError("Invalid frozen shuffled donor")
                    donor_qa = device(student([model_item(lookup[donor_id], args.suite)])["qa"])
                    latent = model.predict_latents(donor_qa, num_loops=prediction_rounds)
                    del donor_qa
                elif args.mode in ("teacher", "teacher_shuffled"):
                    source, donor_id = teacher_record(row, lookup, donors, shuffled=args.mode == "teacher_shuffled")
                    encoder = device(teacher([model_item(source, args.suite, teacher=True)])["encoder"])
                    latent, _ = model.network.encode(encoder)
                    del encoder
                if latent is not None and (tuple(latent.shape) != (1, 8, 512) or not torch.isfinite(latent).all()):
                    raise ValueError("Invalid inference latent")
                network = model if args.mode in ("base", "plain") else model.network
                tokens = network.generate_answers(qa, latent, max_new_tokens=args.max_new_tokens)
                output = {"id": row["id"], "benchmark": row["benchmark"], "task": row["task"], "difficulty": row["difficulty"],
                          "mode": args.mode, "donor_id": donor_id, "prediction": processor.tokenizer.decode(tokens[0], skip_special_tokens=True),
                          "generated_tokens": int(tokens.shape[-1]), "reached_token_limit": int(tokens.shape[-1]) >= args.max_new_tokens,
                          "seconds": time.monotonic()-start}
                output["prediction_num_loops"] = prediction_rounds
                results.append(output)
                with journal.open("a") as f:
                    f.write(json.dumps(output, ensure_ascii=False)+"\n"); f.flush()
                print("GENERATED", len(results), "/", len(records), args.mode, row["id"], flush=True)
        del model; gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    scorer_input, scorer_output = args.output/"scorer_input.jsonl", args.output/"scores.jsonl"
    scorer_input.write_text("".join(json.dumps({"record": lookup[r["id"]], "prediction": r["prediction"],
        "reached_token_limit": r["reached_token_limit"]}, ensure_ascii=False)+"\n" for r in results))
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"src"), MPLBACKEND="Agg", LVR_REASONING_SUITE=str(args.suite.resolve()))
    subprocess.run([str(ROOT/".venv-mentis/bin/python"), "-m", "lvr.eval.reasoning_scoring", str(scorer_input), str(scorer_output),
                    "--answer-policy", args.answer_policy, "--mira-policy", args.mira_policy], env=environment, check=True)
    scores = {r["id"]: r for r in map(json.loads, scorer_output.read_text().splitlines())}
    merged = [{**r, **scores[r["id"]]} for r in results]
    (args.output/"predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in merged))
    write_json(args.output/"summary.json", {"status": "complete", "mode": args.mode, "samples": len(merged),
               "selection": args.selection, "scope": "acceptance_subset" if args.per_task else "full_selected_shard",
               "metrics": summary(merged), "plan": plan})
    print("COMPLETE", args.mode, len(merged), flush=True)


if __name__ == "__main__":
    main()
