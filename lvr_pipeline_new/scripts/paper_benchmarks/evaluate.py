"""Run the complete paper benchmark with native processor resolution and direct answers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lvr.eval.paper_benchmark import (DEFAULT_SUITE, ROOT, answer_parsing_metadata,
                                      file_sha, load_suite, model_item, write_json)


def bind_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Resume inputs/code/model differ: {path}; use a new run directory")
    write_json(path, value)


def read_journal(path, wanted):
    """Recover only a torn final write; reject duplicate/foreign/interior damage."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows, good_bytes = [], 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt completed journal line: {path}")
            with path.open("r+b") as stream:
                stream.truncate(good_bytes)
            break
        rows.append(row)
        good_bytes += len(line)
    if rows and good_bytes == len(raw) and not raw.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(wanted):
        raise ValueError("Duplicate/foreign generation IDs in resume journal")
    return rows


def smoke_rows(rows):
    chosen = {}
    for row in rows:
        chosen.setdefault(row["benchmark"], row)
    for key, fn in (("largest_image", lambda r: max(im["width"]*im["height"] for im in r["images"])),
                    ("largest_total", lambda r: sum(im["width"]*im["height"] for im in r["images"])),
                    ("most_images", lambda r: len(r["images"]))):
        chosen[key] = max(rows, key=fn)
    return list({r["id"]: r for r in chosen.values()}.values())


def worker(args):
    config = json.loads((args.output / "run_config.json").read_text())
    rows, protocol, _ = load_suite(Path(config["suite"]))
    if file_sha(Path(config["suite"]) / "manifest.json") != config["suite_manifest_sha256"]:
        raise ValueError("Suite changed since launch")
    for relative, sha in config["code"].items():
        if file_sha(ROOT / relative) != sha:
            raise ValueError(f"Evaluation code changed since launch: {relative}")
    selected = smoke_rows(rows) if args.smoke else rows[args.shard_index::config["shards"]]
    output = args.output / ("smoke" if args.smoke else f"shards/{args.shard_index:02d}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {"parent_config_sha256": file_sha(args.output / "run_config.json"),
                   "shard_index": args.shard_index, "smoke": args.smoke, "ids": [r["id"] for r in selected]}
        bind_json(output / "run_config.json", binding)
        journal = output / "generations.jsonl"
        done = {r["id"] for r in read_journal(journal, binding["ids"])}
        pending = [r for r in selected if r["id"] not in done]
        if not pending:
            write_json(output / "status.json", {"status": "complete", "generations": len(done)})
            return
        import torch
        from transformers import AutoProcessor
        from lvr.data.crop_qa import CropQACollator
        from lvr.models.visual_bottleneck import VisualBottleneck
        torch.set_num_threads(2)
        torch.manual_seed(42)
        for key in ("checkpoint_identity", "source_checkpoint_identity"):
            identity = config[key]
            if identity:
                stat = Path(identity["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
                    raise ValueError("Checkpoint changed after launch")
        for filename, sha in config["processor_files"].items():
            if file_sha(Path(config["model"]) / filename) != sha:
                raise ValueError("Base processor/configuration changed after launch")
        device = f"cuda:{args.shard_index}"
        torch.cuda.set_device(device)
        processor = AutoProcessor.from_pretrained(config["model"], local_files_only=True)
        processor.tokenizer.padding_side = "right"
        inference = protocol["inference"]
        if inference["original_pixels"] is not None or inference["processor"] != "base_model_defaults":
            raise ValueError("The authoritative protocol forbids an additional resolution cap")
        # Never call cap_qwen_pixels or load a processor from the training recipe.
        collator = CropQACollator(processor=processor, image_root=config["suite"], original_pixels=None,
                                  crop_pixels=262144, max_length=inference["max_input_tokens"],
                                  system_prompt=inference["system_prompt"], include_answers=False,
                                  include_encoder=False, num_latents=0 if config["mode"] in ("base", "plain") else 8)
        bind_json(output / "processor.json", processor.image_processor.to_dict())
        if config["mode"] == "base":
            model = VisualBottleneck.from_pretrained(config["model"], device)
        elif config["mode"] == "plain":
            model = VisualBottleneck.from_checkpoint(config["checkpoint"], device, base_model_path=config["model"])
            if model.hparams.num_latents != 0:
                raise ValueError("Plain SFT evaluation requires a zero-latent checkpoint")
        else:
            from lvr.models.latent_prediction import LatentPrediction
            model = LatentPrediction.from_checkpoint(config["checkpoint"], device,
                      source_checkpoint=config["source_checkpoint"], base_model_path=config["model"])
        if config["expected_step"] is not None and model.checkpoint_step != config["expected_step"]:
            raise ValueError("Loaded checkpoint does not match the development-selected step")
        network = model if config["mode"] in ("base", "plain") else model.network
        max_positions = network.qa.config.text_config.max_position_embeddings
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in pending:
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                batch = collator([model_item(row, Path(config["suite"]))])["qa"]
                input_tokens = int(batch["input_ids"].shape[-1])
                if input_tokens + inference["max_new_tokens"] > max_positions:
                    raise ValueError(f"Context overflow without truncation: {row['id']}: {input_tokens}")
                grid = batch["image_grid_thw"].tolist()
                qa = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                latent = model.predict_latents(qa) if config["mode"] == "predicted" else None
                tokens = network.generate_answers(qa, latent, max_new_tokens=inference["max_new_tokens"])
                patch = processor.image_processor.patch_size
                result = {"id": row["id"], "benchmark": row["benchmark"], "task": row["task"],
                          "prediction": processor.tokenizer.decode(tokens[0], skip_special_tokens=True),
                          "generated_tokens": int(tokens.shape[-1]),
                          "reached_token_limit": int(tokens.shape[-1]) >= inference["max_new_tokens"],
                          "input_tokens": input_tokens, "image_grid_thw": grid,
                          "original_sizes_wh": [[im["width"], im["height"]] for im in row["images"]],
                          "processed_sizes_wh": [[w*patch, h*patch] for _, h, w in grid],
                          "peak_memory_reserved": torch.cuda.max_memory_reserved(device),
                          "seconds": time.monotonic() - started}
                with journal.open("a") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                done.add(row["id"])
                write_json(output / "status.json", {"status": "running", "completed": len(done), "total": len(selected), "last_id": row["id"]})
                print("GENERATED", len(done), "/", len(selected), row["id"], flush=True)
                del batch, qa, tokens, latent
        write_json(output / "status.json", {"status": "complete", "generations": len(done)})


def code_files():
    return [*sorted((ROOT / "scripts/paper_benchmarks").glob("*.py")),
            *sorted((ROOT / "src/lvr").rglob("*.py"))]


def make_config(args, protocol):
    def identity(path):
        return {"path": str(path.resolve()), "sha256": file_sha(path), "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns} if path else None
    model = args.model.resolve()
    weights = {p.name: file_sha(p) for p in sorted(model.glob("*.safetensors"))}
    if not weights:
        raise ValueError("Base model safetensors are missing")
    return {"schema": "paper-benchmark-run-v1", "suite": str(args.suite.resolve()),
            "suite_manifest_sha256": file_sha(args.suite / "manifest.json"),
            "mode": args.mode, "model": str(model), "model_weights": weights,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_identity": identity(args.checkpoint), "expected_step": args.expected_step,
            "source_checkpoint": str(args.source_checkpoint.resolve()) if args.source_checkpoint else None,
            "source_checkpoint_identity": identity(args.source_checkpoint), "shards": args.gpus,
            "inference": protocol["inference"],
            "answer_parsing": answer_parsing_metadata(),
            "processor_files": {p.name: file_sha(p) for p in model.iterdir() if p.suffix in (".json", ".jinja")},
            "code": {str(p.relative_to(ROOT)): file_sha(p) for p in code_files()}}


def run(args):
    rows, protocol, manifest = load_suite(args.suite)
    authoritative = args.suite.parent / "configs/protocol.json"
    if authoritative.exists() and json.loads(authoritative.read_text()) != protocol:
        raise ValueError("Prepared data differ from the authoritative protocol")
    if args.dry_run:
        print(json.dumps({"suite": str(args.suite), "records": len({r['base_id'] for r in rows}),
                          "generations": len(rows), "mode": args.mode, "shards": args.gpus,
                          "inference": protocol["inference"], "expected_step": args.expected_step}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = make_config(args, protocol)
        bind_json(args.output / "run_config.json", config)
        import torch
        if torch.cuda.device_count() < args.gpus:
            raise RuntimeError("Insufficient visible GPUs")
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                           PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        cache = args.output / "cache"
        for name, path in {"TMPDIR": ROOT.parent / "paper_experiments/.tmp", "TRITON_CACHE_DIR": cache / "triton",
                           "TORCH_EXTENSIONS_DIR": cache / "torch_extensions", "XDG_CACHE_HOME": cache / "xdg",
                           "HF_HOME": ROOT.parent / "paper_experiments/models/.cache/huggingface"}.items():
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(args.output)]
        children = []
        try:
            write_json(args.output / "status.json", {"status": "smoke_running"})
            with (args.output / "smoke.log").open("a") as log:
                subprocess.run([*command, "--shard-index", "0", "--smoke"], stdout=log, stderr=subprocess.STDOUT,
                               env=environment, cwd=ROOT, check=True)
            write_json(args.output / "status.json", {"status": "running", "shards": args.gpus})
            for index in range(args.gpus):
                with (args.output / f"shard_{index:02d}.log").open("a") as log:
                    children.append(subprocess.Popen([*command, "--shard-index", str(index)], stdout=log,
                                      stderr=subprocess.STDOUT, env=environment, cwd=ROOT, start_new_session=True))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("A benchmark worker failed; inspect shard logs. No resizing or sample skipping was applied.")
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError("A benchmark worker failed")
            subprocess.run([os.environ.get("LVR_SCORING_PYTHON", str(ROOT.parent / ".venv-score/bin/python")), str(Path(__file__).with_name("score.py")),
                            "--suite", str(args.suite), "--run", str(args.output)], cwd=ROOT, env=environment, check=True)
            write_json(args.output / "status.json", {"status": "complete", "generations": len(rows)})
        except BaseException as error:
            write_json(args.output / "status.json", {"status": "failed", "error": str(error)})
            raise
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mode", choices=("base", "plain", "predicted"), default="plain")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.suite = args.suite.resolve()
    if args.worker:
        worker(args)
    else:
        if args.gpus < 1 or args.gpus > 8 or args.model is None:
            parser.error("--model and 1..8 GPUs are required")
        if args.mode != "base" and args.checkpoint is None:
            parser.error("Trained models require --checkpoint")
        if args.mode == "base" and (args.checkpoint or args.source_checkpoint):
            parser.error("Base evaluation only uses --model")
        if args.mode == "predicted" and args.source_checkpoint is None:
            parser.error("Predicted latent evaluation requires --source-checkpoint")
        run(args)


if __name__ == "__main__":
    main()
