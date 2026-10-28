#!/usr/bin/env python3
"""Validate complete VLMEvalKit outputs and collect the six LVR benchmark scores."""

import argparse
import ast
import csv
import json
from pathlib import Path


def read_table(path, delimiter="\t"):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / "status.json").is_file():
        candidates = {path.parent.resolve() for path in args.run_dir.glob("*/T*/status.json")}
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one VLMEvalKit run under {args.run_dir}, found {len(candidates)}")
        args.run_dir = candidates.pop()
    status = json.loads((args.run_dir / "status.json").read_text())
    expected = json.loads(args.datasets.read_text())
    results = {}

    for name, metadata in expected.items():
        state = status["datasets"][name]
        if state.get("error_message"):
            raise ValueError(f"{name}: scoring failed: {state['error_message']}")
        if state["status"] != "done":
            raise ValueError(f"{name}: evaluation is {state['status']}")
        prediction_file = args.run_dir / state["prediction_file"]
        predictions = read_table(prediction_file)
        extras = [ast.literal_eval(row["extra_records"]) for row in predictions]
        raw_outputs = [extra["raw_output"] for extra in extras]
        indices = {row["index"] for row in predictions}
        if len(predictions) != metadata["samples"] or len(indices) != len(predictions):
            raise ValueError(f"{name}: unexpected prediction count or duplicate indices")
        prefix = prediction_file.with_suffix("")
        mme = name == "MME-RealWorld-Lite"
        score_file = Path(f"{prefix}_score.tsv" if mme else f"{prefix}_exact_matching_result.tsv")
        scored = read_table(score_file)
        if len(scored) != len(predictions) or {row["index"] for row in scored} != indices:
            raise ValueError(f"{name}: scored rows do not match predictions")
        hits = [float(row["score" if mme else "hit"]) for row in scored]
        if not all(hit in (0, 1) for hit in hits):
            raise ValueError(f"{name}: unexpected per-question scores")

        if mme:
            rating_file = Path(f"{prefix}_rating.json")
            rating = json.loads(rating_file.read_text())
            metrics = {"Overall": rating["Overall"], **{
                task: rating[task]["Avg"] for task in ("Perception", "Reasoning")
            }}
        else:
            rating_file = Path(f"{prefix}_acc.csv")
            rating = read_table(rating_file, delimiter=",")
            if name.startswith("HRBench"):
                metrics = {row["type"]: float(row["accuracy"]) for row in rating if row["cycle"] == "Average"}
            else:
                if len(rating) != 1:
                    raise ValueError(f"{name}: expected a single aggregate score row")
                metrics = {key: float(value) for key, value in rating[0].items() if key != "split"}

        results[name] = {
            "samples": len(predictions),
            "correct_questions": int(sum(hits)),
            "metrics_percent": {key: value * 100 for key, value in metrics.items()},
            "empty_predictions": sum(not row["prediction"].strip() for row in predictions),
            "generation_failures": sum("Failed to obtain answer" in row["prediction"] for row in predictions),
            "truncated_predictions": sum(extra["truncated"] for extra in extras)
            if all("truncated" in extra for extra in extras) else None,
            "latent_start_outputs": sum("<abs_vis_token>" in raw for raw in raw_outputs),
            "latent_end_outputs": sum("</abs_vis_token>" in raw for raw in raw_outputs),
            "unmatched_answers": None if mme else sum("Failed in Prefetch" in row["log"] for row in scored),
            "prediction_file": str(prediction_file),
            "score_file": str(score_file),
            "rating_file": str(rating_file),
        }

    summary = {"eval_id": status["eval_id"], "model": status["model_name"], "results": results}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
