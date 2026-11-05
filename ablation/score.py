#!/usr/bin/env python3
"""Score only complete five-mode coverage with the repository's frozen scorers."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from common import (COUNTS, MODES, PROJECT, digest, item_path, latent_source,
                    load_data, qa_identity, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from run import check_binding, read_result
    from lvr.eval.paper_benchmark import answer_parsing_metadata, score_record, summarize
    config = json.loads((args.output / "run_config.json").read_text())
    check_binding(config)
    rows, protocol, _, donors = load_data(config["suite"])
    os.environ["LVR_REASONING_SUITE"] = config["suite"]
    binding = digest(config)
    wanted = set(config["selected_ids"])
    rows = [row for row in rows if row["id"] in wanted]
    if config["scope"] == "full" and len(rows) != sum(COUNTS.values()):
        raise ValueError("A full result must contain every benchmark record")
    summaries = {}
    for mode in MODES:
        expected_paths = {item_path(args.output, "answer", row["id"], mode) for row in rows}
        actual = set((args.output / "predictions" / mode).glob("*.json"))
        if actual != expected_paths:
            raise ValueError(f"Missing or foreign predictions for {mode}: {len(actual)}/{len(expected_paths)}")
        scored = []
        for row in rows:
            result = read_result(item_path(args.output, "answer", row["id"], mode), row["id"], mode, binding)
            source, kind = latent_source(mode, row["id"], donors)
            if (result.get("latent_source_id"), result.get("latent_kind"), result.get("qa_inputs_sha256")) != (
                    source, kind or "zero_before_projector", qa_identity(row)):
                raise ValueError("Latent source or recipient QA differs from the frozen five-mode condition")
            score = score_record(row, result["prediction"], truncated=result["reached_token_limit"])
            scored.append({**result, **score})
        if config["scope"] == "full":
            summary = summarize(rows, scored, protocol)
        else:
            summary = {"status": "smoke_only", "generations": len(scored),
                       "sample_accuracy": sum(bool(r["correct"]) for r in scored) / len(scored)}
        selected = [(row, result) for row, result in zip(rows, scored) if row["benchmark"] == "HRBench4K"]
        summary["hrbench4k_helper_kind"] = {}
        for kind in sorted({r["metadata"]["latent_ablation"]["helper_kind"] for r, _ in selected}):
            values = [result for row, result in selected if row["metadata"]["latent_ablation"]["helper_kind"] == kind]
            summary["hrbench4k_helper_kind"][kind] = {"records": len(values),
                       "accuracy": sum(bool(r["correct"]) for r in values) / len(values)}
        summaries[mode] = summary
        directory = args.output / "scores" / mode
        write_json(directory / "summary.json", summary)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored))
    result = {"scope": config["scope"], "run_sha256": binding, "answer_parsing": answer_parsing_metadata(),
              "modes": summaries}
    write_json(args.output / "summary.json", result)
    if config["scope"] == "full":
        table = []
        for name in COUNTS:
            entry = {"benchmark": name}
            for mode in MODES:
                entry[mode] = next(r["accuracy"] for r in summaries[mode]["metrics"] if r["benchmark"] == name)
            table.append(entry)
        table.append({"benchmark": "Macro mean", **{mode: summaries[mode]["macro_accuracy"] for mode in MODES}})
        with (args.output / "main_table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["benchmark", *MODES])
            writer.writeheader()
            writer.writerows(table)
        lines = ["| Benchmark | " + " | ".join(MODES) + " |", "| --- | " + " | ".join(["---:"] * len(MODES)) + " |"]
        lines.extend("| " + row["benchmark"] + " | " + " | ".join(f"{100 * row[mode]:.2f}" for mode in MODES) + " |" for row in table)
        (args.output / "main_table.md").write_text("\n".join(lines) + "\n")
    print("SCORED", len(rows) * len(MODES), config["scope"], "generations", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT / "src"))
    main()
