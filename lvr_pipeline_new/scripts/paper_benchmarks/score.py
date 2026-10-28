"""CPU acceptance and final scoring for the unified paper protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lvr.eval.paper_benchmark import (DEFAULT_SUITE, DEFAULT_MATHVISTA_POLICY, DEFAULT_WRAPPER_POLICY, DEFAULT_ANSWER_POLICY,
                                      ROOT, answer_parsing_metadata, check_counts, file_sha, load_suite,
                                      score_record, summarize, write_json)
from lvr.eval.reasoning_benchmark import asset_path
from lvr.eval.answer_extraction import ANSWER_POLICIES, MATHVISTA_POLICIES, WRAPPER_POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, help="Optional new directory for offline rescoring")
    parser.add_argument("--wrapper-policy", choices=WRAPPER_POLICIES,
                        help="Default: use the run's bound template or the latest template")
    parser.add_argument("--mathvista-policy", choices=MATHVISTA_POLICIES,
                        help="numeric_v1 extracts one number from numeric answers; strict reproduces the original policy")
    parser.add_argument("--answer-policy", choices=ANSWER_POLICIES,
                        help="Default: run-bound policy, or the latest template for new runs")
    args = parser.parse_args()
    os.environ["LVR_REASONING_SUITE"] = str(args.suite.resolve())
    config = json.loads((args.run / "run_config.json").read_text()) if args.run and not args.accept else {}
    bound = config.get("answer_parsing", {})
    args.wrapper_policy = args.wrapper_policy or bound.get("wrapper_policy", DEFAULT_WRAPPER_POLICY)
    args.mathvista_policy = args.mathvista_policy or bound.get("mathvista_policy", DEFAULT_MATHVISTA_POLICY)
    args.answer_policy = args.answer_policy or bound.get("answer_policy", DEFAULT_ANSWER_POLICY)
    parsing = answer_parsing_metadata(wrapper_policy=args.wrapper_policy, mathvista_policy=args.mathvista_policy,
                                      answer_policy=args.answer_policy)
    if args.run and args.output is None and any(bound.get(k, parsing[k]) != parsing[k]
                                              for k in ("wrapper_policy", "mathvista_policy", "answer_policy")):
        parser.error("Changing a run's parsing policy requires --output with a new directory")
    if args.accept:
        manifest = json.loads((args.suite / "manifest.json").read_text())
        for path, sha in manifest["files"].items():
            if file_sha(args.suite / path) != sha:
                raise ValueError(f"Acceptance source mismatch: {path}")
        rows = [json.loads(line) for line in (args.suite / "records.jsonl").read_text().splitlines()]
        protocol = json.loads((args.suite / "protocol.json").read_text())
        check_counts(rows, protocol)
        checked = set()
        scores = []
        for row in rows:
            for im in row["images"] + row["helper_images"]:
                if im["path"] not in checked:
                    asset_path(args.suite, im)
                    checked.add(im["path"])
            # MathVista stores the answer TEXT; its prompt requests the option
            # position. Text such as '(b)' can itself be an option at position E.
            reference = (chr(65 + row["options"].index(row["answer"]))
                         if row["scoring"]["kind"] == "mathvista" and row["scoring"]["question_type"] == "multi_choice"
                         else row["answer"])
            result = score_record(row, "<answer>" + reference + "</answer>", mathvista_policy=args.mathvista_policy,
                                  wrapper_policy=args.wrapper_policy, answer_policy=args.answer_policy)
            if not result["correct"]:
                raise ValueError(f"Reference answer failed: {row['id']}: {result}")
            scores.append({"id": row["id"], **result})
        report = {"status": "passed", "reference_correct": len(scores), "verified_assets": len(checked),
                  "records": protocol["total_records"], "generations": len(rows),
                  "summary": summarize(rows, scores, protocol, mathvista_policy=args.mathvista_policy,
                                       wrapper_policy=args.wrapper_policy, answer_policy=args.answer_policy)}
        write_json(args.suite / "acceptance.json", report)
        manifest["files"]["acceptance.json"] = file_sha(args.suite / "acceptance.json")
        manifest["status"] = "accepted"
        write_json(args.suite / "manifest.json", manifest)
        print("ACCEPTED", len(rows), "generations;", len(checked), "assets", flush=True)
        return
    if args.run is None:
        parser.error("--run is required unless --accept")
    output = args.output or args.run
    if args.output is not None:
        if output.resolve() == args.run.resolve() or (output.exists() and any(output.iterdir())):
            parser.error("--output must be a new, empty directory separate from --run")
        output.mkdir(parents=True, exist_ok=True)
    rows, protocol, manifest = load_suite(args.suite)
    if config["suite_manifest_sha256"] != file_sha(args.suite / "manifest.json"):
        raise ValueError("Prediction suite binding differs")
    lookup = {r["id"]: r for r in rows}
    positions = {r["id"]: index for index, r in enumerate(rows)}
    results = []
    shard_paths = sorted((args.run / "shards").glob("*/generations.jsonl"))
    if len(shard_paths) != config["shards"]:
        raise ValueError("Missing shard journals")
    for path in shard_paths:
        index = int(path.parent.name)
        shard_config = json.loads((path.parent / "run_config.json").read_text())
        if shard_config["parent_config_sha256"] != file_sha(args.run / "run_config.json") or shard_config["shard_index"] != index:
            raise ValueError("Shard configuration binding differs")
        for line in path.read_text().splitlines():
            result = json.loads(line)
            row = lookup[result["id"]]
            if positions[row["id"]] % config["shards"] != index:
                raise ValueError("Prediction belongs to another shard")
            results.append({**result, **score_record(row, result["prediction"], truncated=result["reached_token_limit"],
                                                    mathvista_policy=args.mathvista_policy,
                                                    wrapper_policy=args.wrapper_policy, answer_policy=args.answer_policy)})
    summary = summarize(rows, results, protocol, mathvista_policy=args.mathvista_policy,
                        wrapper_policy=args.wrapper_policy, answer_policy=args.answer_policy)
    summary["run_config_sha256"] = file_sha(args.run / "run_config.json")
    scoring_config = {"source_run": str(args.run.resolve()), "run_config_sha256": summary["run_config_sha256"],
                      "suite_manifest_sha256": file_sha(args.suite / "manifest.json"),
                      "mathvista_policy": args.mathvista_policy,
                      "answer_parsing": parsing,
                      "generation_files_sha256": {str(p.relative_to(args.run)): file_sha(p) for p in shard_paths},
                      "code_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in (
                          Path(__file__), ROOT / "src/lvr/eval/paper_benchmark.py",
                          ROOT / "src/lvr/eval/reasoning_scoring.py", ROOT / "src/lvr/eval/answer_extraction.py")}}
    write_json(output / "scoring_config.json", scoring_config)
    summary["scoring_config_sha256"] = file_sha(output / "scoring_config.json")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    write_json(output / "summary.json", summary)
    lines = ["| Benchmark | Records | Generations | Accuracy (%) |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['benchmark']} | {r['records']} | {r['generations']} | {100*r['accuracy']:.2f} |" for r in summary["metrics"]]
    lines += [f"| Macro mean | | | {100*summary['macro_accuracy']:.2f} |"]
    lines += ["", "Full declared splits; known training/development image overlaps are retained and disclosed in the suite audit.",
              f"Answer parsing: {parsing['version']}; wrappers={args.wrapper_policy}, MathVista={args.mathvista_policy}.",
              "BLINK: task macro average. MathVista: official normalization with local numeric extraction when selected.",
              f"SAT: {args.answer_policy}, averaged over both option orders; the author's reference score is diagnostic only."]
    (output / "main_table.md").write_text("\n".join(lines) + "\n")
    print("SCORED", len(results), "generations", flush=True)


if __name__ == "__main__":
    main()
