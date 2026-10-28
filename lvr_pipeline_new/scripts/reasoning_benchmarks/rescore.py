"""Rescore existing reasoning predictions offline into a separate output directory."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from lvr.eval.answer_extraction import ANSWER_POLICIES
from lvr.eval.mira_scoring import MIRA_POLICIES
from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT, file_sha, load_suite
from lvr.eval.reasoning_scoring import score


def rescore_rows(records, predictions, *, answer_policy="structured_v1", mira_policy="task_normalized_v1"):
    """Apply the same policy to every output; verify the frozen scores first."""
    lookup = {r["id"]: r for r in records}
    if len(lookup) != len(records) or len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate benchmark or prediction IDs")
    if {r["id"] for r in predictions} != lookup.keys():
        raise ValueError("Prediction IDs do not match the selected benchmark records")
    rescored = []
    for prediction in predictions:
        row = lookup[prediction["id"]]
        if row["benchmark"] != prediction["benchmark"]:
            raise ValueError("Prediction benchmark differs from its record")
        old = score(row, prediction["prediction"])
        for key in ("correct", "parsed", "extracted"):
            if old[key] != prediction[key]:
                raise ValueError(f"Frozen score differs for {row['id']}: {key}")
        previous = (score(row, prediction["prediction"], mira_policy=mira_policy)
                    if row["benchmark"] == "MIRA" else old)
        new = score(row, prediction["prediction"], answer_policy=answer_policy,
                    mira_policy=mira_policy, truncated=prediction.get("reached_token_limit", False))
        rescored.append({"id": row["id"], "benchmark": row["benchmark"], "task": row["task"],
                         "mode": prediction["mode"], "prediction": prediction["prediction"],
                         "reached_token_limit": prediction.get("reached_token_limit", False),
                         "strict": old, "previous": previous, "rescored": new})
    return rescored


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["benchmark"]].append(row)
    summary = []
    for benchmark, items in sorted(groups.items()):
        result = {"benchmark": benchmark, "n": len(items)}
        for policy in ("strict", "previous", "rescored"):
            result[policy + "_correct"] = sum(bool(r[policy]["correct"]) for r in items)
            result[policy + "_parse_failures"] = sum(not r[policy]["parsed"] for r in items)
            result[policy + "_accuracy_percent"] = 100 * result[policy + "_correct"] / len(items)
        result["gained"] = sum(not r["previous"]["correct"] and r["rescored"]["correct"] for r in items)
        result["lost"] = sum(r["previous"]["correct"] and not r["rescored"]["correct"] for r in items)
        assert result["rescored_correct"] - result["previous_correct"] == result["gained"] - result["lost"]
        summary.append(result)
    return summary


def write_rescored(records, input_path, output, *, answer_policy="structured_v1", mira_policy="task_normalized_v1"):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if output == input_path.parent or output in input_path.parents:
        raise ValueError("Use a separate directory; never overwrite source evaluation artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Rescore output must be empty")
    before = file_sha(input_path)
    predictions = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    rows = rescore_rows(records, predictions, answer_policy=answer_policy, mira_policy=mira_policy)
    path = output / "predictions_rescored.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if file_sha(input_path) != before:
        raise RuntimeError("Source predictions changed during rescoring")
    result = {"status": "passed", "samples": len(rows), "input": str(input_path), "input_sha256": before,
              "answer_policy": answer_policy, "mira_policy": mira_policy,
              "metrics": summarize(rows), "code_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in (
                  Path(__file__), ROOT / "src/lvr/eval/answer_extraction.py", ROOT / "src/lvr/eval/reasoning_scoring.py",
                  ROOT / "src/lvr/eval/mira_scoring.py")}}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original predictions.jsonl with frozen strict scores")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--selection", choices=("clean", "full"), default="clean")
    parser.add_argument("--answer-policy", choices=ANSWER_POLICIES, default="structured_v1")
    parser.add_argument("--mira-policy", choices=MIRA_POLICIES, default="task_normalized_v1")
    args = parser.parse_args()
    os.environ["LVR_REASONING_SUITE"] = str(args.suite.resolve())
    predictions = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    modes = {r["mode"] for r in predictions}
    if len(modes) != 1:
        parser.error("Input must contain exactly one evaluation mode")
    records, _ = load_suite(args.suite, benchmarks={r["benchmark"] for r in predictions}, selection=args.selection)
    if modes == {"teacher"}:
        records = [r for r in records if r["helper_images"]]
    result = write_rescored(records, args.input, args.output,
                            answer_policy=args.answer_policy, mira_policy=args.mira_policy)
    print(json.dumps({k: v for k, v in result.items() if k != "code_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
