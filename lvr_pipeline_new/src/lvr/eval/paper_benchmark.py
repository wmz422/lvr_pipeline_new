"""The single, complete paper benchmark protocol (no historical suite grouping)."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

from lvr.eval.reasoning_benchmark import file_sha, model_item, safe_path
from lvr.eval.answer_extraction import ANSWER_POLICIES, MATHVISTA_METRICS, WRAPPER_POLICIES, repair_answer_wrapper

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT.parent / "paper_experiments/2026-10-23-unified-benchmark"
DEFAULT_SUITE = EXPERIMENT / "data"
SCHEMA = "paper-benchmark-v1"
ANSWER_TEMPLATE_PATH = EXPERIMENT / "configs/answer_parsing.json"
_ANSWER_TEMPLATE_BYTES = ANSWER_TEMPLATE_PATH.read_bytes()
ANSWER_TEMPLATE = json.loads(_ANSWER_TEMPLATE_BYTES)
if (ANSWER_TEMPLATE["schema"] != "lvr-answer-parsing-template-v1"
        or ANSWER_TEMPLATE["wrapper_policy"] not in WRAPPER_POLICIES
        or ANSWER_TEMPLATE["mathvista_policy"] not in MATHVISTA_METRICS
        or ANSWER_TEMPLATE["answer_policy"] not in ANSWER_POLICIES
        or ANSWER_TEMPLATE["mira_policy"] != "strict"):
    raise ValueError("Unsupported answer parsing template")
DEFAULT_MATHVISTA_POLICY = ANSWER_TEMPLATE["mathvista_policy"]
DEFAULT_WRAPPER_POLICY = ANSWER_TEMPLATE["wrapper_policy"]
DEFAULT_ANSWER_POLICY = ANSWER_TEMPLATE["answer_policy"]


def answer_parsing_metadata(*, wrapper_policy=DEFAULT_WRAPPER_POLICY,
                            mathvista_policy=DEFAULT_MATHVISTA_POLICY, answer_policy=DEFAULT_ANSWER_POLICY):
    if (wrapper_policy not in WRAPPER_POLICIES or mathvista_policy not in MATHVISTA_METRICS
            or answer_policy not in ANSWER_POLICIES):
        raise ValueError("Unknown answer parsing policy")
    current = (wrapper_policy == DEFAULT_WRAPPER_POLICY and mathvista_policy == DEFAULT_MATHVISTA_POLICY
               and answer_policy == DEFAULT_ANSWER_POLICY)
    return {"version": ANSWER_TEMPLATE["version"] if current else "custom",
            "base_template_version": ANSWER_TEMPLATE["version"],
            "template_sha256": hashlib.sha256(_ANSWER_TEMPLATE_BYTES).hexdigest(),
            "wrapper_policy": wrapper_policy, "mathvista_policy": mathvista_policy,
            "answer_policy": answer_policy, "mira_policy": "strict"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_counts(rows, protocol):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate generation IDs")
    counts = Counter(r["benchmark"] for r in rows)
    expected = {k: v["generations"] for k, v in protocol["benchmarks"].items()}
    if counts != expected or len(rows) != protocol["total_generations"]:
        raise ValueError(f"Incomplete benchmark coverage: {counts} != {expected}")
    for name, spec in protocol["benchmarks"].items():
        selected = [r for r in rows if r["benchmark"] == name]
        if len({r["base_id"] for r in selected}) != spec["records"]:
            raise ValueError(f"Incorrect base record count: {name}")
    if Counter(r["task"] for r in rows if r["benchmark"] == "BLINK") != protocol["blink_tasks"]:
        raise ValueError("BLINK must contain all 14 complete validation tasks")
    sat = defaultdict(list)
    for row in rows:
        if row["benchmark"] == "SAT":
            sat[row["base_id"]].append(row)
    for pair in sat.values():
        pair.sort(key=lambda r: r["option_order"])
        if len(pair) != 2 or [r["option_order"] for r in pair] != [0, 1]:
            raise ValueError("SAT requires original and reversed options for each question")
        if pair[0]["options"] != pair[1]["options"][::-1]:
            raise ValueError("SAT options were not reversed")
        answers = [r["options"][ord(r["answer"]) - 65] for r in pair]
        if answers[0] != answers[1]:
            raise ValueError("SAT reversed target does not refer to the same answer")


def load_suite(root=DEFAULT_SUITE):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "accepted":
        raise ValueError("Unified benchmark preparation/acceptance is incomplete")
    for relative, sha in manifest["files"].items():
        if file_sha(safe_path(root, relative)) != sha:
            raise ValueError(f"Frozen benchmark changed: {relative}")
    protocol = json.loads((root / "protocol.json").read_text())
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    check_counts(rows, protocol)
    return rows, protocol, manifest


def sat_reference_correct(row, prediction):
    """Reproduce the author's target-dependent fallback for *diagnostics only*.

    This reference can credit an ambiguous response containing both A and B.
    It must never be used to extract an answer or as the primary paper metric.
    """
    target = row["answer"]
    match = re.search(r"answer is \s*\n?\s*\(?([A-Z])\)?", prediction)
    parsed = match[1] if match else ""
    if parsed != target:
        parsed = target if target in prediction else prediction.strip()
    return parsed.lower().strip() == target.lower().strip()


def score_record(row, prediction, *, truncated=False, mathvista_policy=DEFAULT_MATHVISTA_POLICY,
                 wrapper_policy=DEFAULT_WRAPPER_POLICY, answer_policy=DEFAULT_ANSWER_POLICY):
    from lvr.eval.reasoning_scoring import score
    if wrapper_policy not in WRAPPER_POLICIES:
        raise ValueError(f"Unknown wrapper policy: {wrapper_policy}")
    scored_prediction, transform = prediction, "unchanged"
    if wrapper_policy == "tag_repair_v1":
        scored_prediction, transform = repair_answer_wrapper(
            prediction, mira=row["scoring"]["kind"] in ("mira", "mira_pair"), truncated=truncated)
    result = score(row, scored_prediction, answer_policy=answer_policy, mira_policy="strict", truncated=truncated,
                   mathvista_policy=mathvista_policy)
    if wrapper_policy != "strict":
        result.update(wrapper_policy=wrapper_policy, wrapper_transform=transform)
        if row["scoring"]["kind"] in ("mira", "mira_pair"):
            result["metric"] = "mira_em_with_tag_repair_v1"
    if row["benchmark"] == "SAT":
        result["metric"] = ("strict_choice_option_order_mean" if answer_policy == "strict"
                            else answer_policy + "_choice_option_order_mean")
        result["reference_correct"] = sat_reference_correct(row, prediction)
    return result


def summarize(rows, results, protocol, *, mathvista_policy=DEFAULT_MATHVISTA_POLICY,
              wrapper_policy=DEFAULT_WRAPPER_POLICY, answer_policy=DEFAULT_ANSWER_POLICY):
    """Require exact complete coverage before creating a paper main table."""
    check_counts(rows, protocol)
    ids = [r["id"] for r in results]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in rows}:
        raise ValueError("Missing, duplicate or foreign predictions; cannot publish a complete summary")
    lookup = {r["id"]: r for r in rows}
    groups = defaultdict(list)
    for result in results:
        groups[lookup[result["id"]]["benchmark"]].append(result)
    metrics = []
    for name, spec in protocol["benchmarks"].items():
        selected = groups[name]
        tasks = defaultdict(list)
        cycles = defaultdict(list)
        for result in selected:
            row = lookup[result["id"]]
            tasks[row["task"]].append(result)
            cycles[str(row.get("metadata", {}).get("cycle_category", "all"))].append(result)
        mean = lambda items: sum(bool(r["correct"]) for r in items) / len(items)
        micro = mean(selected)
        if spec["aggregation"] == "task_mean":
            primary = sum(mean(items) for items in tasks.values()) / len(tasks)
        elif spec["aggregation"] == "cycle_mean":
            if len(cycles) != 4:
                raise ValueError(f"{name} requires four official cycles")
            primary = sum(mean(items) for items in cycles.values()) / len(cycles)
        else:
            primary = micro
        entry = {"benchmark": name, "records": spec["records"], "generations": len(selected),
                 "aggregation": spec["aggregation"], "accuracy": primary, "sample_accuracy": micro,
                 "parse_failures": sum(not r["parsed"] for r in selected),
                 "token_limit_reached": sum(r.get("reached_token_limit", False) for r in selected),
                 "tasks": {k: {"n": len(v), "accuracy": mean(v)} for k, v in sorted(tasks.items())}}
        if name == "SAT":
            entry["author_reference_accuracy"] = sum(r["reference_correct"] for r in selected) / len(selected)
            entry["reference_disagreements"] = sum(r["correct"] != r["reference_correct"] for r in selected)
        if name.startswith("HRBench"):
            entry["cycles"] = {k: {"n": len(v), "accuracy": mean(v)} for k, v in sorted(cycles.items())}
        metrics.append(entry)
    return {"status": "complete", "records": protocol["total_records"], "generations": len(results),
            "metrics": metrics, "macro_accuracy": sum(r["accuracy"] for r in metrics) / len(metrics),
            "overlap_policy": protocol["reporting"]["overlap_policy"],
            "mathvista_metric": MATHVISTA_METRICS[mathvista_policy],
            "mathvista_policy": mathvista_policy,
            "answer_parsing": answer_parsing_metadata(wrapper_policy=wrapper_policy, mathvista_policy=mathvista_policy,
                                                      answer_policy=answer_policy),
            "test_benchmarks_used_for_selection": False}
