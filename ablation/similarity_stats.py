#!/usr/bin/env python3
"""Two first-eight-position cross-case cosine tables/figures on a shared cohort."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import COUNTS, digest, load_data, sha256, write_json
from similarity_cache import read_trace, trace_path

METHODS = ("monet", "lvr", "ours")
KINDS = ("teacher", "predicted")
LABELS = {"monet": "Monet-7B", "lvr": "LVR-7B", "ours": "Ours Qwen3-VL-4B (4-stage)"}


def independent_groups(rows):
    """Exclude variants of one problem and records with the same original image set."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in (("group", row["group_id"]),
                    ("images", tuple(sorted(im["sha256"] for im in row["images"])))):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return [find(i) for i in range(len(rows))]


def pair_sums(unit, groups):
    """Return the unordered eligible-pair dot-product sum and pair count in O(ND)."""
    total = unit.sum(axis=0)
    numerator = np.einsum("sd,sd->s", total, total)
    denominator = len(unit) ** 2
    buckets = {}
    for i, group in enumerate(groups):
        buckets.setdefault(group, []).append(i)
    for indices in buckets.values():
        summed = unit[indices].sum(axis=0)
        numerator -= np.einsum("sd,sd->s", summed, summed)
        denominator -= len(indices) ** 2
    return numerator / 2, denominator // 2


def cosine_curve(values, groups, tasks, scope="within-benchmark"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("Expected finite [cases, 8, hidden] values")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if (norm == 0).any():
        raise ValueError("Zero-norm cosine is undefined")
    unit = values / norm
    total, count = pair_sums(unit, groups)
    if scope != "within-benchmark":
        within = np.zeros(8)
        within_count = 0
        for task in sorted(set(tasks)):
            ids = [i for i, value in enumerate(tasks) if value == task]
            score, n = pair_sums(unit[ids], [groups[i] for i in ids])
            within += score
            within_count += n
        if scope == "within-task":
            total, count = within, within_count
        elif scope == "cross-task":
            total, count = total - within, count - within_count
        else:
            raise ValueError("Unknown pair scope")
    return (total / count if count else np.full(8, np.nan)), count


def load_cohort(output, rows, *, allow_smoke=False):
    manifest_sha = None
    ids = None
    configs = {}
    traces = {}
    coverage = {}
    lookup = {row["id"]: row for row in rows}
    for method in METHODS:
        for kind in KINDS:
            key = f"{method}/{kind}"
            config_path = output / method / kind / "run_config.json"
            config = json.loads(config_path.read_text())
            if (config["schema"], config["method"], config["kind"], config["retained_positions"]) != (
                    "cross-case-latent-run-v1", method, kind, 8):
                raise ValueError(f"Invalid trace config: {key}")
            if config["scope"] != "full" and not allow_smoke:
                raise ValueError("Smoke traces cannot produce a formal table; use --allow-smoke for a labelled pipeline check")
            if manifest_sha is None:
                manifest_sha, ids = config["suite_manifest_sha256"], config["selected_ids"]
            if config["suite_manifest_sha256"] != manifest_sha or config["selected_ids"] != ids:
                raise ValueError("All six branches must use exactly the same data version and selected IDs")
            if config["scope"] == "full" and set(ids) != set(lookup):
                raise ValueError("Formal extraction must include all 2787 records")
            values = {}
            lengths = {}
            for identity in ids:
                if identity not in lookup:
                    raise ValueError(f"Foreign benchmark ID: {identity}")
                path = trace_path(output, method, kind, identity)
                if not path.exists():
                    raise ValueError(f"Incomplete extraction: {key} {identity}; rerun extraction before aggregation")
                value, _ = read_trace(path, identity, digest(config))
                values[identity] = value.numpy()
                lengths[identity] = len(value)
            configs[key] = {"sha256": sha256(config_path), "config": config}
            traces[key] = values
            coverage[key] = {benchmark: {
                "total": sum(lookup[i]["benchmark"] == benchmark for i in ids),
                "positions": [sum(lookup[i]["benchmark"] == benchmark and lengths[i] >= s for i in ids) for s in range(1, 9)],
                "fewer_than_8_ids": [i for i in ids if lookup[i]["benchmark"] == benchmark and lengths[i] < 8]}
                for benchmark in COUNTS}
    selected = [lookup[i] for i in ids]
    common = [row for row in selected if all(len(values[row["id"]]) == 8 for values in traces.values())]
    return common, traces, coverage, configs


def write_table(path, records):
    columns = ["benchmark", "method", "cases", "independent_groups", "eligible_pairs", *[f"step_{i}" for i in range(1, 9)]]
    with path.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in record.items()})
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in records:
        lines.append("| " + " | ".join(f"{record[k]:.6f}" if k.startswith("step_") else str(record[k]) for k in columns) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def plot_curves(path, records, kind, scope, smoke, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    colors = {"monet": "#d95f02", "lvr": "#7570b3", "ours": "#1b9e77"}
    for axis, benchmark in zip(axes, COUNTS):
        for row in records:
            if row["benchmark"] == benchmark:
                axis.plot(range(1, 9), [row[f"step_{i}"] for i in range(1, 9)], marker="o", markersize=3,
                          label=LABELS[row["method"]], color=colors[row["method"]])
                n, pairs = row["cases"], row["eligible_pairs"]
        axis.set_title(f"{benchmark}\ncommon cases={n}, pairs={pairs:,}", fontsize=10)
        axis.set_xlabel("Latent position (first 8)")
        axis.set_xticks(range(1, 9))
        axis.set_ylim(-1.02, 1.02)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Mean cross-case cosine similarity")
    axes[-1].legend(fontsize=7, loc="lower right")
    prefix = "SYNTHETIC PIPELINE CHECK — " if synthetic else ("SMOKE — " if smoke else "")
    title = f"{prefix}{kind.title()} — {scope} pairs"
    fig.suptitle(title)
    fig.tight_layout()
    note = ("Teacher definitions: ours = encoder slots; Monet = SFT stage2 helper-conditioned slots; LVR = projected helper patches."
            if kind == "teacher" else
            "Positions: Monet/LVR = native consumed recurrent vectors; ours = learned latent slots. See coverage.json for exclusions.")
    fig.text(.5, -.03, note, ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Extraction root holding monet/, lvr/, ours/")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Defaults to OUTPUT/report")
    parser.add_argument("--pair-scope", choices=["within-benchmark", "within-task", "cross-task"], default="within-benchmark")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    rows, _, _, _ = load_data(args.benchmark)
    common, traces, coverage, configs = load_cohort(args.output, rows, allow_smoke=args.allow_smoke)
    expected = sha256(args.benchmark / "manifest.json")
    if any(c["config"]["suite_manifest_sha256"] != expected for c in configs.values()):
        raise ValueError("Report benchmark does not match cached traces")
    smoke = any(c["config"]["scope"] != "full" for c in configs.values())
    synthetic = any(c["config"].get("synthetic", False) for c in configs.values())
    if synthetic and not smoke:
        raise ValueError("Synthetic diagnostics must be labelled smoke")
    report = args.report or args.output / "report"
    report.mkdir(parents=True, exist_ok=True)
    primary = {}
    for scope in ("within-benchmark", "within-task", "cross-task"):
        for kind in KINDS:
            records = []
            for benchmark in COUNTS:
                cohort = [row for row in common if row["benchmark"] == benchmark]
                groups = independent_groups(cohort)
                for method in METHODS:
                    if cohort:
                        values = np.stack([traces[f"{method}/{kind}"][row["id"]] for row in cohort])
                        curve, pairs = cosine_curve(values, groups, [r["task"] for r in cohort], scope)
                    else:
                        curve, pairs = np.full(8, np.nan), 0
                    if not pairs and scope == args.pair_scope and not smoke:
                        raise ValueError(f"No eligible common pairs for {benchmark}; inspect coverage rather than publishing a curve")
                    records.append({"benchmark": benchmark, "method": method, "cases": len(cohort),
                                    "independent_groups": len(set(groups)), "eligible_pairs": pairs,
                                    **{f"step_{i+1}": float(value) for i, value in enumerate(curve)}})
            write_table(report / f"{kind}_{scope}", records)
            if scope == args.pair_scope:
                primary[kind] = records
    for kind in KINDS:
        write_table(report / kind, primary[kind])
        plot_curves(report / kind, primary[kind], kind, args.pair_scope, smoke, synthetic)
    write_json(report / "coverage.json", {"branches": coverage,
        "common_cases": {b: [r["id"] for r in common if r["benchmark"] == b] for b in COUNTS}})
    write_json(report / "report.json", {"schema": "cross-case-cosine-report-v1", "scope": "smoke" if smoke else "full", "synthetic": synthetic,
        "pair_scope": args.pair_scope, "cohort": "intersection of all six branches with eight observed positions",
        "exclusions": "same transitive problem group or identical original-image set",
        "aggregation": "uniform mean over eligible unordered pairs within each benchmark; within-task uses pair weighting",
        "inputs": configs, "common_records": len(common)})
    print(f"Wrote Teacher/Predicted CSV, Markdown, PNG, PDF and coverage to {report}; common cases={len(common)}")


if __name__ == "__main__":
    main()
