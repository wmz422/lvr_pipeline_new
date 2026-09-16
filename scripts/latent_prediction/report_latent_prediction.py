#!/usr/bin/env python3
"""Audit matched prediction ablations, compare SFT, and plot completed training curves."""

import argparse
import ast
import csv
import json
import math
from pathlib import Path


def table(path, delimiter=","):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sft", type=Path, help="Optional fresh SFT benchmark summary")
    parser.add_argument("--source", type=Path, help="Optional fresh source-teacher benchmark summary")
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument("--protocol", type=Path, default=repository / "benchmark/latent_prediction")
    args = parser.parse_args()
    root = args.root
    expected = json.loads((args.protocol / "datasets.json").read_text())
    annotations = json.loads((args.protocol / "crop_annotations.json").read_text())["samples"]
    summaries = {mode: root / f"benchmark/summary_{mode}.json" for mode in
                 ("predicted", "teacher", "zero", "shuffled")}
    summaries.update({key: path for key, path in {"plain_sft": args.sft, "source_teacher": args.source}.items() if path})
    results, identities = {}, {}
    for mode, path in summaries.items():
        assert (path.parent / "exit_code").read_text().strip() == "0", path
        summary = json.loads(path.read_text())
        assert set(summary["results"]) == set(expected)
        results[mode] = summary["results"]
        identities[mode] = {}
        for dataset, row in results[mode].items():
            assert row["samples"] == expected[dataset]["samples"]
            assert row["generation_failures"] == 0, (mode, dataset, row)
            assert row["empty_predictions"] == 0, (mode, dataset, row)
            assert row["truncated_predictions"] in (None, 0), (mode, dataset, row)
            records = table(row["prediction_file"], "\t")
            ids = {}
            for record in records:
                extra = ast.literal_eval(record["extra_records"])
                key = str(record["index"])
                ids[key] = (extra["sample_id"], extra["qa_prompt"], record["answer"])
                if mode in ("teacher", "predicted", "zero", "shuffled"):
                    annotation = annotations[extra["sample_id"]]
                    assert annotation["benchmark"] == dataset and str(annotation["index"]) == key
                    assert annotation["qa_prompt"] == extra["qa_prompt"]
                    assert extra["model_kind"] == "latent_prediction"
                    assert extra["checkpoint_step"] == 961 and extra["model_num_latents"] == 8
                    assert extra["latent_mode"] == mode
                    assert ("encoder" in extra["input_tokens"]) == (mode == "teacher")
                    if mode == "shuffled":
                        assert extra["shuffled_source_id"] != extra["sample_id"]
            assert len(ids) == len(records) == expected[dataset]["samples"]
            identities[mode][dataset] = ids
            if mode == "shuffled":
                own = {value[0] for value in ids.values()}
                donors = [ast.literal_eval(r["extra_records"])["shuffled_source_id"] for r in records]
                assert len(set(donors)) == len(donors) and set(donors) == own
    for mode in identities:
        assert identities[mode] == identities["predicted"], f"Different test identities/prompts/answers: {mode}"
    reference = json.loads((repository / "results/latent_prediction_20260916/metrics.json").read_text())
    for mode in ("plain_sft", "source_teacher"):
        results.setdefault(mode, reference["results"][mode])

    training, curves = {}, {}
    for phase, steps in (("warmup", 200), ("joint", 961)):
        assert (root / phase / "train_exit_code").read_text().strip() == "0"
        world_size = json.loads((root / phase / "runtime_rank0.json").read_text())["world_size"]
        ranks = [json.loads((root / phase / f"runtime_rank{rank}.json").read_text()) for rank in range(world_size)]
        for run in ranks:
            assert run["status"] == "training_complete" and run["optimizer_steps"] == steps
            assert run["encoder_trainable_parameters"] == 0
            assert run["world_size"] * run["batch_size_per_gpu"] * run["accumulate_grad_batches"] == 128
        rows = table(root / phase / "metrics/csv/metrics.csv")
        for row in rows:
            for name, value in row.items():
                if value:
                    assert math.isfinite(float(value)), (phase, name, value)
        validation = [{key: float(value) for key, value in row.items() if value and key.startswith("val/")}
                      for row in rows if row.get("val/latent_loss")]
        assert validation, f"Missing full validation for {phase}"
        training[phase] = {"steps": steps, "seconds": ranks[0]["elapsed_seconds"],
                           "peak_allocated_gib": max(r["peak_allocated_gib"] for r in ranks),
                           "peak_reserved_gib": max(r["peak_reserved_gib"] for r in ranks),
                           "validation_optimizer_step": int(next(r["step"] for r in reversed(rows)
                                                                  if r.get("val/latent_loss"))) + 1,
                           "validation": validation[-1]}
        curves[phase] = [row for row in rows if row.get("train/latent_loss_step")]

    total = sum(x["samples"] for x in expected.values())
    aggregate = {mode: {"correct": sum(row["correct_questions"] for row in rows.values()),
                         "samples": total} for mode, rows in results.items()}
    for value in aggregate.values():
        value["accuracy_percent"] = 100 * value["correct"] / total
    report = {"audit": "passed", "total_questions": total,
              "training": training, "results": results, "aggregate_by_question": aggregate}
    (root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Latent prediction results", "", "All conditions use identical question IDs, prompts and reference answers.", "",
             "Historical SFT/source baselines are the recorded reference results unless explicit summaries are supplied.", "",
             "| Benchmark | N | Predicted | Teacher (final B) | Zero | Shuffled | Plain SFT | Source A+B |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, metadata in expected.items():
        scores = [100 * results[mode][dataset]["correct_questions"] / metadata["samples"] for mode in
                  ("predicted", "teacher", "zero", "shuffled", "plain_sft", "source_teacher")]
        lines.append(f"| {dataset} | {metadata['samples']} | " + " | ".join(f"{s:.2f}" for s in scores) + " |")
    scores = [aggregate[mode]["accuracy_percent"] for mode in
              ("predicted", "teacher", "zero", "shuffled", "plain_sft", "source_teacher")]
    lines.append(f"| All questions (sample weighted) | {total} | " +
                 " | ".join(f"{s:.2f}" for s in scores) + " |")
    lines.extend(["", "Scores are percentages; teacher uses the same final student B as predicted/zero/shuffled.",
                  "Source A+B is the preceding original+all-crops checkpoint.", "", "![Training curves](training_curves.png)", ""])
    (root / "report.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    combined = [(int(r["step"]) + 1 + (200 if phase == "joint" else 0), r)
                for phase in ("warmup", "joint") for r in curves[phase]]
    for ax, names, title in ((axes[0, 0], ["latent_loss", "relative_mse"], "Latent alignment"),
                             (axes[0, 1], ["cosine"], "Latent cosine similarity"),
                             (axes[1, 1], ["pred_rms", "teacher_rms"], "Latent magnitude")):
        for name in names:
            points = [(x, float(r[f"train/{name}_step"])) for x, r in combined if r.get(f"train/{name}_step")]
            ax.plot(*zip(*points), label=name)
        ax.axvline(200, color="gray", linestyle="--", label="joint starts")
        ax.set(title=title, xlabel="Cumulative optimizer step")
    ax = axes[1, 0]
    for name in ("ce", "description_ce"):
        points = [(int(r["step"]) + 1, float(r[f"train/{name}_step"])) for r in curves["joint"]
                  if r.get(f"train/{name}_step")]
        ax.plot(*zip(*points), label=name)
    ax.axvline(200, color="gray", linestyle=":", label="50% predicted QA")
    ax.axvline(500, color="gray", linestyle="--", label="100% predicted QA")
    ax.set(title="Joint QA and description CE", xlabel="Joint optimizer step")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root / "training_curves.png", dpi=160)
    fig.savefig(root / "training_curves.svg")
    plt.close(fig)
    print("\n".join(lines[:-3]))


if __name__ == "__main__":
    main()
