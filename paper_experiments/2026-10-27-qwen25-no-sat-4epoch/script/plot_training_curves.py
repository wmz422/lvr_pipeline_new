"""Plot a snapshot of the complete no-SAT run, including its initial 696 steps."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
INITIAL = EXPERIMENT.parent / "2026-10-26-qwen25-no-sat-ablation"
OUTPUT = EXPERIMENT / "results/training_curves"
STEPS_PER_EPOCH = 1346
WINDOW = 10


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    snapshot = OUTPUT / "snapshots" / now.strftime("%Y%m%dT%H%M%SZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = [INITIAL / "runs/main/plain_sft/metrics/csv/metrics.csv"]
    sources += sorted((EXPERIMENT / "runs").glob("epoch_*/plain_sft/metrics/*/metrics.csv"))
    training, validation, provenance = {}, {}, []
    for index, source in enumerate(sources):
        data = source.read_bytes()
        # Active CSV journals may be captured during an append. Keep complete rows.
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
        saved = snapshot / f"{index:02d}_metrics.csv"
        saved.write_bytes(data)
        provenance.append({"path": str(source), "snapshot": str(saved),
                           "sha256": hashlib.sha256(data).hexdigest()})
        for row in csv.DictReader(io.StringIO(data.decode())):
            if row.get("train/loss_step"):
                step = int(row["step"]) + 1
                loss = float(row["train/loss_step"])
                assert math.isfinite(loss) and loss >= 0
                assert math.isclose(loss, float(row["train/ce_step"]), abs_tol=1e-12)
                assert step not in training or training[step]["loss"] == loss, step
                training[step] = {"step": step, "epoch": int(row["epoch"]) + 1,
                                  "loss": loss, "source_index": index}
            if row.get("val/ce"):
                epoch = int(row["epoch"]) + 1
                # End-of-epoch validation logs use either target-1 or target;
                # anchor to the known completed training epoch, not CSV step+1.
                step = epoch * STEPS_PER_EPOCH
                assert int(row["step"]) in (step - 1, step)
                loss = float(row["val/ce"])
                assert math.isfinite(loss) and loss >= 0
                assert epoch not in validation or validation[epoch]["loss"] == loss
                validation[epoch] = {"epoch": epoch, "step": step, "loss": loss,
                                     "csv_step": int(row["step"]), "source_index": index}

    points = [training[k] for k in sorted(training)]
    vals = [validation[k] for k in sorted(validation)]
    assert points and vals
    assert [p["step"] for p in points] == list(range(10, points[-1]["step"] + 1, 10))
    for i, point in enumerate(points):
        point["trailing_10_logged_points"] = (
            sum(p["loss"] for p in points[i-WINDOW+1:i+1]) / WINDOW
            if i >= WINDOW - 1 else None)
    for filename, rows in [("training_metrics.csv", points), ("validation_metrics.csv", vals)]:
        with (OUTPUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#25364b", "xtick.color": "#536174",
                         "ytick.color": "#536174", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, (ax, av) = plt.subplots(1, 2, figsize=(14, 6.4),
                                gridspec_kw={"width_ratios": [2.4, 1]})
    fig.subplots_adjust(left=.065, right=.965, top=.70, bottom=.24, wspace=.26)
    fig.text(.065, .94, "Plain SFT without SAT | Training progress", fontsize=22,
             fontweight="bold", color="#172b45")
    fig.text(.065, .89, "Qwen2.5-VL-7B-Instruct  |  172,224 training examples  |  Global batch 128",
             fontsize=11, color="#536174")
    fig.text(.065, .845,
             f"Snapshot: {now.strftime('%Y-%m-%d %H:%M UTC')}  |  Latest logged update: {points[-1]['step']:,}"
             f"  ({points[-1]['step'] / STEPS_PER_EPOCH:.2f} epochs)",
             fontsize=10, color="#536174")

    ax.plot([p["step"] for p in points], [p["loss"] for p in points],
            color="#81abc9", alpha=.42, lw=.85, label="Logged batch CE")
    smooth = [p for p in points if p["trailing_10_logged_points"] is not None]
    ax.plot([p["step"] for p in smooth], [p["trailing_10_logged_points"] for p in smooth],
            color="#126a94", lw=2.3, label="Trailing mean: 10 logged points")
    ymax = max(p["loss"] for p in points) * 1.12
    for value in vals:
        ax.axvline(value["step"], color="#a8b2bf", ls="--", lw=1)
        ax.text(value["step"] - 30, ymax * .96, f"Epoch {value['epoch']}",
                ha="right", va="top", color="#66758a", fontsize=9)
    completed = vals[-1]["step"]
    if points[-1]["step"] > completed:
        ax.axvspan(completed, points[-1]["step"], color="#e7eff7", alpha=.65, zorder=0)
    ax.set(xlim=(0, points[-1]["step"] * 1.025), ylim=(0, ymax),
           xlabel="Completed optimizer updates", ylabel="Training cross-entropy")
    top = ax.secondary_xaxis("top", functions=(lambda x: x / STEPS_PER_EPOCH,
                                               lambda x: x * STEPS_PER_EPOCH))
    top.set_xlabel("Training epochs", labelpad=7)
    top.set_xticks([v / 2 for v in range(int(points[-1]["step"] / STEPS_PER_EPOCH * 2) + 1)])
    ax.legend(loc="upper right", bbox_to_anchor=(.98, .82), frameon=False, fontsize=9)

    av.plot([v["epoch"] for v in vals], [v["loss"] for v in vals],
            color="#c36a21", marker="o", markersize=7, lw=2.4)
    for v in vals:
        av.annotate(f"{v['loss']:.4f}", (v["epoch"], v["loss"]),
                    xytext=(0, 13), textcoords="offset points", ha="center",
                    color="#a64e10", fontweight="bold", fontsize=12)
    av.set(xlim=(vals[0]["epoch"]-.35, vals[-1]["epoch"]+.35),
           ylim=(0, max(v["loss"] for v in vals) * 1.28),
           xlabel="Completed epoch", ylabel="Validation cross-entropy")
    av.set_xticks([v["epoch"] for v in vals])
    av.set_title("End-of-epoch validation", loc="left", pad=16,
                 fontsize=12, color="#25364b")
    for axis in (ax, av):
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e6eaf0", lw=.8)
        for side in ("bottom", "left"):
            axis.spines[side].set_color("#d4dbe4")
    fig.text(.065, .145,
             "Training logs joined across the initial 696-step run and all continuation stages; dashed lines mark completed epochs.\n"
             "Loss is logged every 10 updates. Smoothing averages 10 logged values (~100 updates), not every batch.\n"
             "Validation: original 7,571 examples, including 537 SAT examples. The active epoch has no validation point yet.",
             fontsize=9, color="#657285", linespacing=1.65, va="top")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"training_curves.{suffix}", dpi=180)
    plt.close(fig)

    audit = {"status": "complete", "snapshot_utc": now.isoformat(), "source_logs": provenance,
             "train_metric": "train/loss_step", "validation_metric": "val/ce",
             "train_step_mapping": "CSV step + 1", "validation_step_mapping": "(CSV epoch + 1) * 1346",
             "smoothing": "Trailing arithmetic mean of 10 logged values, full windows only",
             "logged_points": len(points), "first_logged_update": points[0]["step"],
             "last_logged_update": points[-1]["step"], "latest_smoothed_loss": smooth[-1]["trailing_10_logged_points"],
             "validation": vals, "active_logs_read_from_frozen_snapshots": True,
             "limitations": "Batch CE and validation CE have different aggregation; logged batch values are not full-epoch means."}
    (OUTPUT / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_logs"}, indent=2))
    print(OUTPUT / "training_curves.png")


if __name__ == "__main__":
    main()
