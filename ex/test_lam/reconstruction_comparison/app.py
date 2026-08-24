#!/usr/bin/env python3
"""Streamlit viewer for real-vs-reconstructed feature comparison results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = REPO_ROOT / "data"


def available_results() -> list[Path]:
    return sorted(EXPERIMENT_ROOT.glob("results_*.jsonl"))


@st.cache_data(show_spinner=False)
def load_results(path_string: str, modified_ns: int) -> list[dict[str, Any]]:
    del modified_ns  # Included in the cache key so a rewritten JSONL is reloaded.
    path = Path(path_string)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_image(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else IMAGE_ROOT / path


def metric_mean(rows: list[dict[str, Any]], group: str, metric: str) -> float:
    values = [float(row[group][metric]) for row in rows]
    return sum(values) / len(values)


def show_image(title: str, path_value: str) -> None:
    st.subheader(title)
    path = resolve_image(path_value)
    if path.is_file():
        st.image(str(path), width="stretch")
        st.caption(path_value)
    else:
        st.error(f"Missing image: {path}")


def main() -> None:
    st.set_page_config(page_title="LAM Reconstruction Comparison", layout="wide")
    st.title("LAM Reconstruction Comparison")

    result_paths = available_results()
    if not result_paths:
        st.error(f"No results_*.jsonl files found in {EXPERIMENT_ROOT}")
        return

    with st.sidebar:
        selected_name = st.selectbox("Results", [path.name for path in result_paths])
    selected_path = EXPERIMENT_ROOT / selected_name
    rows = load_results(str(selected_path), selected_path.stat().st_mtime_ns)
    if not rows:
        st.warning(f"No cases found in {selected_path}")
        return

    with st.sidebar:
        st.metric("Cases", len(rows))
        st.metric(
            "Mean pre-merge cosine",
            f"{metric_mean(rows, 'premerge_metrics', 'cosine_similarity'):.4f}",
        )
        st.metric(
            "Mean merged cosine",
            f"{metric_mean(rows, 'merged_metrics', 'cosine_similarity'):.4f}",
        )
        position = st.number_input(
            "Case position",
            min_value=1,
            max_value=len(rows),
            value=1,
            step=1,
        )

    row = rows[int(position) - 1]
    st.caption(
        f"Case {int(position)} / {len(rows)} · dataset index {row.get('index', 'unknown')} · "
        f"checkpoint {Path(str(row.get('checkpoint', ''))).name}"
    )

    image_columns = st.columns(2)
    with image_columns[0]:
        show_image("Original question image", str(row.get("question_image", "")))
    with image_columns[1]:
        show_image("Auxiliary image", str(row.get("auxiliary_image", "")))

    st.subheader("Prompt")
    st.info(str(row.get("prompt", "")))

    pre = row.get("premerge_metrics", {})
    merged = row.get("merged_metrics", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("Pre-merge MSE", f"{float(pre.get('mse', 0.0)):.5f}")
    metric_columns[1].metric(
        "Pre-merge cosine", f"{float(pre.get('cosine_similarity', 0.0)):.5f}"
    )
    metric_columns[2].metric("Merged MSE", f"{float(merged.get('mse', 0.0)):.5f}")
    metric_columns[3].metric(
        "Merged cosine", f"{float(merged.get('cosine_similarity', 0.0)):.5f}"
    )

    answer_columns = st.columns(2)
    with answer_columns[0]:
        st.subheader("Answer from real auxiliary feature")
        st.write(str(row.get("real_feature_answer", "")))
    with answer_columns[1]:
        st.subheader("Answer from reconstructed auxiliary feature")
        st.write(str(row.get("reconstructed_feature_answer", "")))

    with st.expander("Latent statistics and raw record"):
        st.json(
            {
                "latent_statistics": row.get("latent_statistics", {}),
                "record": row,
            }
        )


if __name__ == "__main__":
    main()
