from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from benchmark.data import (
        ViewerDataError,
        available_results,
        join_results,
        load_benchmark_records,
        load_results,
    )
except ModuleNotFoundError:
    from data import (  # type: ignore[no-redef]
        ViewerDataError,
        available_results,
        join_results,
        load_benchmark_records,
        load_results,
    )


OLD_REPO = Path("/private/wmz/project/lvr_pipeline")
DEFAULT_RESULT_DIR = OLD_REPO / (
    "runs/plain_sft/plain_v1_p99_len4096/"
    "eval_hr_mme_realworld_nocap_step7000/plain_v1_p99_len4096_step7000"
)
DEFAULT_BENCHMARK_ROOT = OLD_REPO / "external/bench"


@st.cache_data(show_spinner="Loading local benchmark metadata...")
def load_joined_records(
    result_dir: str,
    benchmark_root: str,
    benchmark: str,
) -> list[dict[str, Any]]:
    results = load_results(result_dir, benchmark)
    source_records = load_benchmark_records(benchmark_root, benchmark)
    return join_results(results, source_records, benchmark)


def main() -> None:
    st.set_page_config(page_title="LVR Benchmark Viewer", page_icon="📊", layout="wide")
    st.title("LVR Benchmark Viewer")
    st.caption("Inspect local evaluation results and trace each prediction back to its source sample.")

    with st.sidebar:
        st.header("Data")
        result_dir = st.text_input("Result directory", value=str(DEFAULT_RESULT_DIR))
        benchmark_root = st.text_input(
            "Benchmark root",
            value=str(DEFAULT_BENCHMARK_ROOT),
            help="plain.yaml uses external/bench, which may be a symbolic link.",
        )

    available = available_results(result_dir)
    if not available:
        st.error(f"No supported result JSON files found in: {result_dir}")
        return

    with st.sidebar:
        benchmark = st.selectbox("Benchmark", available)

    try:
        records = load_joined_records(result_dir, benchmark_root, benchmark)
    except ViewerDataError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    if not records:
        st.info(f"No records found for {benchmark}.")
        return

    categories = sorted({str(record.get("category", "")) for record in records})
    with st.sidebar:
        st.header("Filters")
        correctness = st.radio(
            "Correctness",
            ("Errors only", "All", "Correct only"),
            horizontal=False,
        )
        selected_categories = st.multiselect("Category", categories, default=categories)
        search = st.text_input("Search question or ID").strip().casefold()

    filtered = _filter_records(records, correctness, selected_categories, search)
    correct_count = sum(bool(record.get("correct")) for record in records)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Benchmark", benchmark)
    metric_columns[1].metric("Accuracy", f"{correct_count / len(records) * 100:.1f}%")
    metric_columns[2].metric("Correct / Total", f"{correct_count} / {len(records)}")
    metric_columns[3].metric("Filtered", len(filtered))

    if not filtered:
        st.info("No records match the selected filters.")
        return

    navigation_key = (
        f"record-{benchmark}-{correctness}-"
        f"{','.join(selected_categories)}-{search}"
    )
    position = st.number_input(
        "Record",
        min_value=1,
        max_value=len(filtered),
        value=1,
        step=1,
        key=navigation_key,
        help="Use the step buttons or type a record number.",
    )
    record = filtered[int(position) - 1]

    status = "CORRECT" if record["correct"] else "ERROR"
    st.caption(
        f"{int(position)} / {len(filtered)} filtered · "
        f"ID {record['id']} · {record.get('category', '')} · {status}"
    )

    image_column, detail_column = st.columns((1.15, 1), gap="large")
    with image_column:
        st.subheader("Image")
        image_path = Path(str(record.get("image_path", "")))
        if image_path.is_file():
            st.image(str(image_path), width="stretch")
            st.caption(str(image_path))
        else:
            st.error(f"Image does not exist: {image_path}")

    with detail_column:
        st.subheader("Question")
        st.write(record.get("question", ""))

        options = record.get("options") or []
        if options:
            st.markdown("#### Options")
            for option in options:
                label = str(option.get("label", ""))
                markers = []
                if label == record.get("label"):
                    markers.append("GT")
                if label == record.get("predicted"):
                    markers.append("prediction")
                suffix = f"  **← {', '.join(markers)}**" if markers else ""
                st.write(f"({label}) {option.get('text', '')}{suffix}")

        answer_columns = st.columns(2)
        answer_columns[0].markdown("#### Ground truth")
        answer_columns[0].success(_answer_text(record, "label"))
        answer_columns[1].markdown("#### Model answer")
        if record.get("correct"):
            answer_columns[1].success(_answer_text(record, "predicted"))
        else:
            answer_columns[1].error(_answer_text(record, "predicted"))

        st.markdown("#### Raw model output")
        st.code(record.get("prediction", "") or "<empty>", language=None)


def _filter_records(
    records: list[dict[str, Any]],
    correctness: str,
    selected_categories: list[str],
    search: str,
) -> list[dict[str, Any]]:
    selected = set(selected_categories)
    filtered = []
    for record in records:
        if str(record.get("category", "")) not in selected:
            continue
        if correctness == "Errors only" and bool(record.get("correct")):
            continue
        if correctness == "Correct only" and not bool(record.get("correct")):
            continue
        if search:
            haystack = f"{record.get('id', '')}\n{record.get('question', '')}".casefold()
            if search not in haystack:
                continue
        filtered.append(record)
    return filtered


def _answer_text(record: dict[str, Any], field: str) -> str:
    label = str(record.get(field, "")).strip() or "<empty>"
    for option in record.get("options") or []:
        if str(option.get("label", "")) == label:
            return f"{label} — {option.get('text', '')}"
    return label


if __name__ == "__main__":
    main()
