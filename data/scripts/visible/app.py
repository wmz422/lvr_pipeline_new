from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("llm", "lam", "align")
SPLITS = ("train", "test")


def version_key(version: str) -> tuple[int, str]:
    if version.startswith("v") and version[1:].isdigit():
        return int(version[1:]), version
    return -1, version


def available_stages() -> list[str]:
    stages = [stage for stage in STAGES if (ROOT / stage).is_dir()]
    return stages or ["llm"]


def available_versions(stage: str) -> list[str]:
    stage_root = ROOT / stage
    versions = sorted(
        [path.name for path in stage_root.iterdir() if path.is_dir() and path.name.startswith("v")],
        key=version_key,
    )
    return versions


def load_records(stage: str, version: str, split: str) -> list[dict[str, Any]]:
    path = ROOT / stage / version / f"{split}.json"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def raw_dataset(record: dict[str, Any]) -> str:
    raw = record.get("raw", {})
    if isinstance(raw, dict):
        return str(raw.get("dataset", "unknown"))
    return "unknown"


def main() -> None:
    st.set_page_config(page_title="LVR Data Viewer", layout="wide")
    st.title("LVR Data Viewer")

    with st.sidebar:
        stage = st.selectbox("Stage", available_stages())
        versions = available_versions(stage)
        if not versions:
            st.info(f"No versions found for {stage}.")
            return
        default_version = len(versions) - 1
        version = st.selectbox("Version", versions, index=default_version)
        split = st.radio("Split", SPLITS, horizontal=True)

    records = load_records(stage, version, split)
    if not records:
        st.info(f"No records found for {stage}/{version}/{split}.")
        return

    datasets = sorted({raw_dataset(record) for record in records})
    with st.sidebar:
        selected_datasets = st.multiselect("Raw dataset", datasets, default=datasets)

    filtered = [record for record in records if raw_dataset(record) in selected_datasets]
    if not filtered:
        st.info("No records match the selected filters.")
        return

    with st.sidebar:
        position = st.number_input("Record", min_value=0, max_value=len(filtered) - 1, value=0, step=1)

    record = filtered[int(position)]
    st.caption(f"{stage}/{version}/{split} - {int(position) + 1} / {len(filtered)}")

    image_cols = st.columns(2)
    with image_cols[0]:
        st.subheader("Original")
        question_image = ROOT / record["question_image"]
        if question_image.exists():
            st.image(str(question_image), use_container_width=True)
        else:
            st.error(f"Missing image: {question_image}")

    with image_cols[1]:
        st.subheader("Auxiliary")
        auxiliary_image = ROOT / record["auxiliary_image"]
        if auxiliary_image.exists():
            st.image(str(auxiliary_image), use_container_width=True)
        else:
            st.error(f"Missing image: {auxiliary_image}")

    text_cols = st.columns(2)
    with text_cols[0]:
        st.subheader("Question")
        st.write(record.get("question", ""))
    with text_cols[1]:
        st.subheader("Answer")
        st.write(record.get("answer", ""))

    st.subheader("Record")
    st.json(record)


if __name__ == "__main__":
    main()
