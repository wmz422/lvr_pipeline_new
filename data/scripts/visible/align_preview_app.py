"""Inspect an align-generation preview, including images, crops, and raw JSON.

Usage:
  streamlit run data/scripts/visible/align_preview_app.py -- --data data/align/v0/qwen4b_preview100/test.first100.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def data_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path, required=True)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.data.resolve()


@st.cache_data(show_spinner=False)
def load_records(path_text: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Preview JSON must contain a list.")
    return data


def image(path_value: str, title: str) -> None:
    path = ROOT / path_value
    st.subheader(title)
    if path.is_file():
        st.image(str(path), use_container_width=True)
        st.caption(path_value)
    else:
        st.error(f"Missing: {path_value}")


def main() -> None:
    st.set_page_config(page_title="Align preview", layout="wide")
    path = data_path()
    st.title("Align data preview")
    st.caption(str(path))
    if not path.is_file():
        st.error(f"Preview JSON not found: {path}")
        return
    records = load_records(str(path))
    if not records:
        st.warning("No records.")
        return

    with st.sidebar:
        datasets = sorted({str((item.get("raw") or {}).get("dataset", "unknown")) for item in records})
        selected = st.multiselect("Dataset", datasets, default=datasets)
        filtered = [item for item in records if str((item.get("raw") or {}).get("dataset", "unknown")) in selected]
        position = st.number_input("Record", 0, max(0, len(filtered) - 1), 0, 1)
    if not filtered:
        st.warning("No records match the selected dataset.")
        return
    record = filtered[int(position)]
    st.caption(f"Record {int(position) + 1}/{len(filtered)} · source index {record.get('source_index')}")

    original, auxiliary = st.columns(2)
    with original:
        image(str(record.get("question_image", "")), "Original image")
    with auxiliary:
        image(str(record.get("auxiliary_image", "")), "Auxiliary image")

    st.subheader("Qwen observation")
    st.write(record.get("observation", ""))
    st.subheader("Alignment question")
    st.write(record.get("question", ""))

    crops = record.get("crop_images") or ([record["crop_image"]] if record.get("crop_image") else [])
    if crops:
        st.subheader(f"Crops ({len(crops)})")
        columns = st.columns(min(4, len(crops)))
        for index, crop in enumerate(crops):
            with columns[index % len(columns)]:
                image(str(crop), f"Crop {index + 1}")

    st.subheader("Complete JSON record")
    st.json(record)


if __name__ == "__main__":
    main()
