"""Live progress dashboard for resumable align-generation JSONL shards.

Usage:
  streamlit run data/scripts/visible/align_progress_app.py -- --output-dir data/align/v0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import streamlit as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-total", type=int, default=122_896)
    parser.add_argument("--test-total", type=int, default=1_242)
    parser.add_argument("--num-shards", type=int, default=8)
    args, _ = parser.parse_known_args()
    return args


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def expected_rows(total: int, num_shards: int, shard: int) -> int:
    return (total + num_shards - 1 - shard) // num_shards


def split_status(output_dir: Path, split: str, total: int, num_shards: int) -> tuple[int, list[dict[str, int]]]:
    rows = []
    for shard in range(num_shards):
        completed = line_count(output_dir / f"{split}.part-{shard:02d}.jsonl")
        expected = expected_rows(total, num_shards, shard)
        rows.append({"GPU shard": shard, "completed": completed, "total": expected, "percent": round(completed / expected * 100, 2)})
    return sum(row["completed"] for row in rows), rows


def render(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    st.title("Align generation progress")
    st.caption(f"Refreshing every 10 seconds · {output_dir}")
    for split, total in (("train", args.train_total), ("test", args.test_total)):
        completed, rows = split_status(output_dir, split, total, args.num_shards)
        st.subheader(split)
        st.progress(min(completed / total, 1.0), text=f"{completed:,} / {total:,} ({completed / total:.2%})")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        final = output_dir / f"{split}.json"
        if final.is_file():
            st.success(f"Merged output ready: {final.name}")
        elif completed:
            newest = max((output_dir / f"{split}.part-{i:02d}.jsonl" for i in range(args.num_shards)), key=lambda p: p.stat().st_mtime if p.exists() else 0)
            st.caption(f"Last shard write: {time.strftime('%F %T', time.localtime(newest.stat().st_mtime)) if newest.exists() else '—'}")


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Align progress", layout="wide")

    @st.fragment(run_every="10s")
    def live() -> None:
        render(args)

    live()


if __name__ == "__main__":
    main()
