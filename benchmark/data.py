from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = (
    "vstar",
    "hr_bench_4k",
    "hr_bench_8k",
    "mme_realworld_lite",
)

_OPTION_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.*?)\s*$")


class ViewerDataError(RuntimeError):
    """Raised when result or benchmark data cannot be loaded safely."""


def normalize_id(value: Any) -> str:
    """Normalize the int/string IDs used by the four benchmark loaders."""
    if value is None:
        return ""
    return str(value).strip()


def available_results(result_dir: str | Path) -> list[str]:
    root = Path(result_dir).expanduser()
    return [name for name in BENCHMARKS if (root / f"{name}.json").is_file()]


def load_results(result_dir: str | Path, benchmark: str) -> list[dict[str, Any]]:
    _check_benchmark(benchmark)
    path = Path(result_dir).expanduser() / f"{benchmark}.json"
    if not path.is_file():
        raise ViewerDataError(f"Result file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewerDataError(f"Failed to read result file {path}: {exc}") from exc

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ViewerDataError(f"Expected a list of objects in result file: {path}")
    return rows


def load_benchmark_records(
    benchmark_root: str | Path,
    benchmark: str,
) -> dict[str, dict[str, Any]]:
    """Load display metadata without decoding the embedded high-resolution images."""
    _check_benchmark(benchmark)
    root = Path(benchmark_root).expanduser()
    if not root.is_dir():
        raise ViewerDataError(f"Benchmark root does not exist: {root}")

    if benchmark == "vstar":
        records = _load_vstar(root)
    elif benchmark in {"hr_bench_4k", "hr_bench_8k"}:
        records = _load_hr_bench(root, benchmark)
    else:
        records = _load_mme_realworld(root)

    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = normalize_id(record.get("id"))
        if not record_id:
            raise ViewerDataError(f"{benchmark} contains a record with an empty ID")
        if record_id in indexed:
            raise ViewerDataError(f"{benchmark} contains duplicate ID {record_id!r}")
        indexed[record_id] = record
    return indexed


def join_results(
    results: Iterable[dict[str, Any]],
    benchmark_records: dict[str, dict[str, Any]],
    benchmark: str,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for result in results:
        record_id = normalize_id(result.get("id"))
        source = benchmark_records.get(record_id)
        if source is None:
            missing_ids.append(record_id or "<empty>")
            continue

        result_label = str(result.get("label", "")).strip().upper()
        source_label = str(source.get("label", "")).strip().upper()
        if result_label and source_label and result_label != source_label:
            raise ViewerDataError(
                f"Label mismatch for {benchmark} ID {record_id}: "
                f"result={result_label!r}, benchmark={source_label!r}"
            )

        merged = dict(source)
        merged.update(result)
        merged["id"] = record_id
        merged["label"] = result_label or source_label
        merged["predicted"] = str(result.get("predicted", "")).strip().upper()
        merged["prediction"] = str(result.get("prediction", ""))
        merged["correct"] = bool(result.get("correct", False))
        joined.append(merged)

    if missing_ids:
        preview = ", ".join(repr(value) for value in missing_ids[:8])
        suffix = " ..." if len(missing_ids) > 8 else ""
        raise ViewerDataError(
            f"Could not match {len(missing_ids)} {benchmark} result IDs: {preview}{suffix}"
        )
    return joined


def _load_hr_bench(root: Path, benchmark: str) -> list[dict[str, Any]]:
    dataset = _load_from_disk(root / benchmark)
    dataset = _drop_columns(dataset, ["image"])
    image_root = root / benchmark / "images"

    records = []
    for item in dataset:
        record_id = normalize_id(item.get("index"))
        records.append(
            {
                "id": record_id,
                "question": str(item.get("question", "")).strip(),
                "options": [
                    {"label": label, "text": str(item[label])}
                    for label in ("A", "B", "C", "D")
                    if item.get(label) is not None
                ],
                "label": str(item.get("answer", "")).strip().upper(),
                "category": str(item.get("category") or item.get("cycle_category") or ""),
                "image_path": str(_image_for_id(image_root, record_id)),
            }
        )
    return records


def _load_mme_realworld(root: Path) -> list[dict[str, Any]]:
    benchmark = "mme_realworld_lite"
    dataset = _load_from_disk(root / benchmark)
    dataset = _drop_columns(dataset, ["bytes"])
    image_root = root / benchmark / "images"

    records = []
    for item in dataset:
        record_id = normalize_id(item.get("index"))
        records.append(
            {
                "id": record_id,
                "question": str(item.get("question", "")).strip(),
                "options": _normalize_options(item.get("multi-choice options") or []),
                "label": str(item.get("answer", "")).strip().upper(),
                "category": str(item.get("category") or ""),
                "subcategory": str(item.get("l2-category") or ""),
                "image_path": str(_image_for_id(image_root, record_id)),
            }
        )
    return records


def _load_vstar(root: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ViewerDataError(
            "The 'datasets' package is required. Install benchmark/requirements.txt first."
        ) from exc

    try:
        dataset = load_dataset(
            "craigwu/vstar_bench",
            split="test",
            download_mode="reuse_dataset_if_exists",
        )
    except Exception as exc:
        raise ViewerDataError(
            "Failed to load the locally cached craigwu/vstar_bench test split. "
            f"Original error: {exc}"
        ) from exc

    image_root = root / "vstar_bench"
    records = []
    for item in dataset:
        question, options = _parse_vstar_text(str(item.get("text", "")))
        records.append(
            {
                "id": normalize_id(item.get("question_id")),
                "question": question,
                "options": options,
                "label": str(item.get("label", "")).strip().upper(),
                "category": str(item.get("category") or ""),
                "image_path": str(image_root / str(item.get("image", ""))),
            }
        )
    return records


def _load_from_disk(path: Path):
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ViewerDataError(
            "The 'datasets' package is required. Install benchmark/requirements.txt first."
        ) from exc

    if not path.is_dir():
        raise ViewerDataError(f"Local benchmark dataset does not exist: {path}")
    try:
        dataset = load_from_disk(str(path))
    except Exception as exc:
        raise ViewerDataError(f"Failed to load local dataset {path}: {exc}") from exc

    if isinstance(dataset, DatasetDict):
        if "test" in dataset:
            return dataset["test"]
        if "train" in dataset:
            return dataset["train"]
        first_split = next(iter(dataset.keys()), None)
        if first_split is None:
            raise ViewerDataError(f"Local dataset has no splits: {path}")
        return dataset[first_split]
    return dataset


def _drop_columns(dataset: Any, columns: Iterable[str]):
    existing = [column for column in columns if column in dataset.column_names]
    return dataset.remove_columns(existing) if existing else dataset


def _normalize_options(values: Iterable[Any]) -> list[dict[str, str]]:
    options = []
    for index, value in enumerate(values):
        text = str(value).strip()
        match = _OPTION_RE.match(text)
        if match:
            label, option_text = match.groups()
        else:
            label = chr(ord("A") + index)
            option_text = text
        options.append({"label": label, "text": option_text})
    return options


def _parse_vstar_text(text: str) -> tuple[str, list[dict[str, str]]]:
    question_lines: list[str] = []
    options: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _OPTION_RE.match(line)
        if match:
            label, option_text = match.groups()
            options.append({"label": label, "text": option_text})
        elif not options:
            question_lines.append(line)
    return "\n".join(question_lines).strip(), options


def _image_for_id(image_root: Path, record_id: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"):
        candidate = image_root / f"{record_id}{extension}"
        if candidate.is_file():
            return candidate
    return image_root / f"{record_id}.jpg"


def _check_benchmark(benchmark: str) -> None:
    if benchmark not in BENCHMARKS:
        raise ViewerDataError(
            f"Unsupported benchmark {benchmark!r}; expected one of {', '.join(BENCHMARKS)}"
        )
