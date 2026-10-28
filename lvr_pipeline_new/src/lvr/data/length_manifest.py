"""Bound Plain QA length sidecars and deterministic preflight sample orders."""

import gzip
import hashlib
import json
from pathlib import Path


PROCESSOR_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                   "tokenizer.json", "vocab.json", "merges.txt", "chat_template.json")


def processor_fingerprint(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Length manifests require a local, fixed processor directory")
    result = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in PROCESSOR_FILES if (directory / name).is_file()}
    if not {"config.json", "preprocessor_config.json", "tokenizer_config.json"}.issubset(result):
        raise ValueError("Incomplete processor files for length-manifest binding")
    return result


def plain_qa_items_fingerprint(items):
    digest = hashlib.sha256()
    for item in items:
        row = {key: item.get(key) for key in ("id", "question", "answer", "question_image", "images")}
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_plain_qa_lengths(path, items, processor_directory, original_pixels, max_length):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {"format": "lvr-plain-qa-lengths-v1", "records": len(items),
                "original_pixels": original_pixels,
                "processor_files_sha256": processor_fingerprint(processor_directory),
                "items_sha256": plain_qa_items_fingerprint(items)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Plain QA length manifest mismatch: {key}")
    lengths = manifest.get("lengths")
    if not isinstance(lengths, list) or len(lengths) != len(items):
        raise ValueError("Length manifest must cover every training sample in order")
    if any(type(length) is not int or not 0 < length <= max_length for length in lengths):
        raise ValueError("Length manifest contains an invalid or over-budget length")
    return lengths


def select_sample_items(path, items):
    """Repeat real samples in the exact requested order for an isolated stress run."""
    payload = json.loads(Path(path).read_text())
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids or any(type(x) is not str for x in sample_ids):
        raise ValueError("Preflight sample_ids must be a nonempty list of strings")
    indexed = {item["id"]: item for item in items}
    if len(indexed) != len(items):
        raise ValueError("Original dataset contains duplicate sample IDs")
    missing = set(sample_ids) - indexed.keys()
    if missing:
        raise ValueError(f"Unknown preflight sample IDs: {sorted(missing)[:3]}")
    return [indexed[sample_id] for sample_id in sample_ids]
