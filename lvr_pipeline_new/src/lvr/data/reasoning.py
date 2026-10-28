"""Versioned reasoning records; descriptions are mandatory, ordered targets."""
from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "lvr-reasoning-qa-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_DESCRIPTION_PROMPT = (
    "Describe the visible auxiliary image states represented by the latent visual "
    "information in input order, using State 1, State 2, and so on. Use one to three "
    "concise factual sentences per state. Include visible objects, shapes, colors, "
    "positions, orientations, paths, readable text and diagram labels. Spatial "
    "relationships within each image may be described. Do not infer hidden states, "
    "explain a reasoning process, solve a question, or state an answer conclusion."
)


def assemble_description(parts):
    return " ".join(f"State {i}: {p['text'].strip()}" for i, p in enumerate(parts, 1))


def validate_record(row, *, check_files=False):
    if row.get("schema") != SCHEMA:
        raise ValueError("Unknown reasoning record schema")
    for key in ("id", "source", "source_revision", "category", "task", "question",
                "answer", "answer_type", "group_id", "description"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing/non-string {key}: {row.get('id')}")
    if row.get("description_valid") is not True:
        raise ValueError(f"Unapproved description: {row['id']}")
    helpers = row.get("auxiliary_images", [])
    parts = row.get("description_parts", [])
    if not row.get("problem_images") or not helpers or len(parts) != len(helpers):
        raise ValueError(f"Image/description cardinality mismatch: {row['id']}")
    for i, (im, part) in enumerate(zip(helpers, parts), 1):
        if im.get("step") != i or part.get("step") != i or im.get("sha256") != part.get("image_sha256"):
            raise ValueError(f"Image/description order or hash mismatch: {row['id']}")
        if not part.get("text", "").strip() or part.get("quality_status") != "passed":
            raise ValueError(f"Invalid description part: {row['id']}")
    if row["description"] != assemble_description(parts):
        raise ValueError(f"Assembled description mismatch: {row['id']}")
    provenance = row.get("description_provenance", {})
    if (provenance.get("image_sha256") != [im["sha256"] for im in helpers]
            or not provenance.get("model_revision") or not provenance.get("prompt_version")
            or provenance.get("quality_status") != "passed"):
        raise ValueError(f"Invalid description provenance: {row['id']}")
    for im in row["problem_images"] + helpers:
        if not im.get("path") or len(im.get("sha256", "")) != 64:
            raise ValueError(f"Invalid image asset: {row['id']}")
        if check_files:
            path = PROJECT_ROOT / im["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != im["sha256"]:
                raise ValueError(f"Corrupt image: {path}")
    if row.get("teacher_view_mode") != "auxiliary_images":
        raise ValueError("Reasoning records require auxiliary_images teacher mode")


def as_crop_qa_item(row):
    validate_record(row)
    return {**row, "images": [
        {"path": str(PROJECT_ROOT / im["path"]), "boxes": []}
        for im in row["problem_images"]
    ], "auxiliary_images": [
        {**im, "path": str(PROJECT_ROOT / im["path"])} for im in row["auxiliary_images"]
    ], "boxes": [], "description_prompt": STATE_DESCRIPTION_PROMPT}
