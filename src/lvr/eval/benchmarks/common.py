"""Shared benchmark loading utilities."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def materialize_image_value(
    value: Any,
    *,
    cache_dir: str | Path,
    stem: str,
    image_root: str | Path | None = None,
) -> str:
    """Write an embedded image value to disk and return a path.

    Path-like values are returned directly. Embedded image bytes/base64 are
    decoded one sample at a time, avoiding a full benchmark worth of PIL images
    in memory before evaluation starts.
    """
    if isinstance(value, str):
        text = value.strip()
        if _looks_like_image_path(text):
            return _resolve_path(text, image_root)
        if text.startswith("data:image"):
            text = text.split(",", 1)[1]
        raw = base64.b64decode(text, validate=True)
        return _write_image_bytes(raw, cache_dir=cache_dir, stem=stem)

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _write_image_bytes(value["bytes"], cache_dir=cache_dir, stem=stem)
        if value.get("path"):
            return _resolve_path(value["path"], image_root)

    if isinstance(value, (bytes, bytearray)):
        return _write_image_bytes(bytes(value), cache_dir=cache_dir, stem=stem)

    if isinstance(value, Image.Image):
        cache_path = Path(cache_dir) / f"{stem}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            value.convert("RGB").save(cache_path)
        return str(cache_path)

    raise TypeError(f"Unsupported benchmark image value type: {type(value).__name__}")


def decode_image_value(value: Any, *, image_root: str | Path | None = None) -> Image.Image | str:
    """Normalize common benchmark image encodings.

    Returns a RGB ``PIL.Image`` for embedded image bytes/base64 values and a
    string path for path-like values that are not base64.
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return _resolve_path(value["path"], image_root)

    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")

    if isinstance(value, str):
        text = value.strip()
        if _looks_like_image_path(text):
            return _resolve_path(text, image_root)
        if text.startswith("data:image"):
            text = text.split(",", 1)[1]
        try:
            return Image.open(BytesIO(base64.b64decode(text, validate=True))).convert("RGB")
        except Exception:
            return _resolve_path(value, image_root)

    raise TypeError(f"Unsupported benchmark image value type: {type(value).__name__}")


def option_text(options: list[str]) -> str:
    """Render multiple-choice options, preserving existing labels when present."""
    lines: list[str] = []
    for idx, option in enumerate(options):
        option = str(option).strip()
        if not option:
            continue
        if option.startswith("(") or option[:2].upper() in {"A.", "B.", "C.", "D.", "E."}:
            lines.append(option)
        else:
            letter = chr(ord("A") + idx)
            lines.append(f"({letter}) {option}")
    return "\n".join(lines)


def strip_option_labels(options: list[str]) -> list[str]:
    """Return option content without leading ``(A)``/``A.`` labels."""
    stripped: list[str] = []
    for option in options:
        text = str(option).strip()
        if len(text) >= 3 and text[0] == "(" and text[2:3] == ")":
            text = text[3:].strip()
        elif len(text) >= 2 and text[1:2] == "." and text[0].isalpha():
            text = text[2:].strip()
        stripped.append(text)
    return stripped


def _looks_like_image_path(value: str) -> bool:
    lower = value.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))


def _resolve_path(value: str, image_root: str | Path | None) -> str:
    path = Path(value)
    if image_root is not None and not path.is_absolute():
        path = Path(image_root) / path
    return str(path)


def _write_image_bytes(raw: bytes, *, cache_dir: str | Path, stem: str) -> str:
    ext = _image_ext(raw)
    cache_path = Path(cache_dir) / f"{stem}{ext}"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or cache_path.stat().st_size != len(raw):
        cache_path.write_bytes(raw)
    return str(cache_path)


def _image_ext(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"
