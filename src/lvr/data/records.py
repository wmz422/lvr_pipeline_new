"""Read JSON arrays or ordered, checksummed JSON shards."""

import hashlib
import json
from pathlib import Path


def read_records(path, limit=None):
    path = Path(path)
    if path.is_dir():
        path = path / "index.json"
    content = json.loads(path.read_text())
    if isinstance(content, list):
        return content[:limit] if limit is not None else content
    if content.get("format") != "lvr-json-shards-v1":
        raise ValueError(f"Expected a JSON array or LVR shard index: {path}")
    records = []
    for shard in content["shards"]:
        raw = (path.parent / shard["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != shard["sha256"]:
            raise ValueError(f"Corrupt JSON shard: {shard['file']}")
        rows = json.loads(raw)
        if len(rows) != shard["records"]:
            raise ValueError(f"Incorrect record count: {shard['file']}")
        records.extend(rows)
        if limit is not None and len(records) >= limit:
            return records[:limit]
    if len(records) != content["records"]:
        raise ValueError(f"Incorrect total record count: {path}")
    return records
