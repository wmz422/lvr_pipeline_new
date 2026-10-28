"""Default versioned training release; legacy paths require an explicit override."""

import json
from pathlib import Path


DEFAULT_TRAINING_ROOT = Path("data/default_training")


def training_data_root(args, project_root):
    return Path(getattr(args, "training_data_root", None) or Path(project_root) / DEFAULT_TRAINING_ROOT)


def split_records(directory):
    """Use the selected release's shard index, never a historical sample constant."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    count = index["records"]
    if type(count) is not int or count < 1 or sum(row["records"] for row in index["shards"]) != count:
        raise ValueError(f"Invalid training split count: {directory}")
    return count
