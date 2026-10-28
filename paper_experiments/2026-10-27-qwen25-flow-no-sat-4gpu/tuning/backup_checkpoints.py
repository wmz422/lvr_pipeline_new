"""Copy completed local checkpoints to durable storage without blocking GPUs.

Only checkpoints whose save every rank has passed are eligible. Each version becomes
visible atomically after all bytes are copied and fsynced; SHA256 manifests make
subsequent recovery verifiable. This tool never deletes source checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

PHASES = ("teacher_joint", "warmup", "joint")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def source_inventory(source):
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    if source.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Checkpoint payload contains a symlink")
    return {str(path.relative_to(source) if source.is_dir() else Path(source.name)):
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            for path in paths if path.is_file() for stat in [path.stat()]}


def copy_snapshot(source, destination, *, bytes_per_second, identity):
    if (destination / "backup_manifest.json").exists():
        saved = json.loads((destination / "backup_manifest.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Existing backup belongs to another checkpoint")
        return
    if destination.exists():
        raise ValueError("Refuse to replace an unrecognized backup")
    initial_inventory = source_inventory(source)
    if not initial_inventory:
        raise ValueError("Checkpoint payload is empty")
    files = [source / relative if source.is_dir() else source for relative in initial_inventory]
    if source.is_dir() and any(path.relative_to(source).as_posix() in
                              {"backup_manifest.json", "backup_manifest.json.partial"} for path in files):
        raise ValueError("Source uses a reserved backup manifest filename")
    total = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(destination.parent).free < total + 20 * 2**30:
        raise OSError("Insufficient durable storage for a complete checkpoint")
    temporary = destination.with_name(destination.name + ".copying")
    temporary.mkdir(exist_ok=True)
    manifest_partial = temporary / "backup_manifest.json.partial"
    if manifest_partial.is_symlink():
        raise ValueError("Staging manifest is a symlink")
    # atomic_json owns this reserved file; an interrupted manifest write can
    # safely be replaced after every payload byte is recopied and rehashed.
    if manifest_partial.exists():
        manifest_partial.unlink()
    copied, started, inventory = 0, time.monotonic(), {}
    for path in files:
        relative = path.relative_to(source) if source.is_dir() else Path(source.name)
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as src, target.open("wb") as dst:
            for chunk in iter(lambda: src.read(8 * 2**20), b""):
                dst.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                delay = copied / bytes_per_second - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
            dst.flush()
            os.fsync(dst.fileno())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source checkpoint changed while copying")
        inventory[str(relative)] = {"bytes": before.st_size, "sha256": digest.hexdigest()}
    if source_inventory(source) != initial_inventory:
        raise ValueError("Source checkpoint inventory changed while copying")
    staged = {str(path.relative_to(temporary)) for path in temporary.rglob("*")
              if path.is_file() and path.name != "backup_manifest.json"}
    if staged != set(inventory):
        raise ValueError("Staging directory contains unexpected checkpoint files")
    for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
    atomic_json(temporary / "backup_manifest.json", {
        "format": "flow-checkpoint-backup-v1", "identity": identity,
        "source": str(source), "files": inventory, "copied_bytes": copied,
        "completed_at": time.time(),
    })
    temporary.replace(destination)
    sync_directory(destination.parent)


def prune_superseded_recoveries(phase_directory, identity):
    """Keep the newest verified recovery; candidates/exports stay untouched."""
    if not re.fullmatch(r"recovery-\d+\.ckpt", identity["name"]):
        return
    committed = phase_directory / identity["name"] / "backup_manifest.json"
    if not committed.is_file() or committed.is_symlink():
        raise ValueError("New recovery backup has not been committed")
    current = json.loads(committed.read_text())
    if current.get("format") != "flow-checkpoint-backup-v1" or current.get("identity") != identity:
        raise ValueError("New recovery backup identity differs")
    for path in phase_directory.glob("recovery-*.ckpt"):
        if path.is_symlink() or not path.is_dir():
            continue
        manifest = path / "backup_manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            continue
        saved = json.loads(manifest.read_text())
        previous = saved.get("identity", {})
        if (saved.get("format") == "flow-checkpoint-backup-v1"
                and previous.get("recipe_sha256") == identity["recipe_sha256"]
                and previous.get("name") == path.name
                and previous.get("step", identity["step"]) < identity["step"]):
            shutil.rmtree(path)
    sync_directory(phase_directory)


def candidates(run):
    for phase in PHASES:
        directory = run / phase
        try:
            states = [json.loads((directory / f"runtime_rank{rank}.json").read_text())
                      for rank in range(4)]
            started = json.loads((directory / "training_started.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        safe_step = min(state["optimizer_steps"] for state in states)
        complete_path = directory / "complete.json"
        completed = None
        if complete_path.exists():
            completed = json.loads(complete_path.read_text())
            if (completed.get("status") != "complete"
                    or completed.get("recipe_sha256") != started["recipe_sha256"]
                    or completed.get("optimizer_steps") != safe_step
                    or any(state.get("status") != "training_complete"
                           or state["optimizer_steps"] != safe_step for state in states)):
                raise ValueError("Completed stage disagrees with rank cursors or recipe")
        for path in sorted((directory / "checkpoints").glob("*.ckpt")):
            if path.is_symlink():
                continue
            match = re.fullmatch(r"(?:candidate|recovery)-(\d+)\.ckpt", path.name)
            # Lightning moves ModelCheckpoint callbacks after RunAudit: a
            # cursor equal to this step can be visible BEFORE its save starts.
            # A later update on every rank proves the collective save returned.
            # The final checkpoint instead needs the completed-stage marker,
            # which the workflow writes after rank validation and consolidation.
            if (match and path.is_dir() and
                    (int(match[1]) < safe_step or completed is not None and int(match[1]) <= safe_step)):
                yield phase, path, {"recipe_sha256": started["recipe_sha256"],
                                    "step": int(match[1]), "name": path.name}
        if completed is not None:
            # The export is written after complete.json; the official exporter
            # uses atomic replacement, so a visible regular file is complete.
            for name in ("last_full.ckpt", "student_source.ckpt"):
                path = directory / "checkpoints" / name
                if path.is_file() and not path.is_symlink():
                    yield phase, path, {"recipe_sha256": started["recipe_sha256"],
                                       "step": safe_step, "name": name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mib-per-second", type=float, default=80)
    args = parser.parse_args()
    if args.mib_per_second <= 0:
        parser.error("Copy rate must be positive")
    run, destination = args.run.resolve(), args.destination.resolve()
    if destination == run or destination in run.parents or run in destination.parents:
        parser.error("Source and backup trees must be separate")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    while True:
        try:
            for phase, source, identity in candidates(run):
                target = destination / phase / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(destination / "status.json", {
                    "status": "copying", "source": str(source), "timestamp": time.time()})
                copy_snapshot(source, target, bytes_per_second=args.mib_per_second * 2**20,
                              identity=identity)
                prune_superseded_recoveries(target.parent, identity)
            # Plans and recipes are small, useful for exact resume and interpretation.
            for path in [*run.glob("*.json"), *run.glob("recipes/*.yaml"),
                         *run.glob("*/training_started.json"), *run.glob("*/complete.json"),
                         *run.glob("*/runtime_rank*.json")]:
                target = destination / "run_metadata" / path.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    # Rank cursors are rewritten during training; refuse an
                    # intermediate JSON read instead of committing partial metadata.
                    atomic_json(target, json.loads(path.read_text()))
                    continue
                temporary = target.with_name(target.name + ".partial")
                shutil.copyfile(path, temporary)
                temporary.replace(target)
            atomic_json(destination / "status.json", {"status": "waiting", "timestamp": time.time()})
        except (OSError, ValueError) as error:
            atomic_json(destination / "status.json", {
                "status": "retry_pending", "error": str(error), "timestamp": time.time()})
        time.sleep(30)


if __name__ == "__main__":
    main()
