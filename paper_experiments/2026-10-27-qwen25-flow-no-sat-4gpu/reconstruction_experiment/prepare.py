"""Prepare reconstruction images from the exact decontaminated Plain SFT inputs."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
from flow_reconstruction.gt import GTConfig, digest_file, process_sample, read_records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def record_file(path):
    """Plain's no-SAT reasoning split is a JSON list, legacy splits are directories."""
    path = Path(path)
    return path / "index.json" if path.is_dir() else path


def protocol(data, project):
    paths = {key: str(Path(data[key]).resolve()) for key in (
        "train_path", "val_path", "train_boxes_path", "val_boxes_path",
        "additional_train_path", "additional_val_path")}
    value = {"schema": "lvr-flow-gt-v1", "root": str(project), "image_root": data["image_root"],
             "data_paths": paths, "limit": None,
             "gt_config": asdict(GTConfig(reference_pixels=data["original_pixels"],
                                            masked_original_pixels=data["original_pixels"],
                                            crop_pixels=data["crop_pixels"], helper_pixels=data["crop_pixels"])),
             "input_bindings": {path: digest_file(record_file(path)) for path in paths.values()},
             "implementation_sha256": {"prepare.py": digest_file(__file__),
                                       "gt.py": digest_file(__import__("flow_reconstruction.gt", fromlist=["__file__"]).__file__)}}
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def samples(settings, split):
    paths = settings["data_paths"]
    prefix = "train" if split == "train" else "val"
    qa = read_records(paths[prefix + "_path"])
    annotations = read_records(paths[prefix + "_boxes_path"])
    if len(qa) != len(annotations):
        raise ValueError("Clean QA/annotation counts differ")
    image_root, project = Path(settings["image_root"]), Path(settings["root"])
    for row, annotation in zip(qa, annotations):
        if any(row[k] != annotation[k] for k in ("question_image", "auxiliary_image")):
            raise ValueError("Clean QA/annotation order differs")
        if not annotation["boxes"]:
            raise ValueError("Missing reconstruction crop")
        yield {"id": f"{row['question_image']}:{row['raw']['index']}", "source": row["raw"]["dataset"],
               "refs": [{"path": str(image_root / row["question_image"])}],
               "boxes": annotation["boxes"], "helpers": []}
    for row in read_records(paths["additional_" + prefix + "_path"]):
        helpers = row["auxiliary_images"]
        if row.get("teacher_view_mode") != "auxiliary_images" or not helpers or not row["problem_images"]:
            raise ValueError("Invalid reasoning teacher images")
        if [h["step"] for h in helpers] != list(range(1, len(helpers) + 1)):
            raise ValueError("Helper order differs")
        yield {"id": row["id"], "source": row["source"], "boxes": [],
               "refs": [{**r, "path": str(project / r["path"])} for r in row["problem_images"]],
               "helpers": [{**h, "path": str(project / h["path"])} for h in helpers]}


def bounded_map(pool, fn, items, capacity):
    iterator, pending = iter(items), deque()
    for _ in range(capacity):
        item = next(iterator, None)
        if item is not None:
            pending.append(pool.submit(fn, item))
    while pending:
        yield pending.popleft().result()
        item = next(iterator, None)
        if item is not None:
            pending.append(pool.submit(fn, item))


def validate_ready(output, settings, counts):
    output = Path(output)
    if not (output / "status.json").exists():
        return False
    status = json.loads((output / "status.json").read_text())
    if status["status"] != "complete":
        return False
    if json.loads((output / "protocol.json").read_text()) != settings:
        raise ValueError("Reconstruction protocol differs")
    if status["protocol_sha256"] != settings["protocol_sha256"]:
        raise ValueError("Reconstruction status/protocol mismatch")
    for split, count in counts.items():
        item = status["splits"][split]
        if item["records"] != count or digest_file(output / (split + ".jsonl")) != item["manifest_sha256"]:
            raise ValueError("Reconstruction manifest/count changed")
    return True


def prepare(output, settings, counts, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("Use a new flow-data directory for changed settings")
        if validate_ready(output, settings, counts):
            return
        write_json(path, settings)
        status = {"status": "processing", "protocol_sha256": settings["protocol_sha256"], "splits": {}}
        write_json(output / "status.json", status)
        cfg, seen = GTConfig(**settings["gt_config"]), set()
        try:
            for split, expected in counts.items():
                manifest = output / (split + ".jsonl")
                partial = manifest.with_suffix(".jsonl.partial")
                old_path = manifest if manifest.exists() else partial
                known = {}
                if old_path.exists():
                    raw = old_path.read_bytes()
                    if raw and not raw.endswith(b"\n"):
                        raw = raw[:raw.rfind(b"\n") + 1]
                    for line in raw.decode().splitlines():
                        row = json.loads(line)
                        if row["id"] in known:
                            raise ValueError("Duplicate reconstruction ID")
                        known[row["id"]] = row
                source_counts, kind_counts = Counter(), Counter()

                def job(sample):
                    old = known.get(sample["id"])
                    if old is not None:
                        for item in [old["target"], *old["references"]]:
                            if digest_file(output / item["path"]) != item["sha256"]:
                                raise ValueError("Changed prepared image")
                        for item in old["source_hashes"]:
                            if digest_file(item["path"]) != item["sha256"]:
                                raise ValueError("Changed source image")
                        return old
                    return process_sample(sample, output, cfg)

                n = 0
                with partial.open("w") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
                    for n, row in enumerate(bounded_map(pool, job, samples(settings, split), 2 * workers), 1):
                        if row["id"] in seen:
                            raise ValueError("Duplicate train/validation ID")
                        seen.add(row["id"])
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        source_counts[row["source"]] += 1
                        kind_counts[row["layout"]["kind"]] += 1
                        if n % 1000 == 0:
                            print(f"PREPARE_RECONSTRUCTION {split} {n}/{expected}", flush=True)
                if n != expected:
                    raise ValueError(f"Expected {expected} {split} samples, got {n}")
                partial.replace(manifest)
                status["splits"][split] = {"records": n, "full_records": n,
                    "sources": dict(source_counts), "kinds": dict(kind_counts), "manifest_sha256": digest_file(manifest)}
                write_json(output / "status.json", status)
            status["status"] = "complete"
            write_json(output / "status.json", status)
        except BaseException as exc:
            status.update(status="failed", error=repr(exc))
            write_json(output / "status.json", status)
            raise
