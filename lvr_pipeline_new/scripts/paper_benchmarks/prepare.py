"""Materialize all declared official splits, preserving original image bytes."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image
import pyarrow.parquet as pq
from lvr.eval.paper_benchmark import (DEFAULT_SUITE, EXPERIMENT, ROOT, SCHEMA,
                                      check_counts, digest, file_sha, write_json)

sys.path.insert(0, str(ROOT / "scripts/reasoning_benchmarks"))
from download import verify

LEGACY = ROOT / "benchmark/reasoning_v1"
BENCH = ROOT / "benchmark/bench"
BLINK_REV = "a3666eb249237ba3d5eca8db21176cc47967e040"
SCORERS = {
    "BLINK": ("zeyofu/BLINK_Benchmark", "529b0ba055416dea5df79d594e7f62a910d8a308", ["eval/evaluate.py", "eval/multiple_choice.py", "LICENSE"]),
    "SAT": ("arijitray1993/lmms-eval", "8856d847809d5586059fae171e8039c5af0301e1", ["lmms_eval/tasks/sat_real/utils.py", "lmms_eval/tasks/sat_real/sat_real.yaml", "LICENSE"]),
}


def fetch(url):
    # Honor the workspace's network proxy; direct HF connections are unavailable
    # on some workers. Every dataset payload is checked against its pinned hash.
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "paper-benchmark-v1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read(), dict(response.headers)
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def link_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha(target) != file_sha(source):
            raise ValueError(f"Existing asset differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def prepare_blink(root, tasks):
    dest = root / "raw/BLINK"
    lock_path = dest / "source_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        raw, _ = fetch(f"https://huggingface.co/api/datasets/BLINK-Benchmark/BLINK/revision/{BLINK_REV}?blobs=1")
        info = json.loads(raw)
        if info["sha"] != BLINK_REV:
            raise ValueError("BLINK source revision mismatch")
        paths = {f"{task}/val-00000-of-00001.parquet" for task in tasks} | {"README.md"}
        entries = []
        for item in info["siblings"]:
            if item["rfilename"] in paths:
                entry = {"path": item["rfilename"], "bytes": item["size"]}
                if item.get("lfs"):
                    entry["sha256"] = item["lfs"]["sha256"]
                else:
                    entry["git_blob_sha1"] = item["blobId"]
                entries.append(entry)
        if {e["path"] for e in entries} != paths:
            raise ValueError("Missing official BLINK val files")
        lock = {"repo_id": "BLINK-Benchmark/BLINK", "revision": BLINK_REV, "files": entries}
        write_json(lock_path, lock)
    if lock["revision"] != BLINK_REV or len(lock["files"]) != 15:
        raise ValueError("Wrong BLINK source lock")
    def download(entry):
        target = dest / entry["path"]
        reusable = BENCH / "_raw/reasoning_v1/BLINK" / entry["path"]
        if not target.exists() and reusable.is_file():
            verify(reusable.read_bytes(), entry)
            link_copy(reusable, target)
        if not target.exists():
            raw, _ = fetch(f"https://huggingface.co/datasets/BLINK-Benchmark/BLINK/resolve/{BLINK_REV}/{entry['path']}")
            verify(raw, entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(target)
        verify(target.read_bytes(), entry)
        print("VERIFIED BLINK", entry["path"], flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, lock["files"]))
    return lock


class Builder:
    def __init__(self, root, protocol):
        self.root, self.protocol = root, protocol
        self.rows, self.assets, self.sources = [], {}, {}

    def image(self, value):
        if isinstance(value, dict):
            value = value["bytes"] if value.get("bytes") is not None else Path(value["path"])
        payload = value.read_bytes() if isinstance(value, Path) else base64.b64decode(value, validate=True) if isinstance(value, str) else value
        sha = hashlib.sha256(payload).hexdigest()
        if sha not in self.assets:
            with Image.open(io.BytesIO(payload)) as im:
                width, height = im.size
                suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(im.format, ".image")
            relative = f"assets/{sha[:2]}/{sha}{suffix}"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, Path):
                link_copy(value, path)
            elif not path.exists():
                path.write_bytes(payload)
            elif file_sha(path) != sha:
                raise ValueError(f"Asset changed: {relative}")
            self.assets[sha] = {"path": relative, "sha256": sha, "width": width, "height": height}
        return self.assets[sha]

    def add(self, name, sid, task, question, answer, images, options, *, metadata=None, order=0):
        ims = [self.image(im) for im in images]
        base_id = f"{name}/{sid}"
        self.rows.append({"schema": SCHEMA, "id": base_id + (f"/order_{order}" if name == "SAT" else ""),
                          "base_id": base_id, "benchmark": name, "task": task, "option_order": order,
                          "split": self.protocol["benchmarks"][name]["split"], "difficulty": None,
                          "question": question.rstrip() + self.protocol["inference"]["answer_instruction"],
                          "answer": str(answer).strip("() "), "scoring": {"kind": "choice"},
                          "images": ims, "helper_images": [], "options": options,
                          "group_id": digest([im["sha256"] for im in ims]),
                          "source": {"lock": name, "id": str(sid)}, "metadata": metadata or {}})

    def import_prepared(self):
        manifest = json.loads((LEGACY / "manifest.json").read_text())
        for filename in ("records.jsonl", "source_lock.json", "selection.json", "overlap_report.json"):
            if file_sha(LEGACY / filename) != manifest["files"][filename]:
                raise ValueError(f"Previously accepted source changed: {filename}")
        locks = json.loads((LEGACY / "source_lock.json").read_text())
        for row in map(json.loads, (LEGACY / "records.jsonl").read_text().splitlines()):
            if row["benchmark"] == "BLINK_Spatial":
                continue
            row = copy.deepcopy(row)
            row.update(schema=SCHEMA, base_id=row["id"], option_order=0)
            # The official VSP direct prompt already asks for a boxed final answer.
            # Avoid two competing wrappers: keep the shared direct-answer format.
            row["question"] = row["question"].replace(" The final answer MUST BE put in \\boxed{}.", "")
            for im in row["images"] + row["helper_images"]:
                if im["sha256"] not in self.assets:
                    if file_sha(LEGACY / im["path"]) != im["sha256"]:
                        raise ValueError("Imported image changed")
                    link_copy(LEGACY / im["path"], self.root / im["path"])
                    self.assets[im["sha256"]] = im
            self.rows.append(row)
        self.sources.update({k: v for k, v in locks.items() if k != "BLINK_Spatial"})
        for relative, sha in manifest["files"].items():
            if relative.startswith("upstream/"):
                if file_sha(LEGACY / relative) != sha:
                    raise ValueError("Pinned scorer changed")
                link_copy(LEGACY / relative, self.root / relative)
        for filename in ("selection.json", "overlap_report.json"):
            link_copy(LEGACY / filename, self.root / "audit" / ("prior_" + filename))
        write_json(self.root / "audit/overlap_policy.json", {
            "selection": "full_declared_splits", "excluded_records": 0,
            "known_prior_overlap_records": {"BLINK_Spatial": 32, "MathVista_testmini": 8, "MentisOculi": 99},
            "interpretation": "Retain full sets as requested. Some original/helper images overlap training or development data. The historical audit is not a complete audit of the expanded suite or current training release.",
            "active_training": "VStar-decontaminated release; no changes to the running experiment",
            "prior_manifest_sha256": file_sha(LEGACY / "manifest.json")})

    def local_full(self):
        from datasets import load_from_disk
        locks = json.loads((ROOT / "data/sources.json").read_text())["sources"]
        path = BENCH / "vstar_bench/test_questions.jsonl"
        for item in map(json.loads, path.read_text().splitlines()):
            matches = list(re.finditer(r"(?m)^\(([A-Z])\) (.*)$", item["text"]))
            if not matches:
                raise ValueError("VStar options missing")
            self.add("VStarBench", item["question_id"], item["category"], item["text"], item["label"],
                     [BENCH / "vstar_bench" / item["image"]], [m[2] for m in matches])
        self.sources["VStarBench"] = {**locks["vstar"], "prepared_metadata_sha256": file_sha(path)}
        for name, folder in (("HRBench4K", "hr_bench_4k"), ("HRBench8K", "hr_bench_8k"), ("MME-RealWorld-Lite", "mme_realworld_lite")):
            root = BENCH / folder
            dataset = load_from_disk(str(root))
            for item in dataset:
                mme = name == "MME-RealWorld-Lite"
                options = [v[4:] for v in item["multi-choice options"]] if mme else [item[k] for k in "ABCD"]
                prompt = item["question"] + "\n" + "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(options))
                prompt += "\nAnswer with only the option letter."
                metadata = {k: item[k] for k in ("category", "l2-category", "cycle_category") if k in item}
                task = item["category"] + ("/" + item["l2-category"] if mme else "")
                self.add(name, item["index"], task, prompt, item["answer"],
                         [item["bytes"] if mme else item["image"]], options, metadata=metadata)
            self.sources[name] = {**locks[folder], "prepared_files": {
                p.name: file_sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix in (".json", ".arrow")}}
            print("BUILT", name, len(dataset), flush=True)

    def blink(self):
        lock = prepare_blink(self.root, self.protocol["blink_tasks"])
        for entry in lock["files"]:
            if not entry["path"].endswith(".parquet"):
                continue
            task = entry["path"].split("/")[0]
            table = pq.read_table(self.root / "raw/BLINK" / entry["path"])
            if table.num_rows != self.protocol["blink_tasks"][task]:
                raise ValueError(f"BLINK {task} count changed")
            for item in table.to_pylist():
                prompt = item["question"] + "\nSelect from the following choices.\n"
                prompt += "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(item["choices"]))
                prompt += "\nAnswer with only the option letter."
                self.add("BLINK", str(item["idx"]), task, prompt, item["answer"],
                         [item[k] for k in ("image_1", "image_2", "image_3", "image_4") if item[k] is not None], item["choices"])
        self.sources["BLINK"] = lock

    def sat(self):
        root = ROOT / "data/raw/reasoning_expansion/eval_exclusions/SAT-v2"
        lock = json.loads((root / "source_lock.json").read_text())
        spec = self.protocol["sat"]
        if (lock["repo_id"], lock["revision"]) != (spec["repo_id"], spec["revision"]):
            raise ValueError("SAT version mismatch")
        entry = next(e for e in lock["files"] if e["path"].startswith("data/test-"))
        verify((root / entry["path"]).read_bytes(), entry)
        for index, item in enumerate(pq.read_table(root / entry["path"]).to_pylist()):
            if len(item["answers"]) != 2 or item["answers"].count(item["correct_answer"]) != 1:
                raise ValueError("Expected unique SAT answer among two choices")
            for order, options in enumerate((item["answers"], item["answers"][::-1])):
                prompt = item["question"] + " Choose from the following options:\n"
                prompt += "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(options))
                prompt += "\nAnswer with only the option letter."
                self.add("SAT", f"test_{index:04d}", item["question_type"], prompt,
                         chr(65 + options.index(item["correct_answer"])), item["images"], options, order=order)
        self.sources["SAT"] = {"repo_id": lock["repo_id"], "revision": lock["revision"], "files": [entry]}

    def scorers(self):
        for name, (repo, revision, paths) in SCORERS.items():
            for filename in paths:
                target = self.root / "upstream" / name / filename
                if not target.exists():
                    raw, _ = fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
            self.sources[name]["reference_scorer"] = {"repo": repo, "revision": revision, "files": paths}
        for filename in ("hrbench.py", "multiple_choice.py"):
            source = ROOT / "third_party/VLMEvalKit/vlmeval/dataset/utils" / filename
            link_copy(source, self.root / "upstream/VLMEvalKit" / filename)

    def run(self):
        self.import_prepared()
        self.local_full()
        self.blink()
        self.sat()
        self.scorers()
        self.rows.sort(key=lambda r: (list(self.protocol["benchmarks"]).index(r["benchmark"]), r["id"]))
        check_counts(self.rows, self.protocol)
        (self.root / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows))
        write_json(self.root / "source_lock.json", self.sources)
        write_json(self.root / "protocol.json", self.protocol)
        files = [self.root / name for name in ("protocol.json", "records.jsonl", "source_lock.json")]
        files += [p for folder in ("upstream", "audit") for p in (self.root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        manifest = {"schema": SCHEMA, "status": "prepared", "records": self.protocol["total_records"],
                    "generations": len(self.rows), "images": len(self.assets),
                    "files": {str(p.relative_to(self.root)): file_sha(p) for p in sorted(files)}}
        write_json(self.root / "manifest.json", manifest)
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), LVR_REASONING_SUITE=str(self.root), MPLBACKEND="Agg")
        subprocess.run([os.environ.get("LVR_SCORING_PYTHON", str(ROOT.parent / ".venv-score/bin/python")), str(Path(__file__).with_name("score.py")),
                        "--suite", str(self.root), "--accept"], env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "configs/protocol.json")
    args = parser.parse_args()
    args.suite.mkdir(parents=True, exist_ok=True)
    if (args.suite / "manifest.json").exists():
        if json.loads((args.suite / "manifest.json").read_text())["status"] == "accepted":
            from lvr.eval.paper_benchmark import load_suite
            _, protocol, _ = load_suite(args.suite)
            if protocol != json.loads(args.protocol.read_text()):
                raise ValueError("Accepted suite is immutable; choose a new version/directory")
            print("ALREADY ACCEPTED", args.suite)
            return
    Builder(args.suite, json.loads(args.protocol.read_text())).run()


if __name__ == "__main__":
    main()
