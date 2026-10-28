"""Build the six selected public benchmarks into one reproducible local protocol."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.fft import dctn

from lvr.eval.reasoning_benchmark import ANSWER_INSTRUCTION, DEFAULT_SUITE, EXPECTED, ROOT, SCHEMA, file_sha, make_donors
from download import write_json, verify

EXPANSION = ROOT / "data/raw/reasoning_expansion"
RAW = ROOT / "benchmark/bench/_raw/reasoning_v1"
MENTIS_REV = "c0d486ca77481073d26a8c3447a1e348bb58d39a"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class Builder:
    def __init__(self, target):
        self.target = target
        self.assets = {}
        self.records = []
        self.locks = {}
        self.repairs = []

    def image(self, value):
        if isinstance(value, dict):
            value = value["bytes"] if value.get("bytes") is not None else Path(value["path"])
        payload = value.read_bytes() if isinstance(value, Path) else value
        sha = hashlib.sha256(payload).hexdigest()
        if sha in self.assets:
            return self.assets[sha]
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            rgb = im.convert("RGB")
            width, height = rgb.size
            pixel = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
            coefficients = dctn(np.asarray(rgb.resize((32, 32)).convert("L"), dtype=float), norm="ortho")[:8, :8]
            bits = coefficients > np.median(coefficients.reshape(-1)[1:])
            phash = f'{int("".join("1" if v else "0" for v in bits.flatten()), 2):016x}'
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(im.format, ".image")
        relative = f"assets/{sha[:2]}/{sha}{suffix}"
        path = self.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if file_sha(path) != sha:
                raise ValueError(f"Existing asset mismatch: {path}")
        else:
            path.write_bytes(payload)
        self.assets[sha] = {"path": relative, "sha256": sha, "pixel_sha256": pixel,
                            "phash": phash, "width": width, "height": height}
        return self.assets[sha]

    def add(self, benchmark, task, sid, question, answer, kind, images, helpers=(), *,
            difficulty=None, scoring=None, source=None, group=None, options=None, metadata=None):
        if not str(answer).strip() or not question.strip() or not images:
            raise ValueError(f"Incomplete benchmark row: {benchmark}/{sid}")
        images = [self.image(im) for im in images]
        helpers = [self.image(im) for im in helpers]
        row = {"schema": SCHEMA, "id": f"{benchmark}/{sid}", "benchmark": benchmark, "task": task,
               "split": "testmini" if benchmark == "MathVista_testmini" else "val" if benchmark == "BLINK_Spatial" else "test",
               "difficulty": difficulty, "question": question.strip() + ANSWER_INSTRUCTION,
               "answer": str(answer), "scoring": {"kind": kind, **(scoring or {})},
               "images": images, "helper_images": helpers, "options": options or [],
               "group_id": group or digest([im["pixel_sha256"] for im in images]),
               "source": source or {}, "metadata": metadata or {}}
        self.records.append(row)

    def mentis(self):
        root = EXPANSION / "mentis-oculi"
        tree = json.loads((root / "tree.json").read_text())
        if tree["sha"] != MENTIS_REV:
            raise ValueError("Mentis source changed")
        entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        checked = {}
        def checked_path(path):
            relative = str(path.relative_to(root))
            if relative not in checked:
                raw = path.read_bytes()
                blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                if entries[relative]["sha"] != blob:
                    raise ValueError(f"Mentis upstream bytes differ: {relative}")
                checked[relative] = {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha1": blob}
            return path
        for task in ("form-board", "hinge-folding", "paper-fold", "rushhour", "sliding-puzzle"):
            paths = sorted((root / "datasets" / task / "output").glob("level_*/puzzle_*/metadata.json"))
            if len(paths) != 250:
                raise ValueError(f"Mentis count mismatch: {task}")
            for path in paths:
                m = json.loads(checked_path(path).read_text())
                sid = f"{task}/{path.parent.parent.name}/{path.parent.name}"
                level = int(path.parent.parent.name.split("_")[-1])
                def resolve(ref):
                    for base in (path.parent, path.parent.parent, path.parent.parent.parent):
                        candidate = base / ref
                        if candidate.is_file():
                            return checked_path(candidate)
                    # Pinned Form Board metadata prefixes CoT files with puzzle number;
                    # the actual official tree stores cot_XX.png in each puzzle directory.
                    candidate = path.parent / re.sub(r"^\d+_", "", Path(ref).name)
                    if task == "form-board" and candidate.is_file():
                        self.repairs.append({"id": f"MentisOculi/{sid}", "field": ref, "resolved": candidate.name})
                        return checked_path(candidate)
                    raise FileNotFoundError((path, ref))
                if task == "form-board":
                    refs = [m["combined_image"]]
                    question = m["question"] + " Give the set of option letters."
                    answer = " ".join(m["solution_pieces"]); kind = "set"
                elif task == "paper-fold":
                    refs = [m["combined_image"]]
                    question = m["question"] + " Choose one of A, B, C, D, E."
                    answer = m["correct_choice"]; kind = "choice"
                elif task == "hinge-folding":
                    refs = [m["initial_image"], m["target_image"]]
                    question = "The first image shows an unfolded chain with labeled hinges. The second image shows the target configuration. Rotate the part of the chain to the right of each hinge counterclockwise to form the target. Give the rotations in hinge order as letter and angle in degrees, for example: A 90, B 180."
                    answer = m["rotation_sequence"]; kind = "hinge"
                elif task == "sliding-puzzle":
                    refs = [m["initial_image"], m["target_image"]]
                    question = "The first image is the current sliding puzzle and the second is the target. Move the blank square up, down, left, or right by swapping it with an adjacent tile. Give a shortest sequence of blank-square moves, separated by commas."
                    answer = ", ".join(m["solution_moves"]); kind = "sliding"
                else:
                    refs = [m["initial_state_image"]]
                    question = "Move the red car R to the green exit. Each labeled car can move forward along its arrow or backward along its rail, without rotating or colliding. Each move goes as far as possible until blocked; R can stop at the exit. Give a shortest sequence of label and direction pairs, separated by commas, for example: A forward, R backward."
                    labels = {o["id"]: "R" if o["id"] == "red_car" else o.get("label") for o in m["objects"]}
                    answer = ", ".join(labels[a["object_id"]] + (" forward" if a["direction"] == 1 else " backward") for a in m["actions"])
                    kind = "rushhour"
                self.add("MentisOculi", task, sid, question, answer, kind, [resolve(ref) for ref in refs],
                         [resolve(ref) for ref in m["cot_images"]], difficulty=level,
                         scoring={"state": m}, source={"repo": "Jana-Z/mentis-oculi", "revision": MENTIS_REV,
                         "path": str(path.relative_to(root))}, options=list("ABCDE") if kind == "choice" else [])
        self.locks["MentisOculi"] = {"repo": "Jana-Z/mentis-oculi", "revision": MENTIS_REV, "files": list(checked.values())}

    def blink(self):
        root = RAW / "BLINK"
        lock = json.loads((root / "download_manifest.json").read_text())
        for e in lock["files"]:
            verify((root / e["path"]).read_bytes(), e)
            if not e["path"].endswith(".parquet"):
                continue
            task = e["path"].split("/")[0]
            for r in pq.read_table(root / e["path"]).to_pylist():
                question = r["question"] + "\nSelect from the following choices.\n" + "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(r["choices"]))
                images = [r[k] for k in ("image_1", "image_2", "image_3", "image_4") if r[k] is not None]
                self.add("BLINK_Spatial", task, str(r["idx"]), question, r["answer"].strip("()"), "choice", images,
                         options=r["choices"], source={"repo": lock["repo_id"], "revision": lock["revision"], "id": r["idx"]})
        self.locks["BLINK_Spatial"] = lock

    def vsp(self):
        root = EXPANSION / "Mirage"
        lock = json.loads((root / "download_manifest.json").read_text())
        entry = next(e for e in lock["archives"] if e["file"] == "vsp_spatial_planning_test.tar.gz")
        archive = root / entry["file"]
        if file_sha(archive) != entry["sha256"]:
            raise ValueError("VSP archive changed")
        from lvr.eval.reasoning_scoring import shortest_grid_path
        with tarfile.open(archive) as tar:
            members = {m.name.lstrip("./"): m for m in tar.getmembers() if m.isfile()}
            annotations = next(m for name, m in members.items() if name.endswith("test_direct.jsonl"))
            for line in tar.extractfile(annotations):
                r = json.loads(line)
                matches = [m for name, m in members.items() if r["image_input"].lstrip("./").endswith(name)]
                if len(matches) != 1:
                    raise ValueError("Ambiguous VSP input path")
                grid = r["map_desc"]
                answer = ",".join(shortest_grid_path(grid))
                self.add("VSP", "maze", f"{len(grid)}-{r['map_id']}", r["text_input"].replace("<image>", ""), answer,
                         "grid", [tar.extractfile(matches[0]).read()], difficulty=len(grid), scoring={"grid": grid},
                         group="map:" + digest(grid), source={"repo": lock["repo"], "revision": lock["revision"], "id": r["map_id"]},
                         metadata={"answer_origin": "deterministic_BFS_from_official_map", "helper_availability": "not_provided_upstream"})
        self.locks["VSP"] = {"repo": lock["repo"], "revision": lock["revision"], "archive": entry}

    def prepared(self, name):
        from datasets import Image as DatasetImage, load_from_disk
        root = ROOT / "benchmark/bench" / name
        source = json.loads((ROOT / "data/reasoning_sources.json").read_text())["sources"][name]
        stamp = json.loads((root / "download_manifest.json").read_text())
        if stamp["revision"] != source["revision"]:
            raise ValueError(f"Prepared version mismatch: {name}")
        split = "testmini" if name == "mathvista" else "test"
        ds = load_from_disk(str(root))[split].cast_column("decoded_image", DatasetImage(decode=False))
        for r in ds:
            if name == "mathvista":
                question = r["query"]
                self.add("MathVista_testmini", r["metadata"]["task"], r["pid"], question, r["answer"], "mathvista",
                         [r["decoded_image"]], options=r["choices"] or [],
                         scoring={k: r[k] for k in ("question_type", "answer_type", "precision")},
                         source={"repo": source["repo_id"], "revision": source["revision"], "id": r["pid"]}, metadata=r["metadata"])
            else:
                self.add("VisuLogic", r["tag"], str(r["id"]), r["question"], r["label"], "choice",
                         [r["decoded_image"]], options=list("ABCD"),
                         source={"repo": source["repo_id"], "revision": source["revision"], "id": r["id"]})
        # Bind every local selected Arrow file, as well as pinned upstream provenance.
        self.locks["MathVista_testmini" if name == "mathvista" else "VisuLogic"] = {
            "repo": source["repo_id"], "revision": source["revision"], "split": split,
            "local_files": [{"path": str(p.relative_to(root)), "sha256": file_sha(p)} for p in sorted((root/split).glob("*")) if p.is_file()],
            "upstream_files": source["files"]}

    def mira(self):
        root = RAW / "MIRA"
        lock = json.loads((root / "download_manifest.json").read_text())
        for e in lock["files"]:
            verify((root / e["path"]).read_bytes(), e)
        for path in sorted(root.glob("*/*.jsonl")):
            task = path.parent.name
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                filename = Path(r["image_path"]).name
                candidates = [p for p in (path.parent/"image").iterdir() if p.stem == Path(filename).stem and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
                if len(candidates) != 1:
                    raise ValueError(f"MIRA image resolution ambiguous: {path}/{r['uid']}")
                image = candidates[0]
                helpers = [p for p in (path.parent/"cot").iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                           and (p.stem == image.stem or p.stem.startswith(image.stem + "_"))]
                helpers.sort(key=lambda p: (int(re.search(r"step(\d+)", p.stem)[1]) if re.search(r"step(\d+)", p.stem) else 0, p.name))
                if not helpers:
                    raise ValueError(f"MIRA helper missing: {task}/{r['uid']}")
                pair = task == "rolling_dice_two"
                answer = f"{r['answer1']}, {r['answer2']}" if pair else r["answer"]
                self.add("MIRA", task, f"{task}/{r['uid']}", r["question"], answer, "mira_pair" if pair else "mira",
                         [image], helpers, scoring={"answer1": r["answer1"], "answer2": r["answer2"]} if pair else {},
                         source={"repo": lock["repo_id"], "revision": lock["revision"], "id": r["uid"], "file": str(path.relative_to(root))})
        self.locks["MIRA"] = lock

    def run(self):
        if (self.target / "manifest.json").exists():
            raise FileExistsError("A published protocol is immutable; use a new output directory")
        for label, method in [("MentisOculi", self.mentis), ("VSP", self.vsp),
                              ("MathVista", lambda: self.prepared("mathvista")),
                              ("VisuLogic", lambda: self.prepared("visulogic")),
                              ("BLINK", self.blink), ("MIRA", self.mira)]:
            method()
            print("CONVERTED", label, len(self.records), "records", len(self.assets), "images", flush=True)
        self.records.sort(key=lambda r: r["id"])
        counts = dict(Counter(r["benchmark"] for r in self.records))
        if counts != EXPECTED or len({r["id"] for r in self.records}) != sum(EXPECTED.values()):
            raise ValueError(f"Unexpected benchmark counts: {counts}")
        (self.target / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records))
        write_json(self.target / "source_lock.json", self.locks)
        write_json(self.target / "donors.json", make_donors(self.records))
        write_json(self.target / "build_report.json", {"status": "built_pending_acceptance", "counts": counts,
            "unique_assets": len(self.assets), "helper_references": sum(len(r["helper_images"]) for r in self.records),
            "metadata_path_repairs": self.repairs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    Builder(args.output).run()
