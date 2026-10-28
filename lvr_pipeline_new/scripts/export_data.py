#!/usr/bin/env python3
"""Export the recorded splits and image reconstruction metadata, without images."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def write_shards(rows, directory, max_bytes=25_000_000):
    directory.mkdir(parents=True, exist_ok=True)
    shards, chunk, size = [], [], 2
    def flush():
        if not chunk:
            return
        payload = ("[\n" + ",\n".join(chunk) + "\n]\n").encode()
        filename = f"part-{len(shards):05d}.json"
        (directory / filename).write_bytes(payload)
        shards.append({"file": filename, "records": len(chunk), "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()})
        chunk.clear()
    count = 0
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        n = len(encoded.encode()) + 2
        if chunk and size + n > max_bytes:
            flush()
            size = 2
        chunk.append(encoded)
        size += n
        count += 1
    flush()
    index = {"format": "lvr-json-shards-v1", "records": count, "shards": shards}
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def normalize_image(path):
    if path.startswith("../benchmark/bench/"):
        path = path.replace("../benchmark/bench/", "images/original/benchmark/", 1)
        path = path.replace("/vstar_bench/", "/vstar/", 1)
        path = path.replace("/images/", "/")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.source / "data"
    splits = {"lam/train": "lam/v1/train.json", "lam/test": "lam/v0/test.json",
              "align/train": "align/v0/train.json", "align/test": "align/v0/test.json",
              "align/test_filtered": "align/v1/test.json",
              "sft/train": "llm/v1/train.json", "sft/test": "llm/v1/test.json"}
    pairs, metadata, report = {}, {}, {}
    for name, relative in splits.items():
        raw = (root / relative).read_bytes()
        records = json.loads(raw)
        exported = []
        for record in records:
            q, a = normalize_image(record["question_image"]), normalize_image(record["auxiliary_image"])
            key = (q, a)
            pairs.setdefault(key, set()).add(name.split("/")[0])
            if record.get("box") is not None:
                metadata[a] = [record["box"]]
            clean = {k: record[k] for k in ("question", "answer", "observation", "boxes", "raw") if k in record}
            clean = {"question_image": q, "auxiliary_image": a, **clean}
            exported.append(clean)
        index = write_shards(exported, args.output / name)
        report[name] = {"source": relative, "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "records": len(records), "shards": len(index["shards"])}
        print(name, report[name], flush=True)

    import pandas as pd
    frame = pd.read_parquet(next((root / "raw/TreeVGR-RL-37K").glob("*.parquet")))
    for _, row in frame.iterrows():
        image = row["images"]
        image = image if isinstance(image, str) else list(image)[0]
        image = str(image).removeprefix("images/")
        metadata[f"images/auxiliary/TreeVGR-RL-37K/{image}"] = [list(map(float, x["bbox"])) for x in row["target_instances"]]

    raw = json.loads((root / "raw/seal_vqa_data/with_bbox_191k.json").read_text())
    for index, record in enumerate(raw):
        group = record["image"].split("/")[0]
        boxes = []
        for item in record.get("target_instances", []):
            x, y, w, h = map(float, item["bbox"])
            boxes.append([x, y, x+w, y+h])
        metadata[f"images/auxiliary/{group}/records/{index:06d}.png"] = boxes
    del raw

    raw = json.loads((root / "raw/Visual-CoT/viscot_363k.json").read_text())
    for index, record in enumerate(raw):
        value = record["image"][1]
        boxes = json.loads(value.split("###", 1)[1]) if "###" in value else []
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        metadata[f"images/auxiliary/visual_cot_400k/records/{index:06d}.png"] = boxes
    del raw

    for annotation in (root / "raw/VisDrone").glob("VisDrone2019-DET-*/annotations/*.txt"):
        boxes = []
        for line in annotation.read_text().splitlines():
            fields = line.strip().split(",")
            if len(fields) < 5 or int(float(fields[4])) != 1:
                continue
            x, y, w, h = map(float, fields[:4])
            boxes.append([x, y, x+w, y+h])
        key = f"images/auxiliary/VisDrone/{annotation.parent.parent.name}/{annotation.stem}.png"
        metadata[key] = boxes

    annotation_sources = {}
    for annotation in sorted((root / "images/auxiliary/benchmark/annotations").glob("*.jsonl")):
        for line in annotation.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            key = "images/auxiliary/" + row["output_path"].split("/images/auxiliary/", 1)[1]
            metadata[key] = [item["bbox_xyxy"] for item in row["targets"]]
            annotation_sources[key] = row.get("source")

    image_records = []
    missing = []
    for (q, a), stages in pairs.items():
        if a not in metadata:
            missing.append(a)
            continue
        suffix = q.removeprefix("images/original/")
        group, relative = suffix.split("/", 1)
        source = {"TreeVGR-RL-37K": "treevgr", "Visual_CoT": "monet_visual_cot",
                  "visual_cot_400k": "visual_cot", "VisDrone": "visdrone"}.get(group, group)
        if group == "benchmark":
            source, relative = relative.split("/", 1)
        image_records.append({"question_image": q, "auxiliary_image": a, "source": source,
                              "source_path": relative, "boxes": metadata[a], "stages": sorted(stages),
                              **({"annotation_source": annotation_sources[a]} if a in annotation_sources else {})})
    if missing:
        (args.output / "missing_image_metadata.json").write_text(json.dumps(missing, indent=2))
        raise RuntimeError(f"Missing reconstruction metadata for {len(missing)} image pairs: {missing[:5]}")
    index = write_shards(image_records, args.output / "image_manifest")
    report["images"] = {"pairs": len(image_records), "sources": dict(Counter(r["source"] for r in image_records)),
                        "shards": len(index["shards"])}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["images"]), flush=True)


if __name__ == "__main__":
    main()
