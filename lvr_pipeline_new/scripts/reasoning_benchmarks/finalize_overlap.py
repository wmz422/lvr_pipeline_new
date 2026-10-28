"""Apply explicit image reviews and cached geometry identities to suite selection."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from lvr.data.records import read_records
from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT, file_sha, make_donors
from download import write_json


def main():
    root = DEFAULT_SUITE
    if (root/"manifest.json").exists():
        raise FileExistsError("Do not change a published benchmark selection")
    rows = [json.loads(line) for line in (root/"records.jsonl").read_text().splitlines()]
    selection = json.loads((root/"selection.json").read_text())
    excluded = set(selection["excluded_ids"])
    review = json.loads((root/"overlap_visual_review.json").read_text())
    assert review["records_sha256"] == file_sha(root/"records.jsonl")
    confirmed = [r for r in review["decisions"] if r["decision"] == "confirmed_same_source_image"]
    excluded.update(r["id"] for r in confirmed)
    path = ROOT/"data/processed/reasoning_build/quality/mentis_geometry.json"
    geometry = json.loads(path.read_text())["records"]
    signatures = defaultdict(list)
    missing = []
    training = []
    for split in ("train", "val"):
        for row in read_records(ROOT/"data/processed/reasoning_v1"/split):
            if row["source"] not in ("mentis_hinge-folding", "mentis_paper-fold", "mentis_rushhour"):
                continue
            key = "candidate:"+row["id"]
            if key not in geometry:
                missing.append(row["id"]); continue
            signatures[(row["source"].removeprefix("mentis_"), geometry[key]["signature"])].append({"id": row["id"], "split": split})
    structural = []
    reference_checked = 0
    for row in rows:
        if row["benchmark"] != "MentisOculi" or row["task"] not in ("hinge-folding", "paper-fold", "rushhour"):
            continue
        key = f"reference:official_mentis:{row['task']}:" + row["source"]["path"].split(row["task"]+"/")[1]
        if key not in geometry:
            raise ValueError(f"Official structure missing: {key}")
        reference_checked += 1
        matches = signatures.get((row["task"], geometry[key]["signature"]), [])
        if matches:
            excluded.add(row["id"])
            structural.append({"id": row["id"], "kind": "cached_task_geometry_signature", "matches": matches,
                               "method": geometry[key]["method"]})
    report = {"cache_sha256": file_sha(path), "official_checked": reference_checked,
              "published_source_records_missing_signature": missing, "matches": structural,
              "scope": "Cached color-independent Hinge per-piece geometry, Paper final-hole layouts, and full Rush Hour board/object state. Not a proof of arbitrary semantic independence."}
    write_json(root/"overlap_structures.json", report)
    clean = [r for r in rows if r["id"] not in excluded]
    selection.update(excluded_ids=sorted(excluded), clean_counts=dict(Counter(r["benchmark"] for r in clean)),
        confirmed_near_image_records=len(confirmed), structural_overlap_records=len(structural),
        policy="Conservative exclusion of exact input/helper pixel or map identity with train/development; visually confirmed recompressed/resized source images; matching available task-geometry signatures. Other perceptual candidates remain unconfirmed. This selection does not certify all semantic independence.")
    write_json(root/"selection.json", selection)
    write_json(root/"donors.json", {"full": make_donors(rows), "clean": make_donors(clean)})
    print(json.dumps({"clean_counts": selection["clean_counts"], "excluded": len(excluded),
                      "visual_confirmed": len(confirmed), "structural": len(structural), "missing_signatures": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
