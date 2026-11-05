---
language:
  - en
task_categories:
  - visual-question-answering
pretty_name: LVR five-mode latent ablation benchmarks v1
size_categories:
  - 1K<n<10K
---

# LVR latent ablation benchmarks v1

Frozen evaluation inputs for the five-mode latent ablation in
[wmz422/lvr_pipeline_new, repro/qwen25-flow-20261028](https://github.com/wmz422/lvr_pipeline_new/tree/repro/qwen25-flow-20261028/ablation).
This package contains benchmark inputs and reference answers. It contains no training data or model weights.

| Benchmark | Records | Teacher helpers |
| --- | ---: | --- |
| HRBench4K | 800 | 772 answer-conditioned model-annotated crops; 28 original-image fallbacks |
| VStarBench | 191 | Official bounding-box crops |
| MIRA | 546 | Official ordered helper images |
| MentisOculi | 1250 | Official ordered intermediate-state images |

Total: **2787 records and 6989 distinct image files**. HRBench4K retains the four official option cycles;
800 records are not 800 independent underlying questions. Full declared test scopes are retained,
including known training/development image overlaps; see each record's provenance and source_lock.json.
HRBench4K pseudo-label crops are answer-conditioned; Teacher is an auxiliary-information diagnostic.

Original questions, options, images, reference answers and native scoring states are preserved.
Only originals and questions enter the student. Original images and ordered helpers enter the teacher.
Reference answers and scoring states are exclusively for evaluation.

The package includes the **same global seed-42 one-to-one donor mapping** for Teacher shuffled and
Predicted shuffled. All 2787 records form one pool, without benchmark/task stratification. Self and
same-problem-group donors are excluded; 1894/2787 pairs cross benchmarks. Helpers are physical files,
not links to another machine. Every archive and every extracted file has a SHA256 checksum.

Download and verify with `python3 ablation/download.py` from the linked code repository, or download
all tar files and use `python3 ablation/download.py --source /path/to/downloaded/shards`.
The code release pins the exact dataset revision after publication. No 38-GiB training-data package
is needed for this ablation.

Upstream datasets retain their own terms and attribution:

- HR-Bench: [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench).
- VStarBench: [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench).
- MIRA: [YiyangAiLab/MIRA](https://github.com/YiyangAiLab/MIRA).
- MentisOculi: [Jana-Z/mentis-oculi](https://github.com/Jana-Z/mentis-oculi).

`source_lock.json` records source revisions and hashes. `records.jsonl` records per-question helper
provenance, including `original_fallback`. Bundled `upstream/` files preserve the existing native
scoring implementation and its headers. This aggregate does not assign a new license to upstream assets.
