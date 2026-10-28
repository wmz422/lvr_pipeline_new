---
pretty_name: Qwen2.5 Plain SFT and Flow reproducibility data v1
license: other
license_name: mixed-upstream-licenses
license_link: https://github.com/wmz422/lvr_pipeline_new/tree/repro/qwen25-flow-20261028
task_categories:
- visual-question-answering
tags:
- reproducibility
- multimodal
- visual-reasoning
size_categories:
- 100K<n<1M
---

# Qwen2.5 Plain SFT / Flow — frozen data v1

This repository contains the exact data snapshot for the Qwen2.5-VL-7B-Instruct
Plain SFT and four-GPU Flow experiment prepared on 2026-10-28.
It is a collection of checksummed restoration archives, not a single Hugging Face
`datasets.load_dataset` table. Use the matching code to restore and process it.

Code: [wmz422/lvr_pipeline_new, branch repro/qwen25-flow-20261028](https://github.com/wmz422/lvr_pipeline_new/tree/repro/qwen25-flow-20261028).

## Contents

The 20 uncompressed tar shards occupy 41,161,256,960 bytes (38.334 GiB),
containing 334,540 files with 40,904,275,724 bytes of payload:

| Group | Files | Contents |
| --- | ---: | --- |
| `legacy_images` | 124,138 | Original training images, including the excluded VStar identities |
| `reasoning_images` | 195,853 | Reviewed reasoning images, including SAT records retained to reproduce the filter |
| `benchmark_images` | 14,408 | Exact benchmark image assets |
| `metadata` | 141 | Original and filtered annotation shards, curation/audit evidence, splits, Plain token lengths, benchmark records/protocol/scorers |

The resulting training split has **172,224 records** and validation has **7,571**.
Processing removes 5,925 SAT training records and 191 VStar-contaminated legacy
images (190 training, one validation). SAT validation remains. There is no new
low-resolution filter. Historical description fields are preserved for provenance;
the current Flow run does not train a description branch.

The benchmark protocol contains 9,957 records and 10,107 generations across
VStarBench, HRBench4K, HRBench8K, MME-RealWorld-Lite, BLINK, MentisOculi, VSP,
VisuLogic, MIRA, MathVista_testmini, and SAT. HRBench includes option permutations;
SAT includes original and reversed orders. The answer parsing template is
`answer-parsing-2026-10-26-v2`.

Model weights, training checkpoints, RGB reconstruction caches, and offline VAE
latents are not included. The code downloads pinned models and rebuilds caches.

## Restore

```bash
git clone --branch repro/qwen25-flow-20261028 --single-branch \
  https://github.com/wmz422/lvr_pipeline_new.git
cd lvr_pipeline_new
bash tools/setup.sh
.venv/bin/python reproduce.py restore-data \
  --source https://huggingface.co/datasets/${DATASET_REPO}/resolve/main
.venv/bin/python reproduce.py prepare-data
.venv/bin/python reproduce.py prepare-benchmarks --verify-only
```

The code's release lock checks every archive and every restored file with SHA256.
Use the immutable dataset commit recorded in the code's publication manifest
instead of `main` when recording an experiment. To restore only annotations,
append `--groups metadata`; image groups can be restored separately.

`data_release.json` fixes archive checksums; `data_release.files.jsonl.gz` fixes
individual file paths, sizes, groups, and hashes. `SHA256SUMS` also checks the
standalone source manifests and dataset card. Downloads and restoration can resume.
Plain's length fingerprint is rebound to the new checkout only after verifying the
historical sample identities and order; all measured lengths stay unchanged.

## Sources and terms

This is a frozen compilation of existing upstream datasets and reviewed annotations.
The component datasets retain their original licenses and restrictions; no single
new license is granted over the combined data. In particular, some sources have
non-commercial or share-alike terms. Public availability does not remove those terms.

- `sources.json`: pinned legacy training and image origins.
- `reasoning_sources.json`: reasoning source revisions and recorded licenses.
- `benchmark_downloads.json`: pinned public benchmark files and integrity metadata.
- The metadata archive includes source locks, historical review evidence, benchmark
  scorer sources, and the license files recorded by the accepted benchmark suite.

Manually reviewed selection and historical annotations are deliberately frozen.
Redownloading public sources and regenerating annotations is not guaranteed to
reproduce the same selected samples. Existing overlap auditing is documented in
the released benchmark metadata; decontamination here specifically targets VStar.

## Validation

The release was hashed file by file. The current benchmark's 14,408 images were
rebuilt from pinned raw sources, and all 10,107 reference answers passed the scorer
round-trip check. These are data and scorer checks, not model accuracy results.
The code passed 81 CPU regression tests with one historical fixture-dependent skip,
and was tested after cloning into a different directory. Full GPU training and
model-generated benchmark inference were not rerun as part of packaging.
