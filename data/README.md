# Training data

Only JSON annotations and image reconstruction metadata are distributed. Images are downloaded from the upstream sources listed in `sources.json`.

## Recorded splits

| Stage/split | Records | Original input |
| --- | ---: | --- |
| lam/train | 641,682 | `lam/v1/train.json` |
| lam/test | 1,000 | `lam/v0/test.json` |
| align/train | 122,896 | `align/v0/train.json` |
| align/test | 1,242 | `align/v0/test.json` |
| align/test_filtered | 1,062 | `align/v1/test.json` |
| sft/train | 122,896 | `llm/v1/train.json` |
| sft/test | 1,242 | `llm/v1/test.json` |

The `test` files served as validation inputs in the recorded training runs. Alignment uses the original v0 split by default. The separately supplied `align/test_filtered` is the later filtered split and is not silently substituted into the historical recipe.

Each directory contains an `index.json` with ordered shard names, counts and SHA256 hashes. The loader verifies those hashes. Exporting preserves sample order, questions, answers and observations; obsolete crop paths and unused experiment fields are omitted. Benchmark image references formerly pointing outside `data/` are normalized to `images/original/benchmark/`.

## Image reconstruction

```bash
python scripts/download_data.py --stage sft --list
python scripts/download_data.py --stage sft
python scripts/download_data.py --stage lam
```

The `image_manifest` JSON covers 730,068 unique original/auxiliary image pairs. Every training pair has a source, source image path and pixel-space boxes. Existing alignment observations are included in the JSON and are not regenerated. Red boxes reproduce the recorded renderer: width based on 1% of the shorter image side, clipped for small boxes; JPEG quality 95 without chroma subsampling, or PNG compression level 1.

| Image source | Pairs across released splits |
| --- | ---: |
| visual_cot | 404,109 |
| coco2014 | 19,732 |
| gqa | 102,035 |
| treevgr | 36,752 |
| coco2017 | 70,215 |
| mme_realworld_lite | 1,919 |
| visdrone | 7,019 |
| hr_bench_8k | 200 |
| blink | 263 |
| hr_bench_4k | 200 |
| vstar | 238 |
| monet_visual_cot | 87,386 |

SFT/alignment use TreeVGR and the Visual_CoT image subset from Monet-SFT-125K. LAM additionally uses Visual-CoT, COCO/GQA (SEAL VQA boxes), VisDrone, and benchmark-derived pairs. LAM downloads are large: the original Visual-CoT tar parts alone total about 139 GB. `--stage sft` does not download those LAM-only sources.

## Benchmark exposure

The released LAM train+validation data contains 1,919 MME-RealWorld-lite pairs, 238 VSTAR-family pairs (including OCR/GPT4V-hard), 200 HR-Bench-4K pairs, 200 HR-Bench-8K pairs and 263 BLINK pairs. Several auxiliary boxes are from answer-conditioned Qwen annotations. These facts are preserved in `annotation_source` fields. Historical benchmark numbers therefore describe this recorded experiment, not a fully held-out evaluation.

No new downloads or full image reconstructions were executed during release preparation. Public repository file metadata was checked, JSON hashes were verified, and all split image references were checked against the reconstruction manifest. The archive extraction paths and final reconstructed image equality remain untested under the requested lightweight verification scope.
