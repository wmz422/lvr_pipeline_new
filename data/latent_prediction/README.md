# Recorded crop descriptions

`captions.jsonl.gz` is a lossless, deterministic gzip copy of the exact Description targets used in teacher construction and latent prediction training. `manifest.json` records compressed and original-byte SHA256 checksums. The 124,138 rows cover all 122,896 training and 1,242 validation samples in the existing `data/sft` and `data/align` splits.

Each row contains `id`, `split`, `caption`, `source`, `crop_count`, and ordered pixel `boxes`. Sample IDs are `question_image:raw.index`. Descriptions were generated per annotated bbox crop and assembled in crop order. They describe visible content and are reused unchanged; reproduction does not regenerate them or substitute alignment observations.

Run `python scripts/latent_prediction/prepare.py --check-images` after rebuilding original images with `python scripts/download_data.py --stage sft --originals-only`. The check verifies checksums, QA/box order, caption coverage, box equality, duplicate IDs, and original-image availability.

Images and source datasets retain their upstream terms. The sources and pinned revisions are documented in [../sources.json](../sources.json) and [../README.md](../README.md). These two teacher stages start from official Qwen, rather than the legacy LAM checkpoint. Benchmark boxes are released separately under `benchmark/latent_prediction`, including their original `annotation_source` provenance. Answer-conditioned boxes are privileged evaluation inputs and must not be interpreted as deployable ground truth localization.
