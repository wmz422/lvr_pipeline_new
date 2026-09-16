# Reproduction release verification

## Recorded experiment

The reference experiment completed teacher Description alignment (961 optimizer updates), teacher QA+Description training (961), prediction warm-up (200), and prediction joint training (961). The prediction workflow exited successfully. All four final evaluation conditions completed the same 3,662 questions, with no failed/empty/truncated generations. The final prediction checkpoint's optimizer moments were FP32 and its scheduler/global step were 961.

The teacher score uses target latents with the **final student B**. It is separate from the preceding source A+B score. The final training validation is step 960; benchmark generation uses step 961. See `results/latent_prediction_20260916/metrics.json` and the root README for scores and interpretation.

## Publication changes

The loss, attention, prompt construction, optimizer, sampling curriculum, teacher/prediction gradient boundaries and decoding computation are preserved. The changes for this branch are:

- Package the shared training audit under `lvr.training`, with portable configurations and local launchers.
- Read the exact caption bytes from a checksummed gzip archive.
- Resolve benchmark image locations against `LMUData` while preserving all recorded IDs and shuffle donors.
- Allow explicit teacher/base-model paths when loading a moved checkpoint.
- Add complete setup, training, resume, inference and evaluation instructions.

`recipe_audit.json` compares every explicitly configured non-path model/data/trainer field against the four recorded resolved training configurations. All global batches remain 128. `protocol_audit.json` verifies all 3,662 prompts, boxes, indices, sample IDs and donor assignments against the recorded run. `reproduction_inputs.json` records SHA256 values for the released data and configurations.

## Release checks

- Exact compressed/uncompressed caption hashes and all 124,138 QA/box/description pairings verified.
- All 124,138 required original-image paths verified against the existing reconstructed image directory.
- Four-phase dry run, Python syntax, shell syntax and wheel build checked.
- The wheel was opened and its prediction model, teacher model, collator, evaluation adapter and training-audit modules compared byte-for-byte with the release source.
- CPU regression suite: **45 tests passed** in 25.803 seconds (excluding import/startup time), including actual tiny-Qwen training/generation, teacher freeze/gradient boundaries, answer removal, checkpoint relocation and benchmark path relocation.
- The portable reporting script was run on the recorded full prediction/teacher/zero/shuffled outputs and reproduced all reference scores.

The release was validated with the recorded installed Python 3.12 / PyTorch 2.8 / Transformers 5.15 environment. It did not repeat the expensive GPU training or re-download every public image archive. GPU probes, DDP preflight and complete training/evaluation refer to the recorded experiment. The environment also contains an unused video `decord` distribution that `uv pip check` flags as a foreign-platform build; image-only tests and this protocol do not use it. A fresh installation is instructed to resolve VLMEvalKit dependencies from its pinned upstream source.

This branch provides a reproducible **training recipe and data/protocol**, not downloadable September 16 model weights. Repeated training can differ numerically with hardware, software or batch-layout changes. No statistical significance or sample-specific latent-use claim is inferred from this single run.
