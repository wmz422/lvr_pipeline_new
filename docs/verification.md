# Release verification

Verification was performed on 2026-09-08 using the existing local resources and NVIDIA A100-SXM4-80GB GPUs. No training or dataset/model downloads were run. Public dataset file metadata was queried only to pin source revisions and filenames.

| Check | Result | Record |
| --- | --- | --- |
| BF16 safetensors export | 1,602 tensors saved and read back exactly | `results/export_manifest.json` |
| Independent LAM checkpoint | Original parameter precision retained; exact tensor round trip | `results/export_manifest.json` |
| LAM inference | Finite latent output, shape `[4,32]`, exactly matches bundled LAM | `results/lam.json` |
| Single-image question answering | Strict model load and generated answer completed offline | `results/inference.json` |
| Benchmarks | VSTAR, MMVP, BLINK, HR-Bench-4K/8K, MME-RealWorld-lite; 2 examples each | `results/smoke_summary.json` |
| Original checkpoint comparison | Both VSTAR generated strings exactly match the original checkpoint/code | `results/reference.json` |
| JSON integrity | Ordered shard hashes pass; all training image pairs have reconstruction metadata | `results/data_integrity.json` |

The benchmark smoke test used greedy decoding, `max_new_tokens=64`, forced latent termination at four steps, the original system prompt, and the processor's default pixel budget. It exercised inference, answer extraction and per-category aggregation. The tested BLINK examples were from Counting; the remaining BLINK configurations were not inferred during this lightweight check.

These tiny sample results are functionality checks and should not be interpreted as model-quality estimates. `results/historical_step4000.json` separately preserves the pre-existing full evaluation from the original experiment; it was not rerun for this release. That experiment's LAM training includes benchmark-derived images and auxiliary boxes, as documented in `data/README.md`.

## Re-run using already available resources

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m lvr.infer \
  --model-dir /path/to/exported-bundle \
  --image /path/to/existing-image.jpg \
  --question 'Describe the main objects in the image.' \
  --max-new-tokens 64 --output runs/check/inference.json

python -m lvr.evaluate \
  --model-dir /path/to/exported-bundle \
  --bench-root /path/to/existing-benchmarks \
  --limit 2 --max-new-tokens 64 --output-dir runs/check/benchmarks

python scripts/verify_lam.py \
  --model-dir /path/to/exported-bundle \
  --question-image /path/to/original.jpg \
  --auxiliary-image /path/to/auxiliary.jpg \
  --output runs/check/lam.json
```

Source checkpoint comparison is provided by `scripts/verify_reference.py`; use it after the exported benchmark predictions exist. For a comparison against the unmodified source implementation, select that source's `src` directory through `PYTHONPATH`.

No fresh-environment installation, full training, archive download/extraction, exhaustive reconstructed-image comparison or full benchmark rerun was performed. The observed environment is recorded in `results/environment.json`; the GPU's minimum memory requirement was not measured.
