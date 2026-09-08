# Publishing the release

The code, JSON and download scripts are published at [wmz422/lvr_pipeline_new](https://github.com/wmz422/lvr_pipeline_new). The model bundle is published at [Wing22/lvr-sft-v3-4000](https://huggingface.co/Wing22/lvr-sft-v3-4000). Both repositories are public. Model files, source checkpoint directories, images and runtime logs are excluded from GitHub by `.gitignore`.

## Export

Run with an environment containing the `export` extras. The native source checkpoints must be trusted local files.

```bash
python scripts/export_checkpoint.py \
  --checkpoint /path/to/01-4000.ckpt \
  --lam-checkpoint /path/to/original-lam.ckpt \
  --base-model /path/to/Qwen3-VL-4B-Instruct \
  --train-config /path/to/sft/config.yaml \
  --output /path/to/hf-model-bundle
```

The export includes frozen parameters. BF16 is the default to match the evaluated model's compute dtype. All exported tensors are checked after saving. The independent `lam/lam.ckpt` keeps the original LAM parameter precision and excludes optimizer state. Its parameters are checked against the frozen LAM in the SFT checkpoint after conversion to the selected export dtype.

```text
hf-model-bundle/
  README.md
  lvr_config.json
  model.safetensors.index.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  qwen/                      configuration, tokenizer and processor
  lam/lam.ckpt               standalone LAM model parameters
  export_manifest.json       source and export hashes
```

The original 47.42 GiB ZeRO checkpoint remains local. Uploading only its `model_states.pt` files would not constitute the original checkpoint: reconstruction also needs the optimizer shards. The new bundle avoids that dependency and supports inference with ordinary PyTorch and safetensors.

## Authentication and upload

For an authorized update, log in on the machine instead of adding access tokens to scripts or Git:

```bash
gh auth login
hf auth login
hf upload Wing22/lvr-sft-v3-4000 /path/to/hf-model-bundle .
```

For GitHub, inspect the code/data diff in the prepared release directory and push to `wmz422/lvr_pipeline_new`. The original experiment directory contains unrelated modifications; the publication checkout is separate. The prepared branch is pushed to the repository's default `main` branch without rewriting history.

The main README links to the model files, standalone LAM checkpoint and checksums, and pins the published model revision in its download command. Preserve upstream notices when updating the release. No separate license has been specified for the author's code or exported weights. The bundle is a custom LVR format, not a native Transformers AutoModel checkpoint.
