# Publishing the prepared release

The GitHub payload is this repository. The Hugging Face payload is the separate directory produced by `scripts/export_checkpoint.py`. Model files, source checkpoint directories, images and runtime logs are excluded by `.gitignore`.

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

Prepare the destination repository IDs, public/private setting and licenses before publication. Log in on the machine instead of adding access tokens to scripts or Git:

```bash
gh auth login
hf auth login
hf upload YOUR_HF_ACCOUNT/lvr-sft-v3-4000 /path/to/hf-model-bundle .
```

For GitHub, inspect the clean code/data diff and push the prepared release branch to the selected repository. The source project currently has remote `wmz422/lvr_pipeline_new`; its working directory contains unrelated experimental modifications, so publishing should use the prepared directory rather than staging that entire working directory.

After publication, replace `YOUR_HF_ACCOUNT/lvr-sft-v3-4000` in the README with the real ID and record the model commit in the download command's `--revision` argument. Add the chosen licenses and preserve upstream notices. The bundle is a custom LVR format, not a native Transformers AutoModel checkpoint.
