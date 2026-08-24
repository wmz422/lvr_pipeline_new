# LAM evaluation experiments

The LAM evaluations are separated into two independent experiment folders.

## 1. Reconstruction comparison

Directory: `reconstruction_comparison/`

This experiment compares Qwen3-VL answers produced from:

1. question feature + real auxiliary-image feature;
2. question feature + LAM-reconstructed auxiliary-image feature.

Both branches pass final pre-merger vision states through the same frozen Qwen
vision merger and disable DeepStack, so the reconstructed branch cannot leak
real auxiliary-image intermediate features.

Run the default step-20000 checkpoint on 100 test cases:

```bash
cd /data/private/wmz/lvr_pipeline_new
PYTHONPATH=src /data/private/wmz/tools/miniforge3/envs/lvr/bin/python \
  ex/test_lam/reconstruction_comparison/compare_reconstruction.py \
  --limit 100 --overwrite --device cuda:0
```

Run the same comparison in parallel on eight GPUs (each GPU writes a separate
resumable shard; successful shards are merged in dataset-index order):

```bash
cd /data/private/wmz/lvr_pipeline_new
PYTHONPATH=src /data/private/wmz/tools/miniforge3/envs/lvr/bin/python \
  ex/test_lam/reconstruction_comparison/run_parallel_reconstruction.py \
  --checkpoint runs/lam/v1/checkpoints/lam-epoch=003-step=00035000-val_loss=0.00000.ckpt \
  --output ex/test_lam/reconstruction_comparison/results_step35000_redboxes.jsonl \
  --limit 100 --overwrite
```

The default prompt reports newly drawn red rectangular outlines in the second
image relative to the first, with a numbered description of the object each
outline highlights. The default generation limit is 384 tokens.

Results are written to:

```text
ex/test_lam/reconstruction_comparison/results_step20000.jsonl
```

Without `--overwrite`, completed dataset indices are skipped so interrupted
runs can resume. Use `--checkpoint` and `--output` together when evaluating a
different checkpoint.

Start the visualization app:

```bash
cd /data/private/wmz/lvr_pipeline_new
/data/private/wmz/tools/miniforge3/envs/lvr/bin/streamlit run \
  ex/test_lam/reconstruction_comparison/app.py \
  --server.address 0.0.0.0 --server.port 8502
```

The app displays the original image, auxiliary image, prompt, both generated
answers, pre-merger metrics, merged metrics, and latent statistics for every
case.

Run unit tests:

```bash
cd /data/private/wmz/lvr_pipeline_new/ex/test_lam/reconstruction_comparison
/data/private/wmz/tools/miniforge3/envs/lvr/bin/python -m unittest \
  test_compare_reconstruction.py
```

## 2. Adjacent latent cosine

Directory: `latent_cosine/`

This experiment computes the three adjacent cosine similarities among the
four deterministic posterior-mean latent tokens:

```text
cos(z[0], z[1])
cos(z[1], z[2])
cos(z[2], z[3])
```

Each position is averaged independently over the complete test set. Run it
with:

```bash
cd /data/private/wmz/lvr_pipeline_new
PYTHONPATH=src /data/private/wmz/tools/miniforge3/envs/lvr/bin/python \
  ex/test_lam/latent_cosine/latent_step_cosine.py --device cuda:0
```

The default output is:

```text
ex/test_lam/latent_cosine/latent_step_cosine_step17000.json
```

Both experiments require a free GPU. The reconstruction comparison loads a
full Qwen3-VL language model in addition to LAM and therefore needs
substantially more memory.
