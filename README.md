# LVR: original + crops → predicted latents

This branch reproduces the **2026-09-16 Qwen3-VL-4B latent prediction experiment**. A teacher A encodes the original image and all annotated crops into eight 512-dimensional vectors. Student B predicts those vectors from the original image and question, then consumes its predictions in a second forward pass to answer.

The repository contains the model, training code, exact QA/bbox splits, recorded Description targets, fixed evaluation protocol, and reference results. Paths are configurable; no cluster account or internal workspace is required. The older LAM/alignment/SFT release is documented in [the legacy guide](docs/legacy_lvr.md).

**Weights and images are not stored in Git.** The commands below download the pinned public base model and source images, then train every stage needed to produce the final checkpoint. The September 16 teacher/student checkpoints have not been published as downloadable model artifacts. A user who already has those checkpoints can skip teacher training and override their original paths as shown below. Training again is not a promise of bitwise-identical weights or scores.

## 1. Install

Linux, Python **3.12**, CUDA-compatible PyTorch. The recorded experiment used **4 GPUs with 178.35 GiB each**. Joint prediction training peaked at **121.65 GiB allocated / 141.54 GiB reserved per GPU**. Do not assume this recipe fits four 80 GiB cards. Microbatch reduction preserves global batch but may still be insufficient; this release uses DDP, not optimizer/model sharding.

```bash
git clone --branch feat/latent-prediction-repro-20260916 https://github.com/wmz422/lvr_pipeline_new.git
cd lvr_pipeline_new
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[prediction]' -c constraints-prediction.txt
bash scripts/latent_prediction/install_vlmeval.sh
```

VLMEvalKit is pinned to commit `34a64e619d5466ab3a900f1720041ea88d4a1ed6`. Its optional model packages may add installation requirements even though our exact-matching evaluation does not use an API judge. Do not install the legacy `constraints.txt` or DeepSpeed extras for this experiment.

## 2. Prepare the exact inputs

```bash
# Official Qwen revision ebb281ec70b05090aa6165b016eac8ec08e71b17.
python scripts/download_model.py --base

# Uses the pinned sources in data/sources.json; only originals are needed.
python scripts/download_data.py --stage sft --originals-only
python scripts/latent_prediction/prepare.py --check-images
```

The repository supplies **122,896 training + 1,242 validation records**, with aligned QA, boxes and generated descriptions. `data/latent_prediction/captions.jsonl.gz` contains the exact recorded descriptions; its compressed and uncompressed SHA256 values are checked by `prepare.py`. No caption generation service is needed. Boxes are pixel coordinates, and all boxes retain their recorded order. See [data provenance](data/latent_prediction/README.md).

Set `HF_HOME` if you want a different Hugging Face cache. Benchmark originals will be downloaded/materialized by the pinned VLMEvalKit under `benchmark/LMUData`, or the directory selected by `LMUData`. Tests use the fixed question IDs in `benchmark/latent_prediction/`; they are subsets of the full upstream benchmarks.

Allow substantial disk space: source image downloads plus several complete checkpoints. The recorded final prediction checkpoint with optimizer state is about **65.9 GiB**, and its warm-up checkpoint about **33 GiB**. Teacher training also saves full checkpoints; reserve several hundred GiB beyond the image datasets.

## 3. Train the full pipeline

Run commands from the repository root. First inspect the resolved recipes without allocating GPUs:

```bash
python scripts/latent_prediction/run.py all --dry-run

# Optional: two optimizer steps and validation per stage in runs/latent_prediction/preflight/.
python scripts/latent_prediction/run.py all --preflight

# All four full stages, followed by the four benchmark conditions.
python scripts/latent_prediction/run.py all --evaluate
```

| Stage | Iterations | What is trained | Training loss |
| --- | ---: | --- | --- |
| `teacher_alignment` | 961 | A language layers, A queries/head, projector, B language layers | Description CE |
| `teacher_joint` | 961 | Same components; initialized from teacher alignment | QA CE + Description CE |
| `warmup` | 200 | New B prediction queries/head only | Relative MSE + cosine |
| `joint` | 961 | B language layers/LM head + prediction queries/head | Relative MSE + cosine + QA CE + Description CE |

A receives **original + all crops**, without question/answer text. Its vision tower is frozen. B's QA branch receives the original image and question; the Description decoding branch receives only latent vectors and a fixed instruction. Both stages of teacher construction update A and B language parameters. During the two prediction stages, the entire trained A and original latent projector are frozen; B's vision tower/merger remain frozen throughout.

B appends eight learnable queries to its image/question prefix and predicts eight vectors in one causal forward pass. Answer tokens are physically removed from that pass. QA/Description losses backpropagate through the predicted vectors and the frozen projector; predictions are not detached. Warm-up retains the query gradient through frozen B.

```text
teacher = stop_gradient(A(original, all_crops))
L_mse = mean(mean_d((pred - teacher)^2) / max(mean_d(teacher^2), 1e-6))
L_cos = mean(1 - cosine(pred, teacher))
L_joint = L_mse + L_cos + QA_CE + Description_CE
```

Each CE is normalized by its own valid target tokens; every joint-training example supplies both QA and Description. There is **no KD, latent sampling or KL**. Joint QA selects whole teacher/predicted latent groups per sample: 20% predicted in iterations 1–200, 50% in 201–500, 100% in 501–961. Description and alignment always use predictions. The schedule is keyed by seed, optimizer step and sample ID.

All stages use global batch **128**, seed 42, BF16 autocast with FP32 trainable parameters/AdamW states, weight decay 0.01, clipping 1.0, 3% LR warm-up, and cosine decay to 10% of the initial LR. Language LR is `1e-5`, heads/queries/projector LR `1e-4` when trainable. Teacher alignment uses batch 8/GPU × accumulation 4; the other stages use batch 4/GPU × accumulation 8. Original/crop pixel budgets are **1,048,576 / 262,144**, text sequence cap 8,192. Each new phase starts a fresh optimizer/scheduler. Full phase settings are in [configs/latent_prediction](configs/latent_prediction).

Monitor `runs/latent_prediction/<stage>/train.log`, `runtime_rank*.json`, and `metrics/csv/metrics.csv`. A successful phase has `train_exit_code=0` and `checkpoints/last.ckpt`. Intermediate prediction checkpoints are saved at iterations 400/800. Saving a full checkpoint on network storage took several minutes in the recorded run.

### Start from an existing teacher or resume

```bash
python scripts/latent_prediction/run.py prediction \
  --source-checkpoint /path/to/original_plus_crops_teacher.ckpt \
  --base-model /path/to/Qwen3-VL-4B-Instruct \
  --image-root /path/to/training_images \
  --output /path/to/run --evaluate

# Repeat the same command and arguments with --resume after an interruption.
python scripts/latent_prediction/run.py all --resume --evaluate
```

Completed stages are reused only with the same configuration. An incomplete phase requires `--resume` and a full checkpoint; otherwise use a new output directory. `--gpus` and `--batch-size` recalculate accumulation to keep global batch 128; a different microbatch/device count changes sampler grouping and numerical execution, so it is a resource adaptation rather than the exact recorded recipe. No cluster submission or resource lifecycle commands are embedded in these local scripts.

## 4. Evaluate or predict a single image

```bash
python scripts/latent_prediction/evaluate.py \
  --checkpoint runs/latent_prediction/joint/checkpoints/last.ckpt \
  --source-checkpoint runs/latent_prediction/teacher_joint/checkpoints/last.ckpt

python scripts/latent_prediction/report_latent_prediction.py runs/latent_prediction

python scripts/latent_prediction/infer.py \
  --checkpoint runs/latent_prediction/joint/checkpoints/last.ckpt \
  --source-checkpoint runs/latent_prediction/teacher_joint/checkpoints/last.ckpt \
  --image /path/to/image.jpg --question 'What is written on the sign?' \
  --save-latents runs/example_latents.pt
```

Use `--base-model` when moving checkpoints to a machine with a different model directory. The current checkpoint loader needs the source teacher and base architecture for reconstruction, even for predicted inference. The **prediction computation** itself uses only original images and the question; it does not run A or read crop images.

Evaluation uses the same final B for `predicted`, `teacher`, `zero`, and `shuffled`, greedy decoding, at most 256 generated tokens, and exact matching. **Teacher means A's true target latents supplied to the final jointly trained B**, not the previous B checkpoint. Shuffling uses fixed whole-latent donors from the same benchmark, with no self-matches. Recorded sample IDs and donor assignments survive relocation of `LMUData`.

## 5. Recorded results and checks

| Benchmark | Plain SFT | Source A+B | Predicted | Teacher (final B) | Zero | Shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VStar (191) | 83.77 | 86.91 | 93.72 | 94.24 | 92.15 | 93.19 |
| HR4K (772) | 64.90 | 66.71 | 65.16 | 68.65 | 66.45 | 65.80 |
| HR8K (756) | 59.79 | 62.43 | 61.77 | 64.42 | 61.51 | 60.98 |
| MME-RealWorld-Lite (1704) | 51.23 | 53.17 | 51.23 | 53.52 | 50.00 | 51.47 |
| BLINK (239) | 65.69 | 64.44 | 62.76 | 64.02 | 61.92 | 64.02 |
| All questions, weighted (3662) | 58.52 | 60.43 | 59.31 | 61.77 | 58.82 | 59.45 |

Scores are percentages. The weighted row is a cross-dataset summary, not an official benchmark aggregate. All conditions used matched prompts/IDs/answers, with no empty, failed or truncated generations. Predicted is slightly above plain SFT, but shuffled does not consistently degrade: **this run does not establish that B effectively uses sample-specific predicted latent information**. The final training validation (step 960) reached cosine 0.72006 and relative MSE 0.46457; all benchmark results use step 961.

[Machine-readable reference results](results/latent_prediction_20260916/metrics.json) · [training curves](results/latent_prediction_20260916/training_curves.png) · [release verification](docs/latent_prediction/verification.md)

```bash
# Small real-Qwen CPU tests; the downloaded processor is used by collation tests.
python -m unittest discover -s tests -p 'test_*.py' -v
```

The published models in the legacy guide are **different experiments** and cannot substitute for this teacher. See [NOTICE](NOTICE.md) and [data provenance](data/README.md) for third-party terms and historical benchmark exposure. This new recipe starts from official Qwen and does not load the legacy LAM. The teacher ablation uses curated benchmark boxes (including answer-conditioned annotations for some sources), so its score measures this annotated-input protocol, not a crop-free deployment setting.

## Code layout

```text
src/lvr/models/{visual_bottleneck,latent_prediction}.py  Teacher and shared-B predictor
src/lvr/data/crop_qa.py                                Crops, QA and Description collation
src/lvr/training/audit.py                              Progress/gradient configuration audit
src/lvr/eval/{vlmeval,latent_prediction}.py             Fixed evaluation adapters
configs/latent_prediction/                            Four phase recipes
scripts/latent_prediction/                            Preparation, training, inference, evaluation
benchmark/latent_prediction/                          Fixed 3,662-question protocol and boxes
data/latent_prediction/                               Checksummed recorded descriptions
```
