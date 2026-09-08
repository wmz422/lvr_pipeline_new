# LVR Pipeline

基于 Qwen3-VL-4B-Instruct 的 latent visual reasoning。流程包括 LAM 预训练、Alignment、SFT，以及图像问答和 benchmark 评测。

本发布版对应 `sft/v3` 的 `01-4000.ckpt`。模型参数以 BF16 safetensors 发布，另提供保留 FP32 参数的独立 LAM checkpoint。推理通过本仓库的 `load_bundle` 加载完整模型；不需要原始 DeepSpeed 分片或额外的基础模型权重下载。

代码、JSON 和下载脚本位于 GitHub；模型包位于独立的 Hugging Face 仓库。模型仓库 ID 在正式上传时确定。

## 安装

使用 Python 3.12。先安装适合本机 CUDA 的 PyTorch 2.5.1 / torchvision 0.20.1，然后安装项目：

```bash
python -m pip install -e . -c constraints.txt
```

只有训练或导出原始 DeepSpeed checkpoint 才需要额外依赖：

```bash
python -m pip install -e '.[train,export]' -c constraints.txt
```

核心版本见 `constraints.txt`，实际验证环境见 `results/environment.json`。本次使用现有环境离线验证，未重新安装依赖。

## 下载模型与推理

正式发布后，将下面的仓库 ID 替换为模型仓库：

```bash
python scripts/download_model.py --repo-id YOUR_HF_ACCOUNT/lvr-sft-v3-4000
python -m lvr.infer \
  --model-dir models/lvr-sft-v3-4000 \
  --image /path/to/image.jpg \
  --question 'Describe the main objects in the image.'
```

也可以直接使用已导出的本地模型目录：

```python
from lvr.bundle import load_bundle
model, processor = load_bundle("models/lvr-sft-v3-4000", device="cuda:0")
```

模型包使用本项目的 `LatentVLM` 结构，并非可以直接交给 `AutoModel.from_pretrained` 的原生 Transformers 模型。`qwen/` 中包含基础结构配置及已注册 latent token 的 tokenizer/processor；主目录的 safetensors 包含完整 Qwen、LAM 和 projector 参数。

## Benchmark

```bash
python scripts/download_benchmarks.py --list
python scripts/download_benchmarks.py
python -m lvr.evaluate \
  --model-dir models/lvr-sft-v3-4000 \
  --bench-root benchmark/bench \
  --limit 2 --max-new-tokens 64 \
  --output-dir runs/smoke
```

移除 `--limit` 即评测全部样本；正式评测默认 `max_new_tokens=256`。可以用 `--benchmarks vstar mmvp` 选择数据集。评测固定 greedy decoding、4 个 latent 步和 Qwen processor 默认像素预算；完整配置见 `configs/eval.yaml`。

支持 VSTAR、MMVP、BLINK 的 Counting / IQ_Test / Jigsaw / Relative_Reflectance / Spatial_Relation、HR-Bench 4K/8K 和 MME-RealWorld-lite。下载脚本固定来源 commit、文件及校验信息。

**评测范围：** 历史 LAM 预训练数据包含部分 benchmark 图像及辅助框，其中部分框由答案条件生成。因此历史指标不能解释为完全未见测试集上的泛化成绩。MMVP 保留原实现的逐题 accuracy；它不同于要求一对题目同时正确的配对指标。具体来源与计数见 [数据说明](data/README.md)。

## 训练数据与训练入口

仓库只提供 JSON 和图像重建脚本。JSON 中已经保存训练使用的回答、alignment observation 和框，不需要重新调用模型生成标注。

```bash
python scripts/download_data.py --stage sft --list
python scripts/download_data.py --stage sft
# LAM 预训练需要更多原始图像：
python scripts/download_data.py --stage lam
```

图像路径均相对 `data/`。`data/*/train`、`data/*/test` 是有序 JSON 分片目录，训练读取器直接接受目录路径并校验分片 SHA256。数据划分和样本顺序保留自记录的实验。

训练前准备官方基础模型到 `models/Qwen3-VL-4B-Instruct`，或设置 `QWEN_MODEL_PATH` 指向已有目录：

```bash
python scripts/download_model.py --base

# 使用发布的 LAM 权重，从 alignment 开始：
bash scripts/train_align.sh
bash scripts/train_sft.sh

# 如需从 LAM 预训练开始：
bash scripts/train_lam.sh
LAM_CHECKPOINT_PATH=/path/to/trained/lam.ckpt bash scripts/train_align.sh
LAM_CHECKPOINT_PATH=/path/to/trained/lam.ckpt bash scripts/train_sft.sh
```

基础模型固定到记录中的 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，见 `configs/base_model.json`。

默认记录 4 GPU 设置，alignment 到 step 1000，SFT 到 step 4000。可在命令末尾覆盖 Lightning 参数，例如 `--trainer.devices 2`；调整 GPU 数或累积步数会改变有效 batch size。`PYTHON_BIN`、`DATA_ROOT`（图像根目录）、`JSON_ROOT`（默认本仓库的 `data/`）、`QWEN_MODEL_PATH`、`LAM_CHECKPOINT_PATH` 和 `ALIGN_CHECKPOINT_PATH` 均可由环境变量覆盖。

训练入口和配置已经整理，本次没有运行训练，也不承诺重新训练得到逐比特相同的权重。原 checkpoint 有续训历史，精确恢复优化器进度仍需原始 checkpoint；发布的 safetensors 和 LAM 文件只包含模型参数。

## 验证与目录

本次验证包括：所有 safetensors 张量回读、独立 LAM 参数回读和推理、单图推理、6 个 benchmark 各 2 条样本，以及 2 条 VSTAR 与原始 checkpoint 的输出对照。未下载数据、未训练、未重跑全量 benchmark。详见 [验证记录](docs/verification.md)。

```text
src/lvr/       模型、训练、推理、评测、数据读取
src/lam/       LAM 预训练入口
configs/       alignment、SFT、DeepSpeed、评测配置
scripts/       下载、训练、导出与验证入口
data/          训练 JSON、图像来源/框清单、来源版本
benchmark/     benchmark 下载与评测说明
results/       本次小样本验证和历史汇总
```

导出和上传说明见 [发布说明](docs/release.md)。基础模型和第三方数据的来源见 [NOTICE](NOTICE.md)；项目代码和权重许可证由作者在正式发布时确定。
