# 三组四阶段 Flow 实验

本分支当前提供以下三组实验。每组独立训练四阶段，并在最后阶段结束后自动运行完整 benchmark 和评分。
重构 loss 沿用现有 FLUX Flow matching MSE，系数为 **0.1**。

| 实验入口 | 基座 | Latent | Teacher 两阶段目标 | 默认输出 |
|---|---|---|---|---|
| `train_qwen3_latent2.sh` | Qwen3-VL-4B-Instruct | **2 × 512** | 标准 Flow：第一阶段 Description + 重构；第二阶段 QA + Description + 重构 | `runs/qwen3-flow-4stage-latent-2` |
| `train_qwen3_reconstruction_only.sh` | Qwen3-VL-4B-Instruct | 8 × 512 | **两个阶段都仅 0.1 × 重构 loss；QA、Description 均为 0** | `runs/qwen3-reconstruction-only-4stage` |
| `train_qwen25_flow_4stage.sh` | **Qwen2.5-VL-7B-Instruct** | 8 × 512 | 与标准 Qwen3 四阶段 Flow 相同，启用两个 Teacher 阶段的 Description | `runs/qwen25-flow-4stage` |

第一个实验对照原来的 8 latent 标准 Flow，只改变 latent 数量；原有 4、16 latent 实验不在此次运行中重复训练。
第二个实验保留全部四阶段，第三、四阶段继续采用原来的 latent 对齐和 QA 训练配方。
第三个实验使用完整四阶段配方；相对旧 Qwen2.5 三阶段入口增加 Teacher alignment 和 Teacher Description 监督。
详细阶段配置与对照说明见 [README_qwen3.md](README_qwen3.md)。

## 启动三组实验

Linux，兼容的 NVIDIA 驱动，Python 3.12（训练）和 Python 3.13（几何评分）：

```bash
git clone --branch repro/qwen25-flow-20261028 --single-branch \
  https://github.com/wmz422/lvr_pipeline_new.git qwen25-flow-repro
cd qwen25-flow-repro

REPRO_GPUS=4
bash train_qwen3_latent2.sh --gpus "$REPRO_GPUS"
bash train_qwen3_reconstruction_only.sh --gpus "$REPRO_GPUS"
bash train_qwen25_flow_4stage.sh --gpus "$REPRO_GPUS"
```

每条命令包含环境准备、固定版本数据和模型下载、数据校验、RGB / 离线 FP32 VAE 缓存、
四阶段训练与验证、最终导出、完整 11 项 benchmark 推理和评分。
每组评测完成后再执行下一条命令；checkpoint、配置和评测结果使用独立目录，RGB / VAE 数据缓存可以共享。
首次准备成功后记录输入绑定，同一 checkout 后续复用已准备的数据、模型和缓存。

复现参考硬件为 4 × A100 80GB、64 vCPU、450 GiB RAM。固定 PyTorch 使用 CUDA 12.8。
数据包约 38 GiB；RGB / 约 319 GiB VAE 缓存与多阶段 ZeRO checkpoint 需要额外磁盘，建议准备 TB 级空间。
硬件参数由用户指定，不自动搜索配置或在 OOM 后降低 batch。

## 训练配置

| 项目 | 配置 |
|---|---|
| 标准四阶段 | Teacher alignment → Teacher joint → Student warmup → Student joint |
| 各阶段更新 | **1,346 / 1,346 / 200 / 1,346**；两个 Teacher 阶段和 Student joint 各 1 epoch |
| 训练 / 验证 | **172,224 / 7,571** 条；VStar 去污染，训练集去除 5,925 条 `sat_perspective` |
| 全局 batch | **128**；完整 epoch 尾批 64 条，不补齐、不丢弃 |
| 训练图像预算 | 原图 **2,000,000 pixels**，crop/helper **262,144 pixels** |
| Latent | 确定性、每个位置 512 维；实验 1 为 2 个，实验 2、3 为 8 个 |
| Teacher 输入 | 不看问题；Description 解码不输入图片 |
| 损失系数 | QA、Description、latent 对齐均为 1；重构为 0.1；实验 2 的两个 Teacher 文本损失均为 0 |
| 学习率 | 保留标准 Flow 的 backbone / head 学习率、每阶段 warmup 和调度 |
| 精度 / 随机种子 | BF16 mixed、ZeRO-3、seed **42** |
| Flow / VAE | 冻结 FLUX，梯度通过条件传回 Teacher；离线 FP32 VAE，训练时不加载 VAE |
| 序列保护 | 32,768，拒绝超限输入，不截断 |
| 评测 | Predicted latent、原生 base processor、greedy、`max_new_tokens=512`；不额外限制测试分辨率 |

| 阶段 | 默认 GPU | 每卡 batch | 累积 | 全局 batch | 优化器状态 |
|---|---:|---:|---:|---:|---|
| Teacher alignment | 4 | 8 | 4 | 128 | CPU |
| Teacher joint | 4 | 8 | 4 | 128 | CPU |
| Student warmup | 4 | 8 | 4 | 128 | GPU |
| Student joint | 4 | 4 | 8 | 128 | GPU |

始终按 `accumulation = 128 / (GPU 数 × 每卡 batch)` 计算，要求整除且每卡 batch 至少为 2。
Teacher 的梯度缩放 reference accumulation 保留 16，Student 保留 8，均不随硬件参数改变。
不要把参考配方中的 reference batch 改成实际 microbatch；通过启动参数指定实际 batch。
Student joint 保留原来的 QA 混用：前 200 步使用 20% Predicted latent，200–499 步 50%，500 步起 100%。
最终评测全部使用 Predicted latent。

## 硬件、预览和续跑

三个入口接受相同参数，例如：

```bash
# 仅打印配置，不下载、不启动 GPU、不写训练输出。
bash train_qwen3_latent2.sh --gpus 4 --dry-run
bash train_qwen3_reconstruction_only.sh --gpus 4 --dry-run
bash train_qwen25_flow_4stage.sh --gpus 4 --dry-run

# 八卡，显式选择 batch 和 offload；全局 batch 仍为 128。
bash train_qwen25_flow_4stage.sh --gpus 8 \
  --teacher-batch-size 2 --warmup-batch-size 4 --joint-batch-size 2 \
  --zero-stage 3 --teacher-optimizer-device cpu --joint-optimizer-device cpu
```

| 参数 | 作用 | 默认 |
|---|---|---|
| `--gpus N` | 单机训练卡数 | 4 |
| `--teacher-batch-size` / `--warmup-batch-size` / `--joint-batch-size` | 每卡 batch | 8 / 8 / 4 |
| `--zero-stage 2\|3` | 全部阶段的 ZeRO 策略 | 3 |
| `--teacher-zero-stage` / `--warmup-zero-stage` / `--joint-zero-stage` | 单独覆盖某阶段策略 | 跟随 `--zero-stage` |
| `--teacher-optimizer-device` / `--warmup-optimizer-device` / `--joint-optimizer-device` | `cpu` 或 `cuda` | cpu / cuda / cuda |
| `--parameter-offload none\|cpu` | 参数卸载到 CPU 时要求各阶段均为 ZeRO-3 | none |
| `--flow-generator-batch-size` | 冻结 Flow 生成器内部子 batch | 2 |
| `--eval-gpus N` | 完整 benchmark 推理卡数 | `min(训练卡数, 8)` |
| `--output PATH` | 本组实验输出目录 | 见实验表 |
| `--cpu-threads` | 每个训练进程的 CPU 线程数 | 8 |
| `--preflight` | 正式训练前额外执行独立的每阶段 4 步 GPU 预检 | 关闭 |
| `--skip-bootstrap` | 使用本 checkout 已准备的数据、模型和配置 | 关闭 |
| `--data-source PATH_OR_URL` | 指定本地或 HTTPS 数据包 | 固定公开版本 |

`--dry-run` 需要训练依赖已安装；`--help` 不需要安装环境。
可通过 `CUDA_VISIBLE_DEVICES` 选择设备，通过 `LVR_TRAIN_PYTHON`、`LVR_SCORING_PYTHON` 指定已有的匹配环境。
默认 `NCCL_P2P_DISABLE=1` 沿用原参考平台设置；不同平台可在新的 output 中显式调整。

使用原参数、原输出目录续跑：

```bash
bash train_qwen3_latent2.sh --gpus 4 --resume
bash train_qwen3_reconstruction_only.sh --gpus 4 --resume
bash train_qwen25_flow_4stage.sh --gpus 4 --resume

# 已训练完成，只补跑完整评测；也要保留原来的硬件和 output 参数。
bash train_qwen25_flow_4stage.sh --gpus 4 --eval-only --resume
```

自定义输出时，首次运行、续跑、补评测均传相同 `--output`。
模型、损失、代码或输入绑定发生变化会拒绝混用旧运行；新实验不能接续旧三阶段或其他长度的 checkpoint。
Teacher / Student joint 每 400 次更新保存滚动恢复点，673 / 1346 保存候选；warmup 在 100 / 200 保存候选。
若在第一个可恢复 checkpoint 前中断，使用新的 output。

## 完整 benchmark 与结果

每组默认使用自己的 `joint/checkpoints/last_full.ckpt` 和第二阶段导出的 `student_source.ckpt`。
完成全部推理和评分、通过完整性检查后才写入 `workflow_status.json` 的 `complete` 状态。

| Benchmark | 评测记录 | 生成次数 |
|---|---:|---:|
| VStarBench | 191 | 191 |
| HRBench4K | 800 | 800 |
| HRBench8K | 800 | 800 |
| MME-RealWorld-Lite | 1,919 | 1,919 |
| BLINK | 1,901 | 1,901 |
| MentisOculi | 1,250 | 1,250 |
| VSP | 400 | 400 |
| VisuLogic | 1,000 | 1,000 |
| MIRA | 546 | 546 |
| MathVista_testmini | 1,000 | 1,000 |
| SAT | 150 | 300 |
| **每组合计** | **9,957** | **10,107** |

三组合计 30,321 次生成。HRBench 记录已含四轮选项排列；SAT 保留原顺序和反转顺序。
BLINK 取 14 个任务宏平均，Mentis 取 5 个任务均值；MathVista 使用本地数值提取。
答案解析沿用 `answer-parsing-2026-10-26-v2`，拒绝缺题、重复题和不同 checkpoint / 解析配置的混合结果。
训练集的去污染策略与历史重叠说明见 benchmark 的 `audit/overlap_policy.json`。

每个实验目录保存：

```text
workflow_plan.json
recipes/{teacher_alignment,teacher_joint,warmup,joint}.yaml
teacher_alignment/checkpoints/last_full.ckpt
teacher_joint/checkpoints/{last_full,student_source}.ckpt
warmup/checkpoints/last_full.ckpt
joint/checkpoints/last_full.ckpt
evaluation/predicted/summary.json
evaluation/predicted/main_table.md
workflow_status.json
```

## 固定模型与数据

模型下载和校验由清单锁定：

| 模型 | 固定 revision | 清单 |
|---|---|---|
| Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | `manifests/qwen3_model.json` |
| Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | `manifests/qwen_model.json` |
| FLUX.2-klein-base-4B | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | `manifests/flow_model.json` |

数据包：[Wing22/qwen25-flow-repro-data-v1](https://huggingface.co/datasets/Wing22/qwen25-flow-repro-data-v1)，
固定 revision `7068d035c60c09dc1b343438520bdbd12107e0de`。
20 个分片包含原始训练图像、冻结审核后的推理数据、长度清单和完整 benchmark。
`manifests/data_release.json` 和相邻 `.files.jsonl.gz` 分别锁定分片和解包文件。
包内不含模型权重、训练 checkpoint、RGB 或离线 VAE 缓存。
精确复现使用冻结数据包；重新下载公共数据并生成描述不能保证相同样本集合。

实验 1、3 使用 `assets/qwen3_descriptions/captions.jsonl.gz` 中的固定描述和已有推理样本的状态描述，
按样本 ID 绑定，不重新生成标注；实验 2 不加载 Description sidecar。
缓存身份包含路径、GT 协议与实现哈希，换目录后重新生成缓存绑定。

首次自动准备也可拆开执行，以下以实验 3 为例：

```bash
bash tools/setup.sh
(
export LVR_FLOW_VARIANT=qwen25-four-stage
.venv/bin/python reproduce.py configure
.venv/bin/python reproduce.py restore-data
.venv/bin/python reproduce.py download-models
.venv/bin/python reproduce.py prepare-data
.venv/bin/python reproduce.py prepare-benchmarks --verify-only
.venv/bin/python reproduce.py doctor
)
```

公共原始来源、文件校验和处理实现分别见 `manifests/benchmark_downloads.json`、
`tools/download_raw.py`、`tools/prepare_benchmarks.py` 和 `lvr_pipeline_new/scripts/`。
重新获取原图必须符合冻结图像哈希；不会随上游默认 split 改变而换题。

## 代码验证与其他入口

```bash
# 仓库 CPU 回归，包括真实小模型梯度、四阶段训练、导出、重载和生成。
.venv/bin/python reproduce.py configure
.venv/bin/python reproduce.py test
```

当前三组实验的验证记录见 `manifests/three_experiments.json`。
CPU 检查与小模型测试不代表完成正式 4B / 7B GPU 训练或全量 benchmark 推理；实际运行结果以各实验输出为准。

已训练权重的 Teacher / Teacher shuffled / Predicted / Predicted shuffled / Zero 消融、
公开权重和 cosine 复现见 [ablation/](ablation/README.md)。
数据与模型沿用各自上游条款，见 [NOTICE.md](NOTICE.md)。
