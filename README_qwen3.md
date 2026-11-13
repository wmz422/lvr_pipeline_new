# 三组四阶段 Flow：配置对照与阶段衔接

当前实验入口和完整启动方式见 [README.md](README.md)。本文件保留原文档路径，说明新的三组实验。

| 实验 | 入口 | 与原标准 Qwen3 四阶段 Flow 的差异 |
|---|---|---|
| 1. Latent 长度 2 | `train_qwen3_latent2.sh` | `num_latents: 8 → 2`；维度 512；其余配方一致 |
| 2. Teacher 仅重构 | `train_qwen3_reconstruction_only.sh` | 两个 Teacher 阶段均 `qa_loss_weight=0`、`description_weight=0`、`flow_weight=0.1`；保留四阶段 |
| 3. Qwen2.5 四阶段 | `train_qwen25_flow_4stage.sh` | 基座、processor 和模型锁改为 Qwen2.5-VL-7B-Instruct；训练目标与原标准 Qwen3 四阶段一致 |

## 阶段与损失

记 `R` 为现有 FLUX Flow matching MSE，`D` 为 Description CE，`Q` 为 QA CE，`A` 为原 latent 对齐目标。

| 阶段 | 更新次数 | 原 Qwen3 基线 | 实验 1：2 latent | 实验 2：Teacher 仅重构 | 实验 3：Qwen2.5 |
|---|---:|---|---|---|---|
| Teacher alignment | 1,346 | D + 0.1R | D + 0.1R | **0.1R** | D + 0.1R |
| Teacher joint | 1,346 | Q + D + 0.1R | Q + D + 0.1R | **0.1R** | Q + D + 0.1R |
| Student warmup | 200 | A | A | A | A |
| Student joint | 1,346 | Q + A | Q + A | Q + A | Q + A |

QA、Description、latent 对齐的权重均为 1，重构权重为 0.1。
四阶段各自创建优化器与学习率调度，后续阶段继承前一阶段模型权重。
Student joint 均不执行 Description；前 200 步 QA 使用 20% Predicted latent，200–499 步 50%，500 步起 100%。
验证与最终 benchmark 的 QA 使用 Predicted latent。

实验 2 的两个 Teacher 阶段虽然目标相同，仍分别运行 1,346 次更新，第二阶段从第一阶段权重开始，
不合并阶段、不跳过任何阶段。两个阶段的训练与验证都不执行 QA 或 Description 解码，
重构梯度传回 Teacher encoder、latent query / mu head 和 Flow mapper；FLUX 保持冻结。
导出的 QA backbone 与 latent projector 保留在 checkpoint 中，后续 Student 阶段照常使用。

实验 1 的 Teacher 和 Student 都从该实验自身的基座 / 导出开始，所有 latent 槽位、query、预测头接口和评测占位均使用 2。
不裁剪或补齐已有 8 latent checkpoint；不重复运行原来的 4、16 latent 实验。
8 latent 对照使用原标准 Flow 的结果。

实验 3 使用锁定的 Qwen2.5-VL-7B-Instruct 与原生 processor。
相对旧 `train_flow.sh` 的 Qwen2.5 三阶段配方，增加 Teacher alignment，并在两个 Teacher 阶段启用固定 Description 监督。
原来的三阶段配置和 checkpoint 不作为这组四阶段实验的续跑源。

## 不变的实验条件

训练 / 验证 **172,224 / 7,571** 条；VStar 去污染、训练集去 SAT；全局 batch **128**；
原图预算 **2,000,000 pixels**，crop/helper **262,144 pixels**；BF16 mixed、seed **42**。
沿用原标准 Flow 的学习率、损失系数、QA 混用、缓存数值精度和梯度缩放。
Teacher 不看问题，Description 只使用 latent 和固定指令。
实验 1、3 复用冻结描述，不生成新标注、不改变训练样本集合。
实验 2 的启动和训练均不依赖 Description sidecar。

默认四卡每卡 batch 为 Teacher 两阶段 8、warmup 8、joint 4；累积为 4、4、8。
更改 GPU 卡数时保持 `GPU 数 × 每卡 batch × 累积 = 128`，每卡 batch 至少为 2，保留 64 条尾批。
Teacher / Student 的 reference accumulation 分别保留 16 / 8。

## 输出和初始化

| 实验 | 本地配置 | 默认输出 |
|---|---|---|
| 1 | `.local/qwen3-latent-2-four-stage.yaml` | `runs/qwen3-flow-4stage-latent-2` |
| 2 | `.local/qwen3-reconstruction-only-four-stage.yaml` | `runs/qwen3-reconstruction-only-4stage` |
| 3 | `.local/qwen25-four-stage.yaml` | `runs/qwen25-flow-4stage` |

每组依次衔接：

```text
锁定基座
  → teacher_alignment/checkpoints/last_full.ckpt
  → teacher_joint/checkpoints/last_full.ckpt
  → teacher_joint/checkpoints/student_source.ckpt（去除 Flow mapper，保留来源记录）
  → warmup/checkpoints/last_full.ckpt
  → joint/checkpoints/last_full.ckpt
  → evaluation/predicted/
```

正式训练从自身第一阶段开始，不使用独立 preflight 权重初始化。
三个入口各自配置、checkpoint、断点恢复和结果相互独立；RGB / FP32 VAE 缓存可共享。
使用自定义 `--output` 时，该组后续所有阶段和评测都会跟随该目录。

## 完整评测、续跑和补评

```bash
REPRO_GPUS=4
bash train_qwen3_latent2.sh --gpus "$REPRO_GPUS"
bash train_qwen3_reconstruction_only.sh --gpus "$REPRO_GPUS"
bash train_qwen25_flow_4stage.sh --gpus "$REPRO_GPUS"
```

每组训练后自动完成 **11 项 benchmark、9,957 条记录、10,107 次生成以及全部评分**。
不向 benchmark 命令传入样本数量限制；评测采用各组最终 Student 和自身第二阶段导出。
每组完成后查看：

```text
evaluation/predicted/summary.json
evaluation/predicted/main_table.md
workflow_status.json
```

`workflow_status.json` 只在训练阶段和完整评分均通过检查后标记 `complete`。

```bash
# 配置预览，三个入口都支持。
bash train_qwen3_reconstruction_only.sh --gpus 4 --dry-run

# 原参数和原输出续跑；每组分别执行。
bash train_qwen3_reconstruction_only.sh --gpus 4 --resume

# 训练已完成，只补完整评测。
bash train_qwen3_reconstruction_only.sh --gpus 4 --eval-only --resume

# 自定义路径从首次运行起保持一致。
bash train_qwen25_flow_4stage.sh --gpus 4 --output runs/my-qwen25-flow
bash train_qwen25_flow_4stage.sh --gpus 4 --output runs/my-qwen25-flow --resume
```

三个脚本均支持 `--gpus`、`--eval-gpus`、各阶段 batch / ZeRO / offload、`--preflight` 和 `--skip-bootstrap`。
完整参数、固定模型版本和 benchmark 列表见 [README.md](README.md)。
