# Qwen3-VL-4B：Flow 与两个消融实验

三个入口使用同一份 V* 去重、训练集去 SAT 的数据：训练 **172,224** 条，验证 **7,571** 条。
基座为锁定版本的 `Qwen/Qwen3-VL-4B-Instruct`；latent 为 `8 × 512`，全局 batch 为 128。
Teacher 不看问题，Description 解码不输入图片。

## 三个脚本

| 脚本 | 训练流程 | 默认输出 |
|---|---|---|
| `train_qwen3_flow.sh` | 四阶段；Teacher 两阶段有 FLUX 重构、Description；最终阶段无 Description | `runs/qwen3-flow-4stage` |
| `train_qwen3_no_flow_loss.sh` | 相同四阶段；Teacher 两阶段的 Flow loss 权重均为 0 | `runs/qwen3-no-flow-loss-4stage` |
| `train_qwen3_no_latent_align.sh` | 复用 Flow 第二阶段导出权重；跳过 warmup，只训练最后阶段的答案 CE | `runs/qwen3-no-latent-align` |

前两个脚本的阶段和训练目标：

| 阶段 | 更新次数 | Flow 实验 | 去 Flow loss 实验 |
|---|---:|---|---|
| 1. Teacher alignment | 1,346 | Description CE + 0.1 × Flow MSE | Description CE |
| 2. Teacher joint | 1,346 | QA CE + Description CE + 0.1 × Flow MSE | QA CE + Description CE |
| 3. Student warmup | 200 | latent 对齐 | latent 对齐 |
| 4. Student joint | 1,346 | QA CE + latent 对齐 | QA CE + latent 对齐 |

Description、QA、latent 对齐的系数均为 1。第 4 阶段不执行 Description。
去 Flow loss 的实验仍计算 Flow MSE 作为诊断，因此也需要 FLUX 权重和离线 VAE 缓存。
它与 Flow 实验独立训练，两个 Teacher 阶段均从对应实验的 checkpoint 衔接。

`train_qwen3_no_latent_align.sh` 默认读取：

```text
runs/qwen3-flow-4stage/teacher_joint/checkpoints/student_source.ckpt
```

它直接从第二阶段导出的权重初始化 latent predictor，不加载 warmup checkpoint。
从第一步开始始终使用 Predicted latent，训练和验证均不读取 Teacher 辅助图，
不执行 Teacher latent 前向、不计算 latent 对齐、Description 或 Flow loss。
答案 CE 的梯度会传回 latent 预测头、query 和语言模型。
前两个实验第 4 阶段保留原有 QA 混用策略：前 200 步 20% Predicted，
200–499 步 50%，500 步起 100%。

## 跑三个实验

在 Linux、NVIDIA GPU、Python 3.12 和 Python 3.13 已准备好的设备上，从仓库根目录执行：

```bash
# 按设备修改；卡数也可以分别传给每条命令。
REPRO_GPUS=4

bash train_qwen3_flow.sh --gpus "$REPRO_GPUS"
bash train_qwen3_no_flow_loss.sh --gpus "$REPRO_GPUS"
bash train_qwen3_no_latent_align.sh --gpus "$REPRO_GPUS"
```

首次运行自动安装缺失环境、恢复固定数据、下载并校验模型。
前两个实验构建并复用同一套 RGB / FP32 VAE 缓存；第三个实验不构建缓存、不运行 FLUX。
原有描述标注随仓库的 `assets/qwen3_descriptions/captions.jsonl.gz` 提供，
按固定样本 ID 筛选并校验，不重新生成描述，不改变样本集合。

**每个脚本默认在训练结束后自动运行完整 11 项 benchmark 和评分。**
评测使用各自最终 `joint/checkpoints/last_full.ckpt`，采用相同的 Predicted 协议；
检查 9,957 条记录、10,107 次生成和全部 11 项评分完整后才标记完成。
查看各输出目录下：

```text
evaluation/predicted/summary.json
evaluation/predicted/main_table.md
workflow_status.json
```

## 调整 GPU 和显存占用

三个脚本均接受 `--gpus N`，评测默认使用 `min(N, 8)` 张卡，
也可以用 `--eval-gpus N` 单独设置。
指定可见 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train_qwen3_flow.sh --gpus 2
```

默认每卡 batch：Teacher 两阶段 8、warmup 8、最终阶段 4；梯度累积自动按全局 batch 128 计算。
显存不足时显式调整 batch 或 offload，例如：

```bash
bash train_qwen3_flow.sh --gpus 8 \
  --teacher-batch-size 2 --warmup-batch-size 4 --joint-batch-size 2 \
  --zero-stage 3 --teacher-optimizer-device cpu --joint-optimizer-device cpu
```

`GPU 数 × 每卡 batch` 必须整除 128，每卡 batch 至少为 2，以保留 64 条的 epoch 尾批。
脚本不会自动搜索配置或在 OOM 后修改 batch。第三个实验只使用最终阶段的硬件参数。

## 自定义路径、预览与续跑

使用自定义 Flow 输出时，第三个实验通过 `--source-run` 指向同一个 Flow 运行目录：

```bash
bash train_qwen3_flow.sh --gpus 4 --output runs/my-flow
bash train_qwen3_no_latent_align.sh --gpus 4 \
  --source-run runs/my-flow --output runs/my-no-latent-align
```

第三个实验要求源运行的第二阶段已完成并导出权重；源运行后续阶段可以继续执行。
它不会修改源权重，也不会重复训练前两个阶段。

预览配置，不下载数据、不启动 GPU：

```bash
bash train_qwen3_flow.sh --gpus 4 --dry-run
bash train_qwen3_no_flow_loss.sh --gpus 4 --dry-run
bash train_qwen3_no_latent_align.sh --gpus 4 --dry-run
```

续跑使用相同参数、输出目录和 `--resume`；硬件配置也保持一致：

```bash
bash train_qwen3_flow.sh --gpus 4 --resume
bash train_qwen3_no_flow_loss.sh --gpus 4 --resume
bash train_qwen3_no_latent_align.sh --gpus 4 --resume
```

训练已经完成、只补跑评测时，加 `--eval-only --resume`。
自定义输出或源运行路径时，续跑和补评测也要传入相同的 `--output` / `--source-run`。
新四阶段配置使用独立输出和本地配置文件，不接续旧三阶段 checkpoint。
