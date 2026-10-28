# Qwen2.5 Plain SFT / Flow 可复现实验

本仓库整理了 2026-10-28 正在使用的训练、离线 VAE 缓存和统一评测流程。
默认 Flow 配方对应本次四卡实验；同时提供从同一基座开始的两轮 Plain SFT 入口。
代码是独立副本，不会读写原实验的训练目录。

## 固定的实验条件

| 项目 | 配置 |
| --- | --- |
| 基座 | Qwen/Qwen2.5-VL-7B-Instruct，revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Flow generator | FLUX.2-klein-base-4B，revision `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| 训练 / 验证 | 172,224 / 7,571 条；只从训练集去掉 5,925 条 `sat_perspective` |
| 训练图像 | 原图预算 2,000,000 pixels，crop/helper 262,144 pixels；没有新增低分辨率过滤 |
| Teacher | 1 epoch = 1,346 optimizer updates；QA CE + 0.1 × Flow MSE |
| Warmup | 200 updates；latent alignment |
| Joint | 1 epoch = 1,346 updates；QA + latent alignment |
| latent | 8 × 512，确定性；Teacher 不看问题，不执行 description 分支 |
| VAE | 离线 FP32；训练时不加载 VAE，冻结 FLUX 仍对输入条件反传 |
| 训练精度 | BF16 mixed、ZeRO-3、seed 42 |
| 评测 | 原生 base processor，greedy，max_new_tokens=512；不额外限制测试分辨率 |

| 阶段 | GPU | 每卡 microbatch | 累积 | 全局 batch | AdamW 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher | 4 | 8 | 4 | 128 | CPU |
| Warmup | 4 | 8 | 4 | 128 | GPU |
| Joint | 4 | 4 | 8 | 128 | GPU |
| Plain | 4 | 16 | 2 | 128 | GPU |

每个完整 epoch 最后一次更新包含 64 条，其他更新为 128 条，不补齐、不丢弃。
保留当前 Flow 梯度缩放：Teacher 的 reference accumulation=16，student=8。
**不要直接把 `configs/experiment.yaml` 的 reference batch 改成实际 microbatch**；启动器通过参数选择实际 batch。
Teacher 的序列长度 32,768 是拒绝超限的保护值，不截断输入；Plain 为 8,192。

## 仓库与数据包

Git 保存代码、配方、模型版本、原始来源和逐文件校验清单；大文件位于 `artifacts/data-v1/`，不进入 Git。
配套数据包约 38 GiB，包含原始训练图像、冻结的审核后推理数据、筛选证据、长度清单以及完整 benchmark。
`manifests/data_release.json` 锁定每个 tar 分片；相邻 `.files.jsonl.gz` 锁定每个解包文件。

精确数据复现必须取得配套数据包。当前数据经过人工审核和历史标注，重新下载公共数据并重新生成描述不能保证相同样本集合。
冻结版本保留原始记录与顺序；本次训练不消费其中的 description 字段。
Plain 的实测 token 长度清单包含路径指纹；`prepare-data` 先验证历史指纹，再仅重新绑定新目录，所有长度值保持不变。
公开数据集：[Wing22/qwen25-flow-repro-data-v1](https://huggingface.co/datasets/Wing22/qwen25-flow-repro-data-v1)。
20 个分片已全部上传并通过公开访问、文件大小和 SHA256 校验。
另已在新目录从公开固定版本恢复元数据分片，141 个文件全部通过校验。
固定的数据集版本为 `7068d035c60c09dc1b343438520bdbd12107e0de`，完整发布记录见 `manifests/publication.json`。
默认恢复命令使用该固定版本；也可以用 `--source /path/to/data-v1` 指向本地数据包。
本机另有 `artifacts/qwen25-flow-repro.bundle` 代码仓库备份；从 GitHub 克隆时无需该文件。

数据包不含模型权重、训练 checkpoint、RGB 重建缓存或约 319 GiB 的离线 VAE 缓存。
这几项分别由固定版本下载、训练和预处理生成。缓存身份包含路径与实现哈希，换目录后应重新生成，不复用旧实验的 cache plan。

## 一条命令启动 Flow 训练、验证和评测

在 Linux 上准备 Python 3.12、Python 3.13 和兼容的 NVIDIA 驱动后：

```bash
git clone --branch repro/qwen25-flow-20261028 --single-branch \
  https://github.com/wmz422/lvr_pipeline_new.git qwen25-flow-repro
cd qwen25-flow-repro
bash train_flow.sh --output runs/flow-4gpu
```

首次启动会安装缺失的仓库虚拟环境，下载并校验固定版本的数据和两个模型，
重新执行数据筛选，构建 RGB / FP32 VAE 缓存，然后运行 Teacher → Warmup → Joint。
每阶段保留验证；Joint 完成后自动导出并运行完整 11 项 benchmark 和评分。
准备成功后会记录输入绑定，后续使用同一仓库恢复时复用已准备的数据、模型和缓存。
原始训练逻辑、分辨率、数据、损失权重和答案提取规则保持不变。

硬件参数由你手动指定，脚本不做自动测速、配置搜索或 OOM 后的自动降 batch。
例如四卡配置完整写法为：

```bash
bash train_flow.sh \
  --gpus 4 \
  --teacher-batch-size 8 --warmup-batch-size 8 --joint-batch-size 4 \
  --zero-stage 3 \
  --teacher-optimizer-device cpu \
  --warmup-optimizer-device cuda --joint-optimizer-device cuda \
  --output runs/flow-4gpu

# 八卡的另一组合法 batch；评测默认使用八卡。
bash train_flow.sh \
  --gpus 8 \
  --teacher-batch-size 4 --warmup-batch-size 4 --joint-batch-size 2 \
  --zero-stage 3 --warmup-zero-stage 2 \
  --output runs/flow-8gpu
```

**全局 batch 固定为 128**：每个阶段自动计算 `accumulation = 128 / (GPU 数量 × 单卡 batch)`。
必须整除，并且单卡 batch 至少为 2，以保留 Plain 的 64 条 epoch 尾批。
例如四卡 Teacher 为 `4 × 8 × 4 = 128`，八卡 Teacher 为 `8 × 4 × 4 = 128`；
Teacher / Joint 始终各 1,346 次更新，Warmup 始终 200 次。
同时保留 Teacher=16、student=8 的原始梯度缩放参照，不随硬件参数改变。

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `--gpus` | 单机训练卡数；必须能按固定 batch 和尾批分配 | 4 |
| `--teacher-batch-size` / `--warmup-batch-size` / `--joint-batch-size` | 各阶段单卡 batch | 8 / 8 / 4 |
| `--zero-stage 2\|3` | 全部阶段的 ZeRO 策略 | 3 |
| `--teacher-zero-stage` / `--warmup-zero-stage` / `--joint-zero-stage` | 单独覆盖某阶段的 ZeRO 策略 | 跟随 `--zero-stage` |
| `--teacher-optimizer-device cpu\|cuda` 等三个阶段参数 | 优化器放 CPU 或 GPU | cpu / cuda / cuda |
| `--parameter-offload none\|cpu` | 是否把模型参数卸载到 CPU；cpu 要求各阶段均为 ZeRO-3 | none |
| `--flow-generator-batch-size` | 冻结 Flow 生成器内部子 batch，不改变总 batch | 2 |
| `--eval-gpus` | benchmark 推理卡数 | `min(训练卡数, 8)` |
| `--cpu-threads` | 每个训练进程的 CPU 线程数 | 8 |

ZeRO-2 分片优化器状态和梯度，ZeRO-3 额外分片模型参数；CPU offload 用主机内存换取显存空间。
具体选择取决于显存和互联条件，参数含义见 [DeepSpeed 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)。
这些选择不会修改 batch、损失系数或精度，实际是否更快由平台决定。

```bash
# 只打印实际卡数、batch、累积、ZeRO/offload 和步数，不下载、不启动 GPU。
bash train_flow.sh --gpus 2 --teacher-batch-size 4 \
  --warmup-batch-size 4 --joint-batch-size 2 --dry-run

# 使用相同硬件参数、相同 output 恢复。
bash train_flow.sh --output runs/flow-4gpu --resume
```

这个入口默认直接开始正式训练，不自动进行 GPU 试跑；如需独立的每阶段 4 步预检，显式加 `--preflight`。
预检不参与正式训练初始化，也不会搜索或改变硬件参数。
数据和模型已通过下方手动流程准备时，可以加 `--skip-bootstrap` 跳过下载准备。
本地数据包可通过 `--data-source /path/to/data-v1` 指定。
`--help` 无需安装环境；`--dry-run` 需要训练依赖已安装。
恢复时保留原有参数；更改 GPU、batch 或 ZeRO 策略应使用新的 output。

## 手动准备环境和数据

以下是把自动准备流程拆开执行的方式，也适用于只下载、检查数据。

Linux；训练 Python 3.12，几何评分 Python 3.13。复现参考平台为 4 × A100 80GB、64 vCPU、450 GiB RAM。
需要 NVIDIA 驱动支持所固定的 PyTorch CUDA 12.8 版本。请为 RGB/VAE 缓存和多份原生 ZeRO checkpoint 准备 TB 级磁盘。
Teacher 单个原生 checkpoint 约 163 GiB；训练/导出与预检还会额外占空间。

```bash
git clone --branch repro/qwen25-flow-20261028 --single-branch \
  https://github.com/wmz422/lvr_pipeline_new.git qwen25-flow-repro
cd qwen25-flow-repro
bash tools/setup.sh

# 从公开数据集的固定版本下载并逐文件校验、恢复。
.venv/bin/python reproduce.py restore-data
# 如需显式传入地址，使用固定版本：
# .venv/bin/python reproduce.py restore-data --source https://huggingface.co/datasets/Wing22/qwen25-flow-repro-data-v1/resolve/7068d035c60c09dc1b343438520bdbd12107e0de

# 按锁定 revision 下载两个模型，并校验所有权重/配置文件。
.venv/bin/python reproduce.py download-models

# 重新执行 VStar 去污染与训练集去 SAT，逐字节比对冻结版本。
.venv/bin/python reproduce.py prepare-data
.venv/bin/python reproduce.py prepare-benchmarks --verify-only
.venv/bin/python reproduce.py doctor
.venv/bin/python reproduce.py test
```

`setup.sh` 只创建仓库内两个独立环境，不继承旧机器的 `.pth` 或虚拟环境。
可用 `REPRO_TRAIN_PYTHON`、`REPRO_SCORE_PYTHON` 指定对应 Python 解释器。
训练顶层依赖与传递依赖约束在 `requirements/`，完整原环境版本清单在 `manifests/environment_*.json`。
模型下载沿用 Hugging Face 的标准认证和 `HF_ENDPOINT` 环境设置，不在仓库内保存令牌。

配置会在首次 setup 时写入 `.local/`。直接复用已安装环境时先执行 `python reproduce.py configure`；
可通过 `LVR_TRAIN_PYTHON`、`LVR_SCORING_PYTHON` 指定解释器，但版本必须通过 doctor。
运行命令不依赖当前 shell 的工作目录。

## Flow 训练

```bash
# 不使用 GPU：打印实际三阶段配方、步数和评测命令。
.venv/bin/python reproduce.py train --dry-run

# CPU 配方解析、真实 processor/小样本 FP32 VAE 检查。
.venv/bin/python reproduce.py cpu-check --output runs/flow

# 默认用四张卡构建完整 RGB 和离线 FP32 VAE 缓存；也可以 --cache-device cpu。
.venv/bin/python reproduce.py cache --output runs/flow

# 每阶段 4 个更新、验证、checkpoint/导出预检；使用独立预检权重。
.venv/bin/python reproduce.py preflight --output runs/flow

# 自动预检 → Teacher → Warmup → Joint → 完整 11 项评测。
.venv/bin/python reproduce.py train --output runs/flow --resume
```

预检不作为正式 Teacher 的初始化。正式 Teacher 从锁定基座开始；Warmup 使用正式 Teacher 导出，Joint 使用正式 Warmup。
中断后用同一条带 `--resume` 的命令恢复；代码、数据、参数绑定变化会拒绝混用。
若在首个可恢复 checkpoint 之前中断，原有逻辑要求使用新的 output。
只训练不评测可加 `--train-only`；完成后用 `train --eval-only --resume --output ...` 补评。

Teacher / Joint 每 400 次 optimizer update 保存滚动恢复点，保留最近一个；673 / 1346 保存候选。
Warmup 在 100 / 200 保存候选。默认输出有 `workflow_plan.json`、`recipes/`、各阶段 `train.log`、
`runtime_rank*.json`、CSV/TensorBoard、`checkpoints/` 和 `evaluation/predicted/`。
`tuning/backup_checkpoints.py` 与 `verify_backup.py` 保留了独立备份工具，可按其 `--help` 指定源和目的目录。

`NCCL_P2P_DISABLE=1` 是本次无 NVLink 平台的实际设置。其他硬件可显式设置为 0；这属于平台运行设置变化，应在新 output 中记录。
本仓库没有自动改变 batch、精度、图像尺寸或跳过失败题目的逻辑。

## Plain、Base 与 Flow 的统一评测

```bash
# 从相同基座开始复现 Plain 两轮，保留原 2,784 步 LR horizon / 83 步 warmup。
.venv/bin/python reproduce.py plain --dry-run
.venv/bin/python reproduce.py plain --output runs/plain --epochs 2

# 原生 ZeRO 目录合并成评测需要的完整 Lightning 文件。
.venv/bin/python reproduce.py export \
  --checkpoint runs/plain/plain_sft/checkpoints/candidate-001346.ckpt \
  --output runs/plain/exports/epoch1.ckpt

.venv/bin/python reproduce.py evaluate --mode base --output results/base
.venv/bin/python reproduce.py evaluate --mode plain \
  --checkpoint runs/plain/exports/epoch1.ckpt --expected-step 1346 --output results/plain_epoch1
.venv/bin/python reproduce.py evaluate --mode predicted \
  --checkpoint runs/flow/joint/checkpoints/last_full.ckpt \
  --source-checkpoint runs/flow/teacher_joint/checkpoints/student_source.ckpt \
  --expected-step 1346 --output results/flow

# 原始输出离线重评分，不重新推理。
.venv/bin/python reproduce.py score --run results/flow --output results/flow_rescored
```

Plain 第 2 轮候选为 `candidate-002692.ckpt`。训练入口可用 `--resume` 指向同一 output 内完整原生 checkpoint。
历史 Plain 曾在 696 步暂停后续训；这里从基座运行相同数据、长度分桶、尾批策略与 LR 日程，
不依赖那份历史 checkpoint，也不声称与中断续训结果逐位一致。Flow 的 LR warmup/horizon 与 Plain 不同，保留本次实际设置。

| Benchmark | 评测记录 | 生成次数 |
| --- | ---: | ---: |
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
| 合计 | 9,957 | 10,107 |

HRBench 的记录已含四轮选项排列，不代表 800 道独立问题。SAT 保留原顺序和反转顺序。
当前模板是 `answer-parsing-2026-10-26-v2`：`tag_repair_v1`、`numeric_v1`、`format_compat_v1`。
BLINK 取 14 个任务宏平均，Mentis 取 5 个任务均值；MathVista 使用本地数值提取，没有外部 judge。
完整性检查拒绝缺题、重复题、变更 checkpoint/解析配置的混合结果。
现有去污染只针对 VStar；其余历史重叠情况保留在 benchmark 的 `audit/overlap_policy.json`。

## 公共原始数据下载与再处理

下面是只恢复元数据、从公共源重建图像的替代流程。完整恢复 `data-v1` 后无需再下载原图；
训练原图下载器使用单独的原始文件目录布局，请在仅恢复元数据的新 checkout 中执行。

```bash
# 查看来源与固定版本，不下载。
.venv/bin/python reproduce.py download-models --list
.venv/bin/python reproduce.py download-raw --kind training --list
.venv/bin/python reproduce.py download-raw --kind benchmarks --list

# 训练原图按已冻结的 image_manifest 从上游恢复；审核后的 reasoning 数据仍使用 data-v1。
.venv/bin/python reproduce.py restore-data --source /path/to/data-v1 --groups metadata reasoning_images
.venv/bin/python reproduce.py download-raw --kind training
.venv/bin/python reproduce.py verify-data --groups legacy_images

# 只从 data-v1 恢复元数据后，可以从上游重建 benchmark 图像。
.venv/bin/python reproduce.py restore-data --source /path/to/data-v1 --groups metadata
.venv/bin/python reproduce.py download-raw --kind benchmarks
.venv/bin/python reproduce.py prepare-benchmarks
```

benchmark 上游路径、revision、SHA256/Git blob hash 在 `manifests/benchmark_downloads.json`。
处理器读取原始图像、zip/tar 和 Parquet 内嵌图像，只接受与冻结记录 SHA256 相同的图像；
题目、选项、拆分、评分状态使用冻结协议元数据。不会因上游默认 split 改变而悄悄换题。
完整源转换代码保留在 `scripts/paper_benchmarks/`、`scripts/reasoning_benchmarks/`，
历史训练数据生成与审核代码保留在 `scripts/reasoning_data/`；这些历史构建脚本不是本仓库默认重建入口。

## 布局与验证边界

- `reproduce.py`、`tools/`：可移植的统一入口、安装、下载、释放数据包和验收。
- `lvr_pipeline_new/src/lvr/`：训练模型、collator、采样、checkpoint 与评分实现。
- `doc/flow_reconstruction_parallel_20261015/flow_reconstruction/`：冻结 Flow generator 与 GT 处理。
- `paper_experiments/2026-10-27-qwen25-flow-no-sat-4gpu/`：当前三阶段实现、离线缓存与测试。
- `paper_experiments/2026-10-27-qwen25-no-sat-4epoch/script/continuation_runtime.py`：Plain 的尾批与恢复逻辑。
- `manifests/original_code.json`：整理前来源哈希；`manifests/verification.json`：本次交付的实际验证范围。

整理时不占用训练中的 GPU。CPU 测试与数据/协议校验不能替代新环境上的真实四卡预检，
也不保证不同 CUDA、硬件或分布式归约顺序下逐位相同。训练运行中的旧目录没有改动。
本次验证使用了版本匹配的已安装环境；尚未执行全新虚拟环境的联网安装，也未重跑完整 GPU 训练或模型推理评测。
模型与数据沿用各自上游条款；本地整理不等于已公开发布，项目代码授权说明见 [NOTICE](NOTICE.md)。
