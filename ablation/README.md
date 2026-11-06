# Latent 消融：五模式回答与跨 case 相似度

在**已经训练好的同一学生 checkpoint**上运行 Teacher、Teacher shuffled、Predicted、
Predicted shuffled、Zero。支持 **Qwen3-VL-4B-Instruct** 和 **Qwen2.5-VL-7B-Instruct**，
可手动选择并行推理 GPU 数量。此入口只做推理和评分。

新增 **Monet / LVR / 本方法的跨 case cosine**：在同一批四项 benchmark 上，分别缓存
Teacher 与 Predicted 的前 8 个位置，输出两张曲线和两张数值表。执行步骤见后文。

## 已发布的历史四阶段权重

[Wing22/Qwen3-VL-4B-Flow-4Stage-Ablation](https://huggingface.co/Wing22/Qwen3-VL-4B-Flow-4Stage-Ablation)
提供原始最终学生 `joint/last_full.ckpt` 和匹配教师 `teacher/student_source.ckpt`。
实际基座是 **Qwen3-VL-4B-Instruct**。来源为历史运行
`flow_t8_s16_vae4_continue_20261019`，四阶段依次为 Teacher alignment、Teacher joint、
Student warmup、Student joint，优化步数为 1394 / 1394 / 200 / 1394，保留 Description loss。
这是历史四阶段 Flow 消融，区别于仓库后来使用的三阶段训练配方。

两份文件保留原始 Lightning 格式、精度和参数，合计约 **60.11 GB**；教师导出不含推理无需的
Flow mapper。它们需要本仓库的加载器和单独下载的 Qwen3 基座，不能直接用 HF AutoModel 加载。
`models.lock.json` 固定模型 revision、文件大小和校验值；模型页的 `model_release.json`
记录实际配置、训练步数和 SHA256。下方下载器同时解决新设备没有权重和官方推理源码的问题。

## 需要准备的文件

1. 对应基座的本地模型目录：完整 safetensors、`config.json` 和原生 tokenizer/processor。
2. 已训练完成的学生 Joint 权重：完整的 `joint/checkpoints/last_full.ckpt`。
3. 与学生匹配的教师权重：`teacher_joint/checkpoints/student_source.ckpt`。
   这是训练入口生成的、移除了 Flow mapper 的教师推理权重；checkpoint 内的旧机器绝对路径
   会由命令行参数替换。
4. 下方独立 benchmark 数据包，包含原图、辅助图、参考答案、评分资源和冻结的全局打乱映射。

使用本仓库训练得到的 Qwen3 Flow、Qwen3 无 Flow loss、Qwen2.5 Flow 权重均走同一入口。
需要含 `state_dict`、`hyper_parameters`、`global_step` 的完整 Lightning 格式文件。
如果手中只有 ZeRO 分片，先使用仓库的 `reproduce.py export --checkpoint ... --output ...`
合并学生权重；教师使用训练流程已导出的 `student_source.ckpt`。

**不需要训练图像、训练 JSON、离线 VAE 缓存或 FLUX 权重。** 每张参与推理的 GPU 放置一份完整
模型；增加 `--gpus` 会把题目分给更多 GPU，不会把一个模型拆到多张卡。

## 环境

已经按仓库主 README 安装过环境，可以直接使用 `.venv` 和 `.venv-score`。
只运行消融时，也可以在 Linux 上单独安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ablation/requirements-inference.txt
.venv/bin/python -m pip install -r ablation/requirements-similarity.txt
python3.13 -m venv .venv-score
.venv-score/bin/python -m pip install -r requirements/scoring.txt
```

PyTorch 安装源为 CUDA 12.8，沿用仓库版本；设备需支持 BF16。
已有兼容环境可设置 `LVR_ABLATION_PYTHON=/path/to/python`，并用
`--score-python /path/to/scoring/python` 指定评分环境。不需要安装 DeepSpeed 或 diffusers。

## 下载模型与官方推理源码

```bash
# 默认包含本方法学生+教师、Qwen3 基座、Monet、Monet SFT stage2 教师、LVR，
# 以及 Qwen2.5 的原生 processor 和两种 baseline 的最小官方推理源码。
.venv/bin/python ablation/download_models.py --output /models/latent-ablation

# 只运行本方法的五模式实验，可只下载本方法和 Qwen3 基座。
.venv/bin/python ablation/download_models.py --output /models/latent-ablation \
  --models ours qwen3-base

# 不下载模型，只取固定 commit 的官方推理源码及许可证。
python3 ablation/download_models.py --output /models/latent-ablation --sources-only

# 对已有完整下载重新核验文件哈希。
.venv/bin/python ablation/download_models.py --output /models/latent-ablation --verify-only
```

资产保存到 `models/<name>/` 与 `sources/<method>/`。默认全套模型约 119 GB
（不含 benchmark 与 Python 环境），可用 `--models` 分批选择。
下载器固定 revision，不跟随 `main`，并核验大文件 SHA256、小文件 Git blob SHA 和官方源码 SHA256。
中断后重跑同一命令可继续下载。首次核验完成后写入 `verified.json`；推理检查 revision、
文件大小及修改时间，`--verify-assets` 可在每次推理启动时重新读取全部权重并核验哈希。

| 名称 | 官方权重 | 使用方式 |
| --- | --- | --- |
| ours | 上方历史 Qwen3-VL-4B 四阶段 Flow | 学生与匹配教师 |
| monet | [NOVAglow646/Monet-7B](https://huggingface.co/NOVAglow646/Monet-7B) | 官方 vLLM 连续 latent 推理 |
| monet-teacher | [NOVAglow646/Monet-SFT-7B](https://huggingface.co/NOVAglow646/Monet-SFT-7B)，`stage2/` | 官方 SFT stage3 使用的教师来源 |
| lvr | [vincentleebang/LVR-7B](https://huggingface.co/vincentleebang/LVR-7B) | 官方 QwenWithLVR 生成与视觉监督目标 |

源码分别来自 [Monet](https://github.com/NOVAglow646/Monet) 与
[LVR](https://github.com/VincentLeebang/lvr)，精确 commit 和实际下载文件清单见
[models.lock.json](models.lock.json)。不需要在新设备另找推理脚本。

## 下载消融 benchmark

已发布的独立数据集：
[Wing22/lvr-latent-ablation-benchmarks-v1](https://huggingface.co/datasets/Wing22/lvr-latent-ablation-benchmarks-v1)。
`publication.json` 固定精确 revision `bad9a25067e80e90f82069c09edda5700fe13895`；
下载器逐包、逐文件校验 SHA256。

```bash
# 下载并解包，默认保存到 ablation/data/；只使用 Python 标准库。
python3 ablation/download.py

# 自定义位置。
python3 ablation/download.py --output /data/latent-ablation

# 已在其他设备下载 tar 分片：离线恢复，不访问网络。
python3 ablation/download.py --source /data/downloaded-ablation-shards \
  --output /data/latent-ablation

# 仅验收已经解包的数据。
python3 ablation/download.py --verify-only --output /data/latent-ablation
```

| Benchmark | 条数 | 教师辅助图 |
| --- | ---: | --- |
| HRBench4K | 800 | 772 条已有模型框裁剪；28 条本题原图补位 |
| VStarBench | 191 | 官方框裁剪 |
| MIRA | 546 | 官方有序辅助图 |
| MentisOculi | 1250 | 官方有序中间状态图 |

共 **2787 条、6989 个图像文件**，每个 checkpoint 共 **13,935 次答案生成**。
HR4K 保留四轮选项排列；772 条框来自答案条件模型标注。Teacher 是使用辅助信息的诊断条件。
所有 benchmark 保留完整范围及原评分方式，记录中的已知图像重叠信息继续保留。

## 启动

从仓库根目录执行。所有路径均可换成复现设备上的实际位置。

```bash
# Qwen3-VL-4B，4 张可见 GPU。
bash ablation/run.sh \
  --model-family qwen3-4b \
  --base-model /models/Qwen3-VL-4B-Instruct \
  --checkpoint /weights/qwen3-flow/joint/checkpoints/last_full.ckpt \
  --teacher-checkpoint /weights/qwen3-flow/teacher_joint/checkpoints/student_source.ckpt \
  --benchmark /data/latent-ablation \
  --gpus 4 --output runs/ablation-qwen3-flow

# 直接使用本页下载的历史四阶段模型。
bash ablation/run.sh --model-family qwen3-4b \
  --base-model /models/latent-ablation/models/qwen3-base \
  --checkpoint /models/latent-ablation/models/ours/joint/last_full.ckpt \
  --teacher-checkpoint /models/latent-ablation/models/ours/teacher/student_source.ckpt \
  --benchmark /data/latent-ablation --gpus 4 --output runs/ablation-four-stage-flow

# Qwen2.5-VL-7B，指定两张卡；--devices 会确定并行数量。
bash ablation/run.sh \
  --model-family qwen25-7b \
  --base-model /models/Qwen2.5-VL-7B-Instruct \
  --checkpoint /weights/qwen25-flow/joint/checkpoints/last_full.ckpt \
  --teacher-checkpoint /weights/qwen25-flow/teacher_joint/checkpoints/student_source.ckpt \
  --benchmark /data/latent-ablation \
  --devices 0,2 --output runs/ablation-qwen25-flow
```

`--gpus 1/2/4/8` 选择当前可见 GPU 的前 N 张；也可先设置 `CUDA_VISIBLE_DEVICES`。
`--devices 0,2,3,5` 显式选择设备。两项同时设置时，数量必须一致。
GPU 数量不改变数据、供体、解码设置或评分；保持每卡 batch=1。

在完整命令末尾添加 `--dry-run`：仅核对全部数据、模型类型和权重文件路径，不加载模型或占用 GPU。
添加 `--smoke-per-benchmark 1` 并选择**新的输出目录**：每项挑选一条含较多 helper/较大原图的记录，
先检查真实模型加载、两种 latent 和全部五种回答。该运行标为 smoke，不生成正式主表。

中断后重复原命令即可续跑，**允许改变 `--gpus` 或 `--devices`**。缓存与答案按题目 ID 保存，
与 GPU 分片数无关；权重、数据、代码或测试范围变化时必须使用新输出目录。
原图使用基座原生 processor 默认分辨率，helper 上限 262144 pixels；遇到 OOM 或上下文超限会报错，
不会自动缩图、截断输入或跳题。

## 五种条件与输出

| 模式 | 注入同一 QA 模型的 latent |
| --- | --- |
| Teacher | 本题原图＋helper 经冻结教师提取的 mean latent |
| Teacher shuffled | 全局供体的教师 mean latent |
| Predicted | 本题原图＋问题预测的 latent |
| Predicted shuffled | 同一全局供体的预测 latent |
| Zero | projector 前的 `[1,8,512]` 零张量，保留 8 个位置 |

两种 shuffle 共用全部 2787 条上的 seed42 一一置换，不按 benchmark/任务等分桶，
排除自身和同问题组；1894 对跨 benchmark，允许其余配对在同 benchmark 内。
接收题的原图、问题、选项、学生权重、projector、注入位置和 greedy 解码始终一致。
答案/评分状态不会进入预测器或教师输入。教师是否读取问题和预测循环数沿用 checkpoint 设置。

入口先缓存各题的 Teacher/Predicted latent，再运行五种回答；缓存约 87 MiB。
所有 worker 完成后自动使用仓库冻结的解析和原生评分方式汇总。
正式结果要求五模式全部 2787 条齐全，输出目录包含：

- `run_config.json`：权重 SHA256、代码、数据版本及条件绑定。
- `latents/`：每题的教师与预测 latent；`predictions/<mode>/`：逐题原始输出与供体 ID。
- `logs/`：各 GPU、各阶段日志和真实 processor 设置。
- `scores/<mode>/`：逐题评分、benchmark 主指标和 HR4K 裁剪/原图补位分组。
- `summary.json`、`main_table.csv`、`main_table.md`：四项 benchmark × 五模式，以及四项主分宏平均。

仅重新评分：

```bash
.venv-score/bin/python ablation/score.py --output runs/ablation-qwen3-flow
```

## Monet / LVR / 本方法：不同 case 的前 8 个位置相似度

此实验提取模型的实际 latent，不需要生成五模式的答案。Monet、LVR 的 Predicted 分支仍运行
各自原生生成过程，保存连续 latent 被下一步实际消费时的向量；不会拿普通文本 token 的
hidden state 代替 latent。所有模型使用完全相同的冻结 2787 条记录、原图和有序 helper。

### 分开安装 baseline 环境

三个环境对应实际不同的 Transformers/vLLM API。**LVR 和 Monet Teacher 使用 4.54.1**，
**Monet Predicted 使用 vLLM 0.10.0 + Transformers 4.53.3**；本方法继续使用上方 5.15.0。

```bash
python3.12 -m venv .venv-lvr
.venv-lvr/bin/python -m pip install -r ablation/requirements-lvr.txt
python3.12 -m venv .venv-monet
.venv-monet/bin/python -m pip install -r ablation/requirements-monet.txt
```

### 提取六组 latent

从仓库根目录运行。以下顺序复用一张可见 GPU，每次只加载当前方法。
输出目录要与五模式回答实验分开。

```bash
export ABLATION_ASSETS=/models/latent-ablation
export ABLATION_BENCHMARK=/data/latent-ablation
export ABLATION_SIM_OUTPUT=runs/cosine-four-stage-flow

for kind in teacher predicted; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python ablation/similarity.py \
    --method ours --kind "$kind" --assets "$ABLATION_ASSETS" \
    --benchmark "$ABLATION_BENCHMARK" --output "$ABLATION_SIM_OUTPUT"
  CUDA_VISIBLE_DEVICES=0 .venv-lvr/bin/python ablation/similarity.py \
    --method lvr --kind "$kind" --assets "$ABLATION_ASSETS" \
    --benchmark "$ABLATION_BENCHMARK" --output "$ABLATION_SIM_OUTPUT"
done

CUDA_VISIBLE_DEVICES=0 .venv-lvr/bin/python ablation/similarity.py \
  --method monet --kind teacher --assets "$ABLATION_ASSETS" \
  --benchmark "$ABLATION_BENCHMARK" --output "$ABLATION_SIM_OUTPUT"
CUDA_VISIBLE_DEVICES=0 .venv-monet/bin/python ablation/similarity.py \
  --method monet --kind predicted --assets "$ABLATION_ASSETS" \
  --benchmark "$ABLATION_BENCHMARK" --output "$ABLATION_SIM_OUTPUT"

.venv/bin/python ablation/similarity_stats.py \
  --benchmark "$ABLATION_BENCHMARK" --output "$ABLATION_SIM_OUTPUT"
```

每条提取命令加 `--dry-run` 可在 CPU 上验证数据、下载资产及版本，且不生成正式缓存。
六条命令均加 `--smoke-per-benchmark 2`、换一个输出目录，再给统计命令加 `--allow-smoke`，
可先跑真实模型小样本检查。统计图会标注 SMOKE，不能作为正式结论。

多 GPU 使用独立进程分题，例如给同一条提取命令分别加
`--num-shards 4 --shard-index 0/1/2/3`，每个进程用 `CUDA_VISIBLE_DEVICES` 选择一张卡。
Monet/vLLM 每个进程只暴露一张 GPU，并保持 `--device cuda:0`。
中断后重复原命令即可续跑；要改变分片数，先停止该分支所有旧进程。
已有缓存会检查题目 ID、运行绑定、张量哈希、有限值和维度，损坏或其他版本缓存会报错。
输入超限或 OOM 会报错，不自动缩图或跳题。

### Teacher 的定义及位置含义

| 方法 | Teacher | Predicted | 横轴位置含义 |
| --- | --- | --- | --- |
| 本方法 4B | 冻结教师对原图＋helper 的 mean latent；是否读问题沿用 checkpoint | 学生预测、projector 前的 512 维 latent | 8 个学习槽位，**不是 Flow 迭代次数** |
| Monet 7B | 官方 SFT stage2 模型，按官方 precompute 布局在 helper 后放 8 个 latent pad；取首个 helper 块的目标向量 | 官方 vLLM runner 实际消费的连续向量 | Teacher 为 helper 后的 latent 槽；Predicted 为原生生成顺序 |
| LVR 7B | 官方训练使用的视觉 merger 监督特征；在本套 helper 上按顺序连接后取前 8 个视觉 patch 向量 | 官方 `decoding_strategy=steps, lvr_steps=[8]` 实际消费的循环向量 | Teacher 为有序视觉 patch；Predicted 为原生生成顺序 |

**LVR 没有与本方法同构的自回归 Teacher**，因此其 Teacher 列是视觉监督目标诊断；
这里的前 8 个 patch 也不是重新构造其训练时的 ROI 采样。
三者维度和位置语义不同，只在**各自模型内部**计算不同 case 的 cosine，不直接计算跨模型向量 cosine。
图的横轴统一写为 Latent position，不能解释为等价的 8 次推理。
Teacher 图用于比较各方法监督目标的分散度；LVR 的前几个视觉 patch 也可能受背景和空间位置影响，
不能仅据这条 Teacher 曲线判定其生成 latent 发生坍缩。

Monet Predicted 保留作者原生 `LATENT_SIZE=10`，只观察前 8 个实际消费向量，
跨多个原生 latent 块时按消费顺序取前 8 个。保留作者采样参数
temperature=0.1、top-k=50、top-p=0.8、repetition penalty=1.01、max tokens=4096；
LVR 保留 greedy、512 tokens。运行记录保存 prompt、原生生成文本、长度、processor 和源码版本。
为使分片和续跑结果可复现，Monet 在这些采样参数下使用由 case ID 的 SHA256 派生的固定 seed；
作者示例为 `seed=None`，本入口明确记录这一差异。
Monet 的 observer 只复制被消费的向量，并验证当前引擎 worker 确实加载官方修改版 runner；
不会强制插入 latent、缩短原生块、补零或复制缺失位置。
原图使用 Qwen2.5 原生 processor；helper 上限为 262144 pixels。
Monet 引擎也显式采用这个原生 min/max pixels，并逐题核对本地 processor 与引擎的输入 token 数。
Baseline 输入沿用作者消息格式：去除共享的 `<answer>` 直接回答页脚，Monet 再去除直接输出选项字母的页脚，
以保留其原生 latent 推理；原题内容、选项和任务规则保留。实际 prompt 写入逐题记录。
本方法、Monet Teacher 和 LVR Teacher 均不读取参考答案或评分状态；HR4K 已发布 helper 的标注来源仍如上所述。

### 统计和输出

主结果在每个 benchmark 内，对不同 case 的**同一位置**计算余弦相似度，取所有合格无序 pair
的均值，不是计算同一 case 内不同位置之间的相似度。排除自身、同 `group_id`（含 HR 选项排列）
以及原图集合完全相同的记录；两种关系取传递闭包，避免重复图泄漏进独立 pair。

主曲线使用三种方法 × 两种分支均具有 8 个真实位置的 **共同 case 集**，8 个位置的样本集保持一致。
缺少文件视为运行未完成，会拒绝汇总；原生生成不足 8 个位置的记录保存真实短序列并进入覆盖率报告，
不进入共同集。因此必须同时报告共同集规模及各分支覆盖率；覆盖率低时，曲线只代表被保留的子集。

`report/` 包含：

- `teacher.csv/.md`、`predicted.csv/.md`：每个 benchmark × 方法，横向为位置 1–8，另含 case 数和 pair 数。
- `teacher.png/.pdf`、`predicted.png/.pdf`：两张图，各含四个 benchmark 子图；纵轴为平均 cosine。
- `coverage.json`：每个分支逐位置覆盖数量、不足 8 个位置的 ID、最终共同 case ID。
- `*_within-benchmark.csv`：同 benchmark 内所有合格 pair，默认主表。
- `*_within-task.csv`、`*_cross-task.csv`：同任务与不同任务 pair 分开统计，避免混淆来源。
- `report.json`：数据、权重、源码、环境、统计范围与运行绑定。

要以同任务 pair 为主图，统计时加 `--pair-scope within-task`；跨任务用 `--pair-scope cross-task`。
同任务结果按 pair 数加权；pair 不足的位置显示 NaN，不填 0。正式主图若某 benchmark 没有合格 pair 会拒绝生成。

**较低的 raw cosine 只说明表示方向更分散，不能单独证明编码了更多视觉信息。**
对“视觉信息更多”的结论还需要视觉结构 probe、teacher 检索或干预后的任务表现证据。
这次提交提供复现入口与 CPU 验证，尚未运行这六组完整大模型 GPU 提取，仓库不附伪造数值。

## 本地验证

```bash
.venv/bin/python -m unittest discover -s ablation -p 'test_*.py' -v
```

测试覆盖全局供体、输入字段隔离、不同卡数的题目覆盖、续跑绑定、latent 缓存校验，
以及两种原生 Qwen 小模型在 CPU 上的 checkpoint 搬迁恢复和五种条件生成。
这不代表已经在训练后的完整 4B/7B 权重上完成 GPU 消融。
本次完整数据解包、16 组真实预处理和 13,935 条合成参考答案评分验收见
[verification.json](verification.json)；合成答案只用于检查评分链路。

跨 case 实验另有真实官方代码的 CPU 小模型检查，在 `.venv-lvr` 中运行：

```bash
.venv-lvr/bin/python ablation/check_native_similarity.py --method lvr \
  --sources /models/latent-ablation/sources
.venv-lvr/bin/python ablation/check_native_similarity.py --method monet-teacher \
  --sources /models/latent-ablation/sources
```

这些随机小模型用于核验 latent 捕获及官方接口，不产生训练后模型的 benchmark 分数。
本次 23 项 CPU 测试、官方小模型、12 组真实 processor 输入与合成出图链路的验收范围，
以及尚未执行的 GPU 项目，见 [similarity_verification.json](similarity_verification.json)。
