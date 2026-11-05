# 五模式 latent 消融

在**已经训练好的同一学生 checkpoint**上运行 Teacher、Teacher shuffled、Predicted、
Predicted shuffled、Zero。支持 **Qwen3-VL-4B-Instruct** 和 **Qwen2.5-VL-7B-Instruct**，
可手动选择并行推理 GPU 数量。此入口只做推理和评分。

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
python3.13 -m venv .venv-score
.venv-score/bin/python -m pip install -r requirements/scoring.txt
```

PyTorch 安装源为 CUDA 12.8，沿用仓库版本；设备需支持 BF16。
已有兼容环境可设置 `LVR_ABLATION_PYTHON=/path/to/python`，并用
`--score-python /path/to/scoring/python` 指定评分环境。不需要安装 DeepSpeed 或 diffusers。

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

## 本地验证

```bash
.venv/bin/python -m unittest discover -s ablation -p 'test_*.py' -v
```

测试覆盖全局供体、输入字段隔离、不同卡数的题目覆盖、续跑绑定、latent 缓存校验，
以及两种原生 Qwen 小模型在 CPU 上的 checkpoint 搬迁恢复和五种条件生成。
这不代表已经在训练后的完整 4B/7B 权重上完成 GPU 消融。
本次完整数据解包、16 组真实预处理和 13,935 条合成参考答案评分验收见
[verification.json](verification.json)；合成答案只用于检查评分链路。
