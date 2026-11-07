# 历史记录：2026-11-02 Qwen3 三阶段验证

本页记录旧版三阶段、无 Description 配方的验证，不代表当前训练入口。
当前四阶段 Flow、去 Flow loss、去 latent 对齐三个脚本的用法见
[README_qwen3.md](../README_qwen3.md)。

`train_qwen3_flow.sh` 和 `train_qwen3_no_flow_loss.sh` 复用原三阶段启动器、数据准备、
缓存、checkpoint 导出和统一评测。基座锁在 `manifests/qwen3_model.json`；旧 Qwen2.5
模型锁和数据、benchmark 清单保持不变。

完整实验与消融只改变 Teacher 的 Flow loss 系数（0.1 / 0.0）以及输出目录。
零系数时仍计算 Flow MSE 并记录日志，以维持相同输入、随机数调用和可训练参数集合。
Warmup、Joint 仍分别使用各自训练好的 Teacher，最终评测读取各自 Joint checkpoint。

本地验证使用与仓库锁定版本匹配的已安装 Python 3.12 训练环境，设置
`CUDA_VISIBLE_DEVICES=''`，原有四卡训练继续运行。实际执行范围记录在
`manifests/qwen3_experiments.json`。

- 配方逐字段比较：原配方替换模型路径后与 Qwen3 完整配方一致；消融仅多出 Flow 系数变化。
- 真实的小型 Qwen3-VL 与 FLUX：完整实验 Flow 梯度非零，消融 Flow 梯度为零，
  消融总损失精确等于 QA CE，QA 和 encoder 仍得到有限、非零梯度，冻结生成器不产生参数梯度。
- 两个变体均实际执行小模型 Teacher optimizer update、验证、checkpoint 导出、
  Warmup update、Joint update、验证与 checkpoint 重载后的 greedy 生成。
- 真实 Qwen3 processor、冻结训练/验证样本 ID、离线 FP32 VAE 和实际 datamodule 检查。
- 完整 11 项 benchmark 启动协议检查、仓库已有训练、存储、恢复和评分回归测试。

小模型 CPU 验证使用缩小的 Qwen3/FLUX 网络和合成图像 patch；真实 processor 检查使用
实际冻结数据。CPU VAE 探针在独立目录缩小重建图，仅用于验证缓存与 loader；正式配方
仍使用 2MP 原图预算。上述验证没有运行完整 4B 的 GPU 训练或完整 benchmark 模型推理，
不包含新的实验精度结果。新脚本在正式训练结束后会自动执行完整测试。

验证命令：

```bash
.venv/bin/python reproduce.py test
LVR_FLOW_VARIANT=qwen3 .venv/bin/python reproduce.py cpu-check --output runs/qwen3-cpu-check
LVR_FLOW_VARIANT=qwen3-no-flow-loss .venv/bin/python reproduce.py cpu-check --output runs/qwen3-ablation-cpu-check
bash train_qwen3_flow.sh --dry-run
bash train_qwen3_no_flow_loss.sh --dry-run
```
