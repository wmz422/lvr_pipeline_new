"""评估层：benchmark runner + inference + 各 benchmark。

从旧 sft/evaluation/ 搬运并统一（M6）：
- 复用 models 的统一加载（`LatentVLM` + checkpoint/io，不再自带第二套 model_loader）；
- 复用 data 处理单一真源（lvr.data.prompt 的 build_generation_prompt / load_resized_rgb）。
"""

from lvr.eval.inference import baseline_generate, sft_generate
from lvr.eval.scoring import extract_answer

__all__ = ["sft_generate", "baseline_generate", "extract_answer"]
