"""普通 Qwen3-VL SFT 入口。

复用 LvrCLI 的 runs/{exp}/{version} 复现约定，但模型绑定为 PlainVLM。

用法：
    python -m lvr.plain_cli fit --config configs/base.yaml \
        --config configs/plain_sft/v1.0.0.yaml --exp_name plain_sft --version v1
"""

from __future__ import annotations

from lvr.cli import LvrCLI
from lvr.data import LatentDataModule
from lvr.models import PlainVLM


def main() -> None:
    LvrCLI(
        PlainVLM,
        LatentDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
