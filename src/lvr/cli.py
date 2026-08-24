"""LightningCLI 入口（= 旧 sft/src/main.py）。

LvrCLI 在原生 LightningCLI 上加「版本号 + 一实验一目录」约定：
    --exp_name <name>  --version <ver>
会把本次运行的所有产物（config.yaml / checkpoints / predictions / 日志 / run_meta.json）
自动嵌套到 runs/{exp_name}/{version}/ ，开箱即复现。

用法：
    python -m lvr.cli fit  --config configs/base.yaml --config configs/sft/v1.0.0.yaml \
        --exp_name sft --version v1
    python -m lvr.cli test --config configs/base.yaml --config configs/eval/test_sft.yaml \
        --exp_name sft --version v1
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from lvr.data import LatentDataModule
from lvr.models import LatentVLM


class LvrCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--exp_name", type=str, default="sft", help="实验名")
        parser.add_argument("--version", type=str, default="dev", help="版本号（如 v1/v2），一实验一目录的 key")

    def before_instantiate_classes(self) -> None:
        cfg = getattr(self.config, self.subcommand) if self.subcommand else self.config
        run_dir = Path("runs") / str(cfg.exp_name) / str(cfg.version)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. trainer 输出根目录
        cfg.trainer.default_root_dir = str(run_dir)

        # 2. logger save_dir（若配置了 class_path 形式的 logger）
        logger = getattr(cfg.trainer, "logger", None)
        init_args = getattr(logger, "init_args", None) if logger is not None else None
        if init_args is not None and hasattr(init_args, "save_dir"):
            init_args.save_dir = str(run_dir)

        # 3. ModelCheckpoint dirpath
        callbacks = getattr(cfg.trainer, "callbacks", None) or []
        for cb in callbacks:
            if "ModelCheckpoint" in str(getattr(cb, "class_path", "")):
                cb_args = getattr(cb, "init_args", None)
                if cb_args is not None:
                    cb_args.dirpath = str(run_dir / "checkpoints")

        # 4. 生成 case 输出
        model_cfg = getattr(cfg, "model", None)
        if model_cfg is not None and hasattr(model_cfg, "generation_output_path"):
            model_cfg.generation_output_path = str(run_dir / "predictions.jsonl")

        is_global_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
        if not is_global_zero:
            print(f"[LvrCLI] run dir = {run_dir}")
            return

        # 5. run_meta.json（版本 / 时间 / 子命令 / 是否 resume），便于台账与复现。
        #    每次调用追加一条 history（首训 / 每次 resume / 每次 test），不覆盖历史，
        #    这样 resume 轨迹（从哪个 ckpt 续）有据可查。
        meta_path = run_dir / "run_meta.json"
        history = []
        if meta_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict):
                    history = prev.get("history", [])
            except Exception:  # noqa: BLE001
                history = []
        ckpt_path = getattr(cfg, "ckpt_path", None)
        history.append(
            {
                "subcommand": self.subcommand,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "ckpt_path": str(ckpt_path) if ckpt_path else None,
            }
        )
        meta = {
            "exp_name": str(cfg.exp_name),
            "version": str(cfg.version),
            "history": history,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 6. 把合并后的完整 config 直接 dump 到 run_dir 根（不依赖 logger 位置），保证可复现。
        try:
            cfg_yaml = self.parser.dump(self.config, skip_none=False)
            (run_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[LvrCLI] config dump to run_dir skipped: {exc}")

        print(f"[LvrCLI] run dir = {run_dir}")


def main() -> None:
    LvrCLI(
        LatentVLM,
        LatentDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
