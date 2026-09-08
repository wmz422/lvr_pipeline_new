"""Evaluate native training checkpoints and optional plain-Qwen baselines. Public bundles use lvr.evaluate."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from lvr.eval.benchmarks import BENCHMARKS
from lvr.eval.inference import SFT_IMAGE_SIZE, baseline_generate, known_latent_generate, sft_generate


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_processor(qwen_model_name_or_path: str, latent_pad_token: str, add_latent: bool,
                     max_pixels: int | None = None):
    """构造 Qwen3-VL processor，并注册 latent special token —— 复用 lvr.tokens 的唯一入口，
    保证其 tokenizer 的 latent token id 与模型侧（builder.build_qwen）完全一致。
    max_pixels：与训练一致的动态分辨率像素上限 cap（None=Qwen 默认）。"""
    from transformers import AutoProcessor

    from lvr.data.prompt import cap_qwen_pixels
    from lvr.tokens import register_latent_tokens

    processor = AutoProcessor.from_pretrained(qwen_model_name_or_path)
    tok = processor.tokenizer
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if add_latent:
        register_latent_tokens(tok, latent_pad_token)
    cap_qwen_pixels(processor, max_pixels)
    return processor


def load_sft_model(
    model_cfg: dict[str, Any],
    ckpt_path: str,
    device: str,
    max_pixels: int | None = None,
    preload_checkpoint_path: str | None = None,
):
    """统一加载：LatentVLM(**base.model, init_checkpoint_path=ckpt)。返回 (model, processor)。"""
    from lvr.models import LatentVLM

    # 只挑 LatentVLM.__init__ 认识的参数（base.yaml model 段可能含额外 key）。
    valid = set(inspect.signature(LatentVLM.__init__).parameters)
    kwargs = {k: v for k, v in model_cfg.items() if k in valid}
    kwargs["init_checkpoint_path"] = (
        str(Path(preload_checkpoint_path).expanduser())
        if preload_checkpoint_path is not None
        else str(Path(ckpt_path).expanduser())
    )

    model = LatentVLM(**kwargs)
    if preload_checkpoint_path is not None:
        from lvr.checkpoint.io import load_init_checkpoint

        load_init_checkpoint(model, str(Path(ckpt_path).expanduser()))
    model.to(device)
    model.eval()

    processor = _build_processor(
        model_cfg["qwen_model_name_or_path"],
        model_cfg.get("latent_pad_token", "<abs_vis_token_pad>"),
        model_cfg.get("add_latent_special_tokens", True),
        max_pixels,
    )
    return model, processor


def load_plain_model(model_cfg: dict[str, Any], ckpt_path: str, device: str, max_pixels: int | None = None):
    """加载普通 Qwen3-VL SFT（PlainVLM）的 DeepSpeed ZeRO-3 checkpoint。

    PlainVLM = 微调后的 vanilla Qwen3-VL（无 LAM/latent/projector），训练时
    exclude_frozen_parameters=true 只存 LM + lm_head。这里建 PlainVLM（from_pretrained
    base 权重）后用 checkpoint/io 把 ZeRO 分片里的可训参数 load_state_dict(strict=False)
    覆盖回去，再取其底层 .qwen 走 baseline_generate（HF generate）。返回 (qwen, processor)。
    """
    from lvr.checkpoint.io import load_init_checkpoint
    from lvr.models import PlainVLM

    valid = set(inspect.signature(PlainVLM.__init__).parameters)
    kwargs = {k: v for k, v in model_cfg.items() if k in valid}
    model = PlainVLM(**kwargs)
    load_init_checkpoint(model, str(Path(ckpt_path).expanduser()))

    qwen = model.qwen
    qwen.to(device)
    qwen.eval()

    # plain SFT 不注册 latent special token（add_latent=False），processor 与训练一致。
    processor = _build_processor(model_cfg["qwen_model_name_or_path"], "", add_latent=False,
                                 max_pixels=max_pixels)
    return qwen, processor


def load_baseline_model(model_path: str, device: str, max_pixels: int | None = None):
    """vanilla Qwen3-VL（HF from_pretrained）。返回 (model, processor)。"""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from lvr.data.prompt import cap_qwen_pixels

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    tok = processor.tokenizer
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    cap_qwen_pixels(processor, max_pixels)
    return model, processor


def _summary_from_saved(saved: list[dict[str, Any]], bench_name: str) -> dict[str, Any]:
    """从已存盘的逐样本结果重算汇总（cache-resume 用）。"""
    total = len(saved)
    correct = sum(1 for r in saved if r.get("correct", False))
    cats: dict[str, dict[str, int]] = {}
    for r in saved:
        cat = r.get("category")
        if cat:
            cats.setdefault(cat, {"total": 0, "correct": 0})
            cats[cat]["total"] += 1
            if r.get("correct", False):
                cats[cat]["correct"] += 1
    return {
        "benchmark": bench_name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total > 0 else 0.0,
        "categories": {
            c: {"total": s["total"], "correct": s["correct"],
                "accuracy": s["correct"] / s["total"] * 100 if s["total"] > 0 else 0.0}
            for c, s in cats.items()
        },
    }


def run_model_evaluation(model_name, model, processor, infer_fn, config, out_dir_base,
                         use_cache: bool = True) -> dict[str, Any]:
    device = next(model.parameters()).device
    print(f"\n{'='*64}\nModel: {model_name} (device={device})\n{'='*64}")

    gen_cfg = config["generation"]
    results: dict[str, Any] = {}

    for bench_name in config["benchmarks"]:
        bench_cfg = config["data"].get(bench_name, {})
        benchmark = BENCHMARKS[bench_name](bench_cfg)

        out_dir = os.path.join(out_dir_base, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{bench_name}.json")

        if use_cache and os.path.exists(out_file):  # cache-resume：已有结果直接读（--no-cache 强制重跑）
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            summary = _summary_from_saved(saved, bench_name)
            print(f"\n{bench_name.upper()}: {summary['accuracy']:.1f}% "
                  f"({summary['correct']}/{summary['total']}) [loaded from cache] "
                  f"⚠ 缓存不校验生成参数/ckpt 是否一致，换口径重跑请用 --no-cache 或换 output_dir")
        else:
            summary = benchmark.evaluate(model, infer_fn, out_dir)
            print(f"\n{bench_name.upper()}: {summary['accuracy']:.1f}% "
                  f"({summary['correct']}/{summary['total']})")

        for cat, info in summary.get("categories", {}).items():
            print(f"  {cat}: {info['accuracy']:.1f}% ({info['correct']}/{info['total']})")
        results[bench_name] = summary

    return results


def free_model(model: Any, model_name: str) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[Memory] Freed {model_name}")


def build_infer_fn(model_type: str, processor: Any, gen_cfg: dict[str, Any],
                   image_size=SFT_IMAGE_SIZE, system: str = "", lam_image_size=SFT_IMAGE_SIZE):
    if model_type == "sft":
        return lambda m, images, q: sft_generate(m, processor, images, q, gen_cfg, image_size, system)
    if model_type == "known_latent":
        return lambda m, images, q: known_latent_generate(
            m, processor, images, q, gen_cfg, image_size, system, lam_image_size
        )
    return lambda m, images, q: baseline_generate(m, processor, images, q, gen_cfg, image_size, system)


def print_summary(all_results: dict[str, dict[str, Any]]) -> None:
    print("\n\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    if not all_results:
        return
    bench_names = list(next(iter(all_results.values())).keys())
    header = f"{'Model':<18}" + "".join(f"{bn.upper():>10}" for bn in bench_names)
    print(header + "\n" + "-" * len(header))
    for model_name, bench_results in all_results.items():
        row = f"{model_name:<18}" + "".join(
            f"{bench_results.get(bn, {}).get('accuracy', 0.0):>9.1f}%" for bn in bench_names
        )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="LVR benchmark 评估")
    parser.add_argument("--base-config", default="configs/base.yaml", help="模型/LAM 参数单一真源")
    parser.add_argument("--bench-config", default="configs/eval/bench.yaml", help="benchmark/ckpt/数据路径")
    parser.add_argument("--device", default="cuda:0", help="评估设备")
    # 版本感知（一实验一目录约定）：--version 把「输出」与「待评 ckpt」都路由到 runs/{exp}/{version}/。
    parser.add_argument("--exp", default="sft", help="实验名（配 --version 用），默认 sft")
    parser.add_argument("--version", default=None,
                        help="版本号；给了就评 runs/{exp}/{version}/checkpoints/last.ckpt 并把结果写到 runs/{exp}/{version}/eval/")
    parser.add_argument("--ckpt", default=None, help="显式 checkpoint 路径，覆盖 --version 的 last.ckpt 与 config 的 checkpoints")
    parser.add_argument("--plain-ckpt", default=None,
                        help="显式 PlainVLM checkpoint 路径（plain SFT，走 load_plain_model），覆盖 config 的 plain_checkpoints；与 --ckpt 互斥")
    parser.add_argument("--ckpt-name", default=None, help="显式 ckpt 在结果里的名字（默认取 version 或路径名）")
    parser.add_argument("--output-dir", default=None, help="显式输出目录，优先级最高")
    parser.add_argument("--with-baseline", action="store_true", help="version 模式下仍跑 baseline（默认不跑，baseline 是跨版本参考）")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略 output_dir 下已存在的逐样本结果，强制重跑（默认存在即直接读缓存）")
    args = parser.parse_args()

    model_cfg = load_yaml(args.base_config)["model"]
    config = load_yaml(args.bench_config)
    device = args.device if torch.cuda.is_available() else "cpu"
    # --version：输出 → runs/{exp}/{version}/eval/，待评 ckpt → 该版本 last.ckpt（覆盖 config）。
    if args.version:
        if args.output_dir is None:
            args.output_dir = os.path.join("runs", args.exp, args.version, "eval")
        if args.ckpt is None:
            config["checkpoints"] = [{
                "name": args.version,
                "path": os.path.join("runs", args.exp, args.version, "checkpoints", "last.ckpt"),
            }]
            config["plain_checkpoints"] = []
        if not args.with_baseline:
            config.setdefault("baseline", {})["enabled"] = False
    # 显式 --ckpt 覆盖待评 checkpoint（仍可配 --version 决定输出位置）。
    if args.ckpt and args.plain_ckpt:
        parser.error("--ckpt 与 --plain-ckpt 互斥：LatentVLM ckpt 用 --ckpt，plain SFT ckpt 用 --plain-ckpt")
    if args.ckpt:
        name = args.ckpt_name or args.version or os.path.basename(args.ckpt.rstrip("/"))
        config["checkpoints"] = [{"name": name, "path": args.ckpt}]
        config["plain_checkpoints"] = []
    if args.plain_ckpt:
        name = args.ckpt_name or args.version or os.path.basename(args.plain_ckpt.rstrip("/"))
        config["plain_checkpoints"] = [{"name": name, "path": args.plain_ckpt}]
        config["checkpoints"] = []
    # 显式 --output-dir 优先级最高。
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # SFT 评估口径（须与本实验训练一致）：sft_image_size=null → Qwen 动态分辨率；
    # system_prompt → 与训练 collator 同一句（如 "You are a helpful assistant."）。缺省=旧行为(256/无 system)。
    sft_image_size = config["sft_image_size"] if "sft_image_size" in config else SFT_IMAGE_SIZE
    eval_system = config.get("system_prompt", "")
    qwen_max_pixels = config.get("qwen_max_pixels")  # 动态分辨率像素上限 cap（须与训练一致；None=Qwen 默认）
    # Known-latent LAM inputs have their own fixed resize.  Keep legacy configs
    # at 256 unless they explicitly supply the resolution used for SFT training.
    lam_image_size = config.get("lam_image_size", SFT_IMAGE_SIZE)
    sft_generation_mode = config.get("sft_generation_mode", "sft")
    preload_checkpoint_path = config.get("preload_checkpoint_path")

    out_dir_base = config["output_dir"]
    # 待评 ckpt 不存在时早点报清楚（避免加载到一半才崩）。
    for ck in config.get("checkpoints", []) + config.get("plain_checkpoints", []):
        if not os.path.exists(ck["path"]):
            print(f"[runner] ⚠ checkpoint 不存在: {ck['path']}（name={ck['name']}）")
    os.makedirs(out_dir_base, exist_ok=True)
    all_results: dict[str, dict[str, Any]] = {}

    # --- 逐个 SFT checkpoint（顺序加载，跑完即释放，省显存）---
    for ckpt in config.get("checkpoints", []):
        print(f"\n[Loading] {ckpt['name']} from {ckpt['path']} on {device}")
        model, processor = load_sft_model(
            model_cfg,
            ckpt["path"],
            device,
            qwen_max_pixels,
            preload_checkpoint_path=preload_checkpoint_path,
        )
        infer_fn = build_infer_fn(
            sft_generation_mode, processor, config["generation"], sft_image_size, eval_system,
            lam_image_size,
        )
        all_results[ckpt["name"]] = run_model_evaluation(
            ckpt["name"], model, processor, infer_fn, config, out_dir_base, use_cache=not args.no_cache
        )
        free_model(model, ckpt["name"])

    # --- 普通 Qwen3-VL SFT checkpoint（PlainVLM，走 HF generate）---
    for ckpt in config.get("plain_checkpoints", []):
        print(f"\n[Loading] plain {ckpt['name']} from {ckpt['path']} on {device}")
        model, processor = load_plain_model(model_cfg, ckpt["path"], device, qwen_max_pixels)
        infer_fn = build_infer_fn("baseline", processor, config["generation"], sft_image_size, eval_system)
        all_results[ckpt["name"]] = run_model_evaluation(
            ckpt["name"], model, processor, infer_fn, config, out_dir_base, use_cache=not args.no_cache
        )
        free_model(model, ckpt["name"])

    # --- baseline（对照）---
    if config.get("baseline", {}).get("enabled", False):
        bl = config["baseline"]
        # image_size: 缺省=256（与 SFT 同口径）；YAML 写 null → 原生动态分辨率 no-resize baseline。
        image_size = bl["image_size"] if "image_size" in bl else SFT_IMAGE_SIZE
        print(f"\n[Loading] {bl['name']} on {device} (image_size={image_size})")
        model, processor = load_baseline_model(bl["model_path"], device, qwen_max_pixels)
        # system 与 SFT/plain 分支同源（config.system_prompt），保证 baseline 同口径对照。
        infer_fn = build_infer_fn("baseline", processor, config["generation"], image_size, eval_system)
        all_results[bl["name"]] = run_model_evaluation(
            bl["name"], model, processor, infer_fn, config, out_dir_base, use_cache=not args.no_cache
        )
        free_model(model, bl["name"])

    print_summary(all_results)
    summary_path = os.path.join(out_dir_base, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
