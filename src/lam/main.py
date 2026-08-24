"""Command-line entrypoint for feature-space LAM pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .data import LAMCollator, LAMPairDataset
from .model import FeatureLAM


def _parse_devices(value: str):
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(item) for item in value.split(",") if item.strip()]
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, default=Path("data/lam/v0/train.json"))
    parser.add_argument("--val-json", type=Path, default=Path("data/lam/v0/test.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--vision-model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lam"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logger-version", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of per-device micro-batches to accumulate before one optimizer step.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--image-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=16)
    parser.add_argument("--dec-blocks", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-latent", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=2e-4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--feature-loss", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--val-every-n-steps", type=int, default=1000)
    return parser


def _load_image_processor(vision_model_path: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(vision_model_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError(
            f"No image_processor found in AutoProcessor at {vision_model_path}"
        )
    return image_processor


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.accumulate_grad_batches <= 0:
        raise ValueError(
            "batch-size and accumulate-grad-batches must be positive; num-workers must be non-negative"
        )
    if args.val_every_n_steps <= 0:
        raise ValueError("val-every-n-steps must be positive")
    if args.max_epochs <= 0 and args.max_steps <= 0:
        raise ValueError("Set max-epochs or max-steps to a positive value")

    pl.seed_everything(args.seed, workers=True)
    image_processor = _load_image_processor(args.vision_model_path)
    train_dataset = LAMPairDataset(args.train_json, limit=args.train_limit)
    val_dataset = LAMPairDataset(
        args.val_json or args.train_json,
        limit=args.val_limit if args.val_json else min(args.val_limit or 256, len(train_dataset)),
    )
    collator = LAMCollator(
        image_root=args.image_root,
        image_processor=image_processor,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    model = FeatureLAM(
        args.vision_model_path,
        model_dim=args.model_dim,
        image_dim=args.image_dim,
        latent_dim=args.latent_dim,
        patch_size=args.patch_size,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        num_heads=args.num_heads,
        num_latent=args.num_latent,
        dropout=args.dropout,
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        feature_loss=args.feature_loss,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="lam-{epoch:03d}-{step:08d}-{val_loss:.5f}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        save_on_train_epoch_end=False,
    )
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version=args.logger_version,
    )
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=_parse_devices(args.devices),
        strategy="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_check_interval=args.val_every_n_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=0.3,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
