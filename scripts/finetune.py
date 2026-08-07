#!/usr/bin/env python3
"""Finetune tokenizers from MAISI pretrained weights on OpenMind dataset.

This script supports finetuning all tokenizer types:
- AE: Autoencoder (encoder/decoder only, no VAE sampling)
- VAE: Variational Autoencoder (MAISI-style with separate quant_conv)
- VQ: Vector Quantized VAE
- FSQ: Finite Scalar Quantization VAE
- RESFSQ: Residual FSQ VAE
- LFQ: Lookup-Free Quantization VAE

Training strategy (20 epochs total):
- Epochs 1-5: Warmup phase - reconstruction loss only (L1)
- Epochs 6-20: Full training - reconstruction + perceptual + adversarial losses

Usage:
    # Finetune VAE from MAISI weights
    python scripts/finetune.py --type VAE --maisi_weights weights/autoencoder_v2.pt

    # Finetune VQ-VAE (initializes encoder/decoder from MAISI, random quantizer)
    python scripts/finetune.py --type VQ --maisi_weights weights/autoencoder_v2.pt

    # Full training without MAISI weights
    python scripts/finetune.py --type FSQ --epochs 100
"""

import argparse
import inspect
import os
import warnings
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from medtokenizers import get_config
from medtokenizers.networks.continuous import ContinuousTokenizer
from medtokenizers.networks.discrete import DiscreteTokenizer
from medtokenizers.networks.nvidia_maisi import (
    convert_nvidia_weights,
)
from medtokenizers.training import LossConfig, Trainer
from medtokenizers.training.callbacks import Checkpoint, Logger, ReconstructionLogger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

# Ensure project root is in path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading import get_loaders

warnings.filterwarnings("ignore", ".*pixdim*.")
logger = get_logger(__name__)


def load_maisi_weights(weights_path: str) -> dict:
    """Load NVIDIA MAISI weights and convert to medtokenizers format."""
    # weights_only=True: MAISI checkpoints are plain tensor state dicts.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "unet_state_dict" in state:
        state = state["unet_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    return convert_nvidia_weights(state)


def transfer_encoder_decoder_weights(
    source_state: dict, target_model: torch.nn.Module, verbose: bool = False
) -> tuple[list, list]:
    """Transfer encoder/decoder weights from MAISI to target model.

    Only transfers weights that exist in both source and target.
    Skips quant_conv and quantizer-specific weights.
    """
    target_state = target_model.state_dict()
    transferred = []
    skipped = []

    for key in source_state:
        if (
            key.startswith("encoder.")
            or key.startswith("decoder.")
            or key.startswith("post_quant_conv")
        ):
            if key in target_state:
                if source_state[key].shape == target_state[key].shape:
                    target_state[key] = source_state[key]
                    transferred.append(key)
                else:
                    skipped.append(
                        f"{key} (shape mismatch: {source_state[key].shape} vs {target_state[key].shape})"
                    )
            else:
                skipped.append(f"{key} (not in target)")

    target_model.load_state_dict(target_state, strict=False)

    if verbose:
        print(f"Transferred {len(transferred)} weights")
        if skipped:
            print(f"Skipped {len(skipped)} weights:")
            for s in skipped[:10]:
                print(f"  - {s}")

    return transferred, skipped


def build_continuous_model(args):
    """Build ContinuousTokenizer (AE or VAE)."""
    model_kwargs = {
        "dim": 3,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "z_channels": args.z_channels,
        "latent_channels": args.latent_channels,
        "channels": args.channels,
        "channels_mult": tuple(args.channels_mult),
        "num_res_blocks": args.num_res_blocks,
        "attn_resolutions": tuple(args.attn_resolutions),
        "dropout": args.dropout,
        "resolution": args.crop_size or args.resolution,
        "spatial_compression": args.spatial_compression,
        "formulation": args.type,
        "use_encoder_mid": args.use_encoder_mid,
        "use_output_nonlinearity": args.use_output_nonlinearity,
        "decoder_blocks_per_stage": args.decoder_blocks_per_stage,
        "separate_quant_conv": args.separate_quant_conv,
    }
    return ContinuousTokenizer(**model_kwargs), model_kwargs


def build_discrete_model(args):
    """Build DiscreteTokenizer (VQ, FSQ, RESFSQ, LFQ)."""
    model_kwargs = {
        "dim": 3,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "z_channels": args.z_channels,
        "embedding_dim": args.embedding_dim,
        "channels": args.channels,
        "channels_mult": tuple(args.channels_mult),
        "num_res_blocks": args.num_res_blocks,
        "attn_resolutions": tuple(args.attn_resolutions),
        "dropout": args.dropout,
        "resolution": args.crop_size or args.resolution,
        "spatial_compression": args.spatial_compression,
        "quantizer": args.type,
        "use_encoder_mid": args.use_encoder_mid,
        "use_output_nonlinearity": args.use_output_nonlinearity,
        "decoder_blocks_per_stage": args.decoder_blocks_per_stage,
        "num_codebooks": args.num_codebooks,
        "num_embeddings": args.num_embeddings,
        "codebook_dim": args.codebook_dim,
        "codebook_size": args.codebook_size,
        "levels": args.levels,
        "beta": args.beta,
        "entropy_loss_weight": args.entropy_loss_weight,
        "commitment_loss_weight": args.commitment_loss_weight,
        "use_norm": args.use_norm,
    }
    return DiscreteTokenizer(**model_kwargs), model_kwargs


def build_model(args):
    """Build appropriate model based on type."""
    if args.type in ["AE", "VAE"]:
        model, model_kwargs = build_continuous_model(args)
    else:
        model, model_kwargs = build_discrete_model(args)

    if args.maisi_weights:
        logger.info(f"Loading MAISI weights from {args.maisi_weights}")
        maisi_state = load_maisi_weights(args.maisi_weights)

        if args.type == "VAE" and args.separate_quant_conv:
            missing, unexpected = model.load_state_dict(maisi_state, strict=False)
            logger.info(
                f"Loaded MAISI weights: {len(maisi_state) - len(missing)} matched, {len(missing)} missing, {len(unexpected)} unexpected"
            )
        else:
            transferred, skipped = transfer_encoder_decoder_weights(
                maisi_state, model, verbose=True
            )
            logger.info(
                f"Transferred {len(transferred)} encoder/decoder weights from MAISI"
            )

    return model, model_kwargs


def build_optimizer(model, args) -> torch.optim.Optimizer:
    """Build optimizer with optional layer-wise learning rates."""
    use_fused = (
        torch.cuda.is_available()
        and "fused" in inspect.signature(torch.optim.AdamW).parameters
    )

    if args.pretrained_lr_mult != 1.0 and args.maisi_weights:
        encoder_params = []
        decoder_params = []
        other_params = []

        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            elif name.startswith("decoder."):
                decoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": encoder_params, "lr": args.lr * args.pretrained_lr_mult},
            {"params": decoder_params, "lr": args.lr * args.pretrained_lr_mult},
            {"params": other_params, "lr": args.lr},
        ]
        return torch.optim.AdamW(param_groups, fused=use_fused)

    return torch.optim.AdamW(model.parameters(), lr=args.lr, fused=use_fused)


def build_schedulers(optimizer, args, num_training_steps: int, num_warmup_steps: int):
    """Build main and warmup schedulers."""
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=args.min_lr
    )

    warmup_scheduler = None
    if num_warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps
        )

    return main_scheduler, warmup_scheduler


def build_loss_config(args) -> LossConfig:
    """Build LossConfig for staged training.

    Stage 1 (warmup): reconstruction only
    Stage 2: reconstruction + perceptual + adversarial
    """
    return LossConfig(
        l1_weight=args.l1_weight,
        quant_weight=args.quant_weight,
        vgg_weight=0.0,
        gram_weight=0.0,
        laplacian_weight=args.laplacian_weight,
        stage1_epochs=args.warmup_epochs,
        stage1_vgg_weight=0.0,
        stage1_gram_weight=0.0,
        stage2_vgg_weight=args.vgg_weight,
        stage2_gram_weight=args.gram_weight,
        stage2_warmup_epochs=args.stage2_warmup_epochs,
        laplacian_start_epoch=args.warmup_epochs,
    )


def main():
    args = parse_args()

    mixed_precision = None
    if args.mixed_precision.lower() == "bf16":
        mixed_precision = "bf16"
    elif args.mixed_precision.lower() == "fp16":
        mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision=mixed_precision,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.seed is not None:
        set_seed(args.seed)
        accelerator.print(f"Random seed set to {args.seed}")

    cfg = get_config()
    if args.allow_tf32 is None:
        args.allow_tf32 = cfg.allow_tf32

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision") and args.allow_tf32:
        torch.set_float32_matmul_precision("high")

    model, model_kwargs = build_model(args)
    optimizer = build_optimizer(model, args)

    train_loader, val_loader = get_loaders(
        batch_size=args.batch_size,
        augment=True,
        cache=True,
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        crops_per_volume=args.crops_per_volume,
        reslice_prob=args.reslice_prob,
        sampling_strategy="center",
        center_bias=0.3,
        num_workers=args.num_workers,
        random_crop_size=args.random_crop_size,
        crop_size_choices=args.crop_size_choices,
        crop_size_weights=args.crop_size_weights,
        anisotropic_crops=args.anisotropic_crops,
        spacing_range=tuple(args.spacing_range) if args.spacing_range else None,
    )

    steps_per_epoch = args.steps_per_epoch
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    accelerator.print(
        f"Finetuning {args.type}: {args.epochs} epochs x {steps_per_epoch} steps/epoch = "
        f"{num_training_steps} total steps, {num_warmup_steps} warmup steps"
    )

    main_scheduler, warmup_scheduler = build_schedulers(
        optimizer, args, num_training_steps, num_warmup_steps
    )

    loss_config = build_loss_config(args)

    from medtokenizers.training.losses import CombinedPerceptual

    loss_fn = CombinedPerceptual(
        dim=3,
        input_channels=args.in_channels,
        reconstruction_weight=args.l1_weight,
        vgg_weight=args.vgg_weight,
        gram_weight=args.gram_weight,
        quantization_weight=args.quant_weight
        if args.type not in ["AE", "VAE"]
        else args.kl_weight,
        lpips_slice_stride=args.lpips_slice_stride,
        ssim_weight=args.ssim_weight,
        use_ssim_instead_of_lpips=args.use_ssim,
        adversarial_weight=args.disc_weight,
        discriminator_start_iter=args.disc_start_epoch * args.steps_per_epoch,
        reconstruction_type=args.recon_type,
        use_lecam=True,
        lecam_weight=args.lecam_weight,
    )

    # Create discriminator optimizer if GAN is enabled
    disc_optimizer = None
    if (
        args.disc_weight > 0
        and hasattr(loss_fn, "discriminator")
        and loss_fn.discriminator is not None
    ):
        disc_optimizer = torch.optim.AdamW(
            loss_fn.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9)
        )

    wandb_project = args.wandb_project or cfg.wandb_project
    wandb_entity = args.wandb_entity or cfg.wandb_entity

    run_name = args.name or f"finetune-{args.type.lower()}"

    accelerator.init_trackers(
        wandb_project,
        config={**vars(args), **model_kwargs},
        init_kwargs={
            "wandb": {
                "entity": wandb_entity,
                "name": run_name,
                "settings": wandb.Settings(start_method="fork"),
            }
        },
    )

    output_dir = os.path.join(args.logdir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint")
    callbacks = [
        Logger(),
        Checkpoint(
            filepath=checkpoint_path,
            monitor="val_total" if val_loader else "train_loss",
            save_best_only=True,
            mode="min",
        ),
        ReconstructionLogger(num_samples=4, every_n_epochs=1),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        loss_fn=loss_fn,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        callbacks=callbacks,
        channels_last=True,
        loss_config=loss_config,
        scheduler=main_scheduler,
        warmup_scheduler=warmup_scheduler,
        warmup_steps=num_warmup_steps,
        log_every_steps=1,
        nan_threshold=0.1,
        enable_compile_train_step=args.compile,
        enable_gradient_checkpointing=args.gradient_checkpointing,
        disc_optimizer=disc_optimizer,
        model_ema_decay=args.model_ema_decay,
    )

    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        steps_per_epoch=steps_per_epoch,
        val_interval=args.val_interval,
        max_val_batches=args.max_val_batches,
    )

    trainer.end_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune tokenizers from MAISI pretrained weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--name", type=str, default=None, help="Run name (default: finetune-{type})"
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./outputs")

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["AE", "VAE", "VQ", "LFQ", "FSQ", "RESFSQ"],
        help="Tokenizer type to finetune",
    )
    parser.add_argument(
        "--maisi_weights",
        type=str,
        default=None,
        help="Path to MAISI pretrained weights",
    )

    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Epochs for reconstruction-only warmup",
    )
    parser.add_argument(
        "--stage2_warmup_epochs",
        type=int,
        default=5,
        help="Epochs to warmup perceptual/adversarial losses",
    )
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--pretrained_lr_mult",
        type=float,
        default=0.1,
        help="LR multiplier for pretrained encoder/decoder (when using MAISI weights)",
    )
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Max validation batches (None = full dataset)",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "no"]
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--data_dir", type=str, required=True, help="OpenMind dataset directory"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=96,
        help="Base crop size (used when random_crop_size=False)",
    )
    parser.add_argument("--crops_per_volume", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--random_crop_size",
        action="store_true",
        help="Randomly sample crop size from crop_size_choices",
    )
    parser.add_argument(
        "--crop_size_choices",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 80, 96, 112, 128],
        help="Valid crop sizes when random_crop_size=True",
    )
    parser.add_argument(
        "--crop_size_weights",
        nargs="+",
        type=float,
        default=None,
        help="Sampling weights for crop sizes (default: favor 64-96)",
    )
    parser.add_argument(
        "--anisotropic_crops",
        action="store_true",
        help="Sample crop size independently per axis (e.g., 16x80x128)",
    )
    parser.add_argument(
        "--reslice_prob",
        type=float,
        default=0.5,
        help="Probability of reslicing (0=native, 1=always reslice)",
    )
    parser.add_argument(
        "--spacing_range",
        nargs=2,
        type=float,
        default=None,
        help="Random target spacing range in mm (e.g., --spacing_range 1.0 3.0)",
    )

    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-4,
        help="KL divergence weight for VAE (passed via quantization channel)",
    )
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--quant_weight", type=float, default=1.0)
    parser.add_argument(
        "--vgg_weight",
        type=float,
        default=0.5,
        help="VGG perceptual loss weight (stage 2)",
    )
    parser.add_argument(
        "--gram_weight",
        type=float,
        default=0.1,
        help="Gram style loss weight (stage 2)",
    )
    parser.add_argument("--laplacian_weight", type=float, default=0.1)
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.0,
        help="SSIM loss weight (can be used alongside LPIPS)",
    )
    parser.add_argument(
        "--use_ssim",
        action="store_true",
        help="Use SSIM instead of LPIPS for perceptual loss (much faster, good for medical imaging)",
    )
    parser.add_argument(
        "--lpips_slice_stride",
        type=int,
        default=12,
        help="Stride for LPIPS slice sampling in 3D (higher = faster but less accurate)",
    )

    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--channels_mult", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", nargs="*", type=int, default=[])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spatial_compression", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--use_encoder_mid", action="store_true", default=False)
    parser.add_argument("--use_output_nonlinearity", action="store_true", default=False)
    parser.add_argument(
        "--decoder_blocks_per_stage", nargs="+", type=int, default=[2, 2, 0]
    )
    parser.add_argument("--separate_quant_conv", action="store_true", default=True)
    parser.add_argument(
        "--no_separate_quant_conv", dest="separate_quant_conv", action="store_false"
    )

    parser.add_argument("--z_channels", type=int, default=4)
    parser.add_argument("--latent_channels", type=int, default=4)

    parser.add_argument("--embedding_dim", type=int, default=6)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--levels", nargs="+", type=int, default=[8, 5, 5, 5])
    parser.add_argument("--num_embeddings", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--codebook_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument(
        "--use_norm", action="store_true", help="L2 normalize VQ codebook"
    )

    parser.add_argument(
        "--disc_weight",
        type=float,
        default=0.02,
        help="Discriminator adversarial loss weight",
    )
    parser.add_argument(
        "--disc_start_epoch",
        type=int,
        default=5,
        help="Epoch to start discriminator training",
    )
    parser.add_argument(
        "--model_ema_decay", type=float, default=0.9999, help="Model EMA decay rate"
    )
    parser.add_argument(
        "--lecam_weight", type=float, default=0.001, help="LeCAM regularization weight"
    )
    parser.add_argument(
        "--recon_type",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Reconstruction loss type",
    )

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--allow_tf32", action=argparse.BooleanOptionalAction, default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
