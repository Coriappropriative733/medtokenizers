"""Training script for medical tokenizers using the Trainer API.

This script handles only orchestration:
- CLI argument parsing
- Model, optimizer, scheduler construction
- Dataloader creation
- Trainer instantiation and fit() call

All training logic (loop, checkpointing, logging, NaN handling) is in the Trainer class.
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
from medtokenizers.training import LossConfig, Trainer
from medtokenizers.training.callbacks import Checkpoint, Logger
from medtokenizers.training.losses import Combined
from medtokenizers.training.losses.compound import VAEGANLoss, VQGANLoss

# ============================================================================
# Memory optimization settings
# ============================================================================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch._inductor.config.triton.cudagraphs = False

# Import data loading utilities
try:
    from scripts.data_loading import get_2d_loaders, get_loaders
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.data_loading import get_2d_loaders, get_loaders

warnings.filterwarnings("ignore", ".*pixdim*.")
logger = get_logger(__name__)


def build_model(args):
    """Build the tokenizer model based on args.

    Returns:
        Tuple of (model, model_kwargs)
    """
    model_resolution = args.crop_size if args.crop_size else args.resolution

    if args.type in ["AE", "VAE"]:
        model_kwargs = {
            "dim": 2 if args.twod else 3,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "z_channels": args.z_channels,
            "z_factor": 2 if args.type == "VAE" else 1,
            "latent_channels": args.latent_channels,
            "channels": args.channels,
            "channels_mult": args.channels_mult,
            "num_res_blocks": args.num_res_blocks,
            "attn_resolutions": args.attn_resolutions,
            "dropout": args.dropout,
            "resolution": model_resolution,
            "spatial_compression": args.spatial_compression,
            "patch_size": args.patch_size,
            "patch_method": args.patch_method,
            "voronoi_jitter": args.voronoi_jitter,
            "formulation": args.type,
        }
        return ContinuousTokenizer(**model_kwargs), model_kwargs
    else:  # VQ, LFQ, FSQ, RESFSQ
        model_kwargs = {
            "dim": 2 if args.twod else 3,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "z_channels": args.z_channels,
            "embedding_dim": args.embedding_dim,
            "channels": args.channels,
            "channels_mult": args.channels_mult,
            "num_res_blocks": args.num_res_blocks,
            "attn_resolutions": args.attn_resolutions,
            "dropout": args.dropout,
            "resolution": model_resolution,
            "spatial_compression": args.spatial_compression,
            "patch_size": args.patch_size,
            "patch_method": args.patch_method,
            "voronoi_jitter": args.voronoi_jitter,
            "quantizer": args.type,
            # LFQ entropy regularization is single-codebook by construction, so it
            # must always use num_codebooks=1 (see LFQuantizer in modules/quant.py).
            "num_codebooks": 1 if args.type == "LFQ" else args.num_codebooks,
            "num_embeddings": args.num_embeddings,
            "codebook_dim": args.codebook_dim,
            "codebook_size": args.codebook_size,
            "levels": args.levels,
            "beta": args.beta,
            "entropy_loss_weight": args.entropy_loss_weight,
            "commitment_loss_weight": args.commitment_loss_weight,
            "default_temp": args.quant_temp,
            "use_norm": args.use_norm,
            "mcq_heads": args.mcq_heads,
        }
        return DiscreteTokenizer(**model_kwargs), model_kwargs


def build_optimizer(model, args) -> torch.optim.Optimizer:
    """Build the optimizer."""
    use_fused = (
        torch.cuda.is_available()
        and "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, fused=use_fused)


def build_schedulers(optimizer, args, num_training_steps: int, num_warmup_steps: int):
    """Build main and warmup schedulers."""
    # Main scheduler
    if args.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=args.min_lr
        )
    elif args.scheduler == "linear":
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.min_lr / args.lr,
            total_iters=num_training_steps - num_warmup_steps,
        )
    elif args.scheduler == "polynomial":
        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=num_training_steps - num_warmup_steps,
            power=args.scheduler_power,
        )
    else:  # constant
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # Warmup scheduler
    warmup_scheduler = None
    if num_warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps
        )

    return main_scheduler, warmup_scheduler


def build_loss_config(args) -> LossConfig:
    """Build LossConfig from args."""
    return LossConfig(
        l1_weight=args.l1_weight,
        quant_weight=args.quant_weight,
        vgg_weight=0.0,  # Managed via stage scheduling
        gram_weight=0.0,
        laplacian_weight=args.laplacian_weight,
        stage1_epochs=args.stage1_epochs,
        stage1_vgg_weight=args.stage1_vgg_weight,
        stage1_gram_weight=args.stage1_gram_weight,
        stage2_vgg_weight=args.stage2_vgg_weight,
        stage2_gram_weight=args.stage2_gram_weight,
        stage2_warmup_epochs=args.stage2_warmup_epochs,
        laplacian_start_epoch=args.laplacian_start_epoch,
    )


def validate_resume_path(resume_path: str) -> str:
    """Validate that resume path is an Accelerate checkpoint directory.

    Args:
        resume_path: Path to checkpoint

    Returns:
        Validated directory path

    Raises:
        ValueError: If path is a legacy .pt file or doesn't exist
    """
    path = Path(resume_path)

    # Check for legacy .pt file
    if path.suffix in (".pt", ".pth", ".ckpt"):
        raise ValueError(
            f"Legacy checkpoint file detected: {resume_path}\n"
            "This script now uses Accelerate directory checkpoints ONLY.\n"
            "Expected a checkpoint DIRECTORY, not a .pt file.\n"
            "To migrate old checkpoints, load the .pt file manually and "
            "save via Trainer.save_checkpoint()."
        )

    if not path.exists():
        raise ValueError(f"Checkpoint path does not exist: {resume_path}")

    if not path.is_dir():
        raise ValueError(
            f"Expected a checkpoint directory, got file: {resume_path}\n"
            "Accelerate checkpoints are directories containing model, optimizer, "
            "scheduler states, and metadata.json."
        )

    # Check for required files
    metadata_file = path / "metadata.json"
    if not metadata_file.exists():
        raise ValueError(
            f"Invalid checkpoint directory: {resume_path}\n"
            "Missing metadata.json. This doesn't appear to be a valid "
            "Accelerate checkpoint created by Trainer.save_checkpoint()."
        )

    return str(path)


def main():
    """Main training entry point."""
    args = parse_args()

    # Validate resume path early
    resume_dir = None
    if args.resume:
        resume_dir = validate_resume_path(args.resume)

    # -------------------------------------------------------------------------
    # Initialize Accelerator
    # -------------------------------------------------------------------------
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

    # Configure determinism
    if torch.cuda.is_available():
        if args.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            accelerator.print("Deterministic mode enabled (slower but reproducible)")
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        accelerator.print(f"Random seed set to {args.seed}")
    elif args.deterministic:
        accelerator.print("Warning: --deterministic requires --seed. Setting seed=42.")
        set_seed(42)

    # Configure TF32
    cfg = get_config()
    if args.allow_tf32 is None:
        args.allow_tf32 = cfg.allow_tf32
    if args.channels_last is None:
        args.channels_last = cfg.channels_last

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision") and args.allow_tf32:
        torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------------------
    # Build model and optimizer
    # -------------------------------------------------------------------------
    model, model_kwargs = build_model(args)
    optimizer = build_optimizer(model, args)

    # -------------------------------------------------------------------------
    # Build dataloaders
    # -------------------------------------------------------------------------
    if args.twod:
        train_loader, val_loader = get_2d_loaders(
            batch_size=args.batch_size,
            cache_size=args.cache_size,
            test_run=args.test_run,
            data_dir=args.data_dir,
            num_workers=getattr(args, "num_workers", None),
            prefetch_factor=getattr(args, "prefetch_factor", None),
            persistent_workers=None
            if getattr(args, "num_workers", None) is None
            else not args.no_persistent_workers,
        )
    else:
        train_loader, val_loader = get_loaders(
            batch_size=args.batch_size,
            lowres=args.lowres,
            augment=getattr(args, "augment", True),
            cache=getattr(args, "cache", True),
            resize_threshold=getattr(args, "resize_threshold", 256),
            data_dir=args.data_dir,
            crop_size=getattr(args, "crop_size", None),
            crops_per_volume=getattr(args, "crops_per_volume", 4),
            reslice_prob=getattr(args, "reslice_prob", 1.0),
            sampling_strategy=getattr(args, "sampling_strategy", "center"),
            center_bias=getattr(args, "center_bias", 0.3),
            foreground_threshold=getattr(args, "foreground_threshold", 0.01),
            num_workers=getattr(args, "num_workers", None),
            prefetch_factor=getattr(args, "prefetch_factor", None),
            persistent_workers=None
            if getattr(args, "num_workers", None) is None
            else not args.no_persistent_workers,
        )

    # -------------------------------------------------------------------------
    # Build schedulers
    # -------------------------------------------------------------------------
    steps_per_epoch = args.steps_per_epoch
    if args.test_run:
        steps_per_epoch = 1

    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    accelerator.print(
        f"Training config: {args.epochs} epochs x {steps_per_epoch} steps/epoch = "
        f"{num_training_steps} total steps, {num_warmup_steps} warmup steps"
    )

    main_scheduler, warmup_scheduler = build_schedulers(
        optimizer, args, num_training_steps, num_warmup_steps
    )

    # -------------------------------------------------------------------------
    # Build loss config and loss function
    # -------------------------------------------------------------------------
    loss_config = build_loss_config(args)
    dim = 2 if args.twod else 3

    disc_optimizer = None
    if args.type == "VAE":
        loss_fn = VAEGANLoss(
            dim=dim,
            input_channels=args.in_channels,
            reconstruction_weight=args.l1_weight,
            kl_weight=1e-6,
            perceptual_weight=args.lpips_weight,
            adversarial_weight=args.disc_weight,
            discriminator_start_iter=args.disc_start_epoch * args.steps_per_epoch,
            lpips_slice_stride=4 if not args.twod else 1,
            reconstruction_type=args.recon_type,
            use_lecam=True,
            lecam_weight=args.lecam_weight,
        )
    elif args.type in ["VQ", "FSQ", "LFQ", "RESFSQ"]:
        loss_fn = VQGANLoss(
            dim=dim,
            input_channels=args.in_channels,
            reconstruction_weight=args.l1_weight,
            quantization_weight=args.quant_weight,
            perceptual_weight=args.lpips_weight,
            adversarial_weight=args.disc_weight,
            discriminator_start_iter=args.disc_start_epoch * args.steps_per_epoch,
            lpips_slice_stride=4 if not args.twod else 1,
            reconstruction_type=args.recon_type,
            use_lecam=True,
            lecam_weight=args.lecam_weight,
        )
    else:  # AE
        if args.lpips_weight > 0 or args.disc_weight > 0:
            # AE with perceptual/adversarial losses: use VAEGANLoss with kl_weight=0.
            # AE model returns no kl_loss key, so KL term is naturally skipped.
            loss_fn = VAEGANLoss(
                dim=dim,
                input_channels=args.in_channels,
                reconstruction_weight=args.l1_weight,
                kl_weight=0.0,
                perceptual_weight=args.lpips_weight,
                adversarial_weight=args.disc_weight,
                discriminator_start_iter=args.disc_start_epoch * args.steps_per_epoch,
                lpips_slice_stride=4 if not args.twod else 1,
                reconstruction_type=args.recon_type,
                use_lecam=True,
                lecam_weight=args.lecam_weight,
            )
        else:
            loss_fn = Combined(
                reconstruction_weight=args.l1_weight,
                quantization_weight=0.0,
                reconstruction_type=args.recon_type,
            )

    # Create discriminator optimizer if GAN is enabled
    if (
        args.disc_weight > 0
        and hasattr(loss_fn, "discriminator")
        and loss_fn.discriminator is not None
    ):
        use_fused = (
            torch.cuda.is_available()
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        disc_optimizer = torch.optim.AdamW(
            loss_fn.discriminator.parameters(),
            lr=args.lr,
            betas=(0.5, 0.9),
            fused=use_fused,
        )

    # -------------------------------------------------------------------------
    # Initialize wandb via Accelerate
    # -------------------------------------------------------------------------
    wandb_project = getattr(args, "wandb_project", None) or cfg.wandb_project
    wandb_entity = getattr(args, "wandb_entity", None) or cfg.wandb_entity

    accelerator.init_trackers(
        wandb_project,
        config=vars(args),
        init_kwargs={
            "wandb": {
                "entity": wandb_entity,
                "name": args.name,
                "settings": wandb.Settings(start_method="fork"),
            }
        },
    )

    # -------------------------------------------------------------------------
    # Build callbacks
    # -------------------------------------------------------------------------
    output_dir = os.path.join(args.logdir, args.name)
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
    ]

    # -------------------------------------------------------------------------
    # Create Trainer and fit
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        loss_fn=loss_fn,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        callbacks=callbacks,
        channels_last=args.channels_last or False,
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

    # Fit - all training logic is handled by Trainer
    # accelerator.prepare() happens inside Trainer.fit()
    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        steps_per_epoch=steps_per_epoch,
        val_interval=args.val_interval,
        resume=resume_dir,
    )

    # End training
    trainer.end_training()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train medical tokenizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic training args
    parser.add_argument("--name", type=str, required=True, help="Name of WandB run")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "polynomial", "constant"],
    )
    parser.add_argument("--scheduler_power", type=float, default=0.9)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "no"],
    )
    parser.add_argument("--logdir", type=str, default="./")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Accelerate checkpoint DIRECTORY to resume from",
    )
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--lowres", action="store_true")
    parser.add_argument(
        "--no_augment", dest="augment", action="store_false", default=True
    )
    parser.add_argument("--no_cache", dest="cache", action="store_false", default=True)
    parser.add_argument("--resize_threshold", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--twod", action="store_true")
    parser.add_argument("--cache_size", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--no_persistent_workers", action="store_true", default=False)

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=4.0)
    parser.add_argument("--quant_weight", type=float, default=1.0)
    parser.add_argument(
        "--lpips_weight", type=float, default=0.1, help="LPIPS perceptual loss weight"
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
        default=50,
        help="Epoch to start discriminator training",
    )
    parser.add_argument(
        "--model_ema_decay", type=float, default=0.9999, help="Model EMA decay rate"
    )
    parser.add_argument(
        "--recon_type",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Reconstruction loss type",
    )
    parser.add_argument(
        "--lecam_weight", type=float, default=0.001, help="LeCAM regularization weight"
    )

    # Model type
    parser.add_argument(
        "--type",
        type=str,
        default="FSQ",
        choices=["AE", "VAE", "VQ", "LFQ", "FSQ", "RESFSQ"],
    )
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)

    # Network architecture
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--channels_mult", nargs="+", type=int, default=[2, 4, 4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", nargs="+", type=int, default=[32])
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--spatial_compression", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument(
        "--patch_method",
        type=str,
        default="haar",
        choices=["haar", "rearrange", "voronoi"],
    )
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--crops_per_volume", type=int, default=4)
    parser.add_argument("--reslice_prob", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="center",
        choices=["uniform", "center", "foreground"],
    )
    parser.add_argument("--center_bias", type=float, default=0.3)
    parser.add_argument("--foreground_threshold", type=float, default=0.01)
    parser.add_argument("--voronoi_jitter", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Continuous tokenizer (AE/VAE)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--z_channels", type=int, default=16)

    # Discrete tokenizer
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--levels", nargs="+", type=int, default=[8, 8, 8, 5, 5, 5])

    # VQ specific
    parser.add_argument("--num_embeddings", type=int, default=8192)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--use_norm", action="store_true")

    # LFQ specific
    parser.add_argument("--codebook_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--quant_temp", type=float, default=0.01)
    parser.add_argument("--mcq_heads", type=int, default=1)

    # Multi-stage schedule
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_warmup_epochs", type=int, default=25)
    parser.add_argument("--stage1_vgg_weight", type=float, default=0.0)
    parser.add_argument("--stage2_vgg_weight", type=float, default=1.0)
    parser.add_argument("--stage1_gram_weight", type=float, default=0.0)
    parser.add_argument("--stage2_gram_weight", type=float, default=1.0)
    parser.add_argument("--laplacian_weight", type=float, default=0.0)
    parser.add_argument("--laplacian_start_epoch", type=int, default=0)

    # Compilation and memory optimization
    parser.add_argument(
        "--compile", action="store_true", help="Compile model with torch.compile"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory at cost of compute",
    )
    parser.add_argument(
        "--allow_tf32", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--channels_last", action=argparse.BooleanOptionalAction, default=None
    )

    args = parser.parse_args()

    # Adjust z_channels for discrete tokenizers
    if args.type in ["VQ", "LFQ", "FSQ", "RESFSQ"]:
        args.z_channels = 256

    # Validation
    if args.stage1_epochs >= args.epochs:
        raise ValueError("stage1_epochs must be less than total epochs")
    if args.test_run:
        args.gradient_accumulation_steps = 1

    return args


if __name__ == "__main__":
    main()
