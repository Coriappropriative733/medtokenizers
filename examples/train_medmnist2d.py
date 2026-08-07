"""Example: Training a 2D tokenizer on MedMNIST datasets.

This example demonstrates how to train tokenizers on 2D medical imaging datasets
from the MedMNIST collection. MedMNIST+ provides standardized 2D datasets at
various resolutions (28, 64, 128, 224).

Available 2D datasets:
- PathMNIST: Colorectal histology (9 classes) - 3 channels
- ChestMNIST: Chest X-ray disease (14 classes) - 1 channel
- DermaMNIST: Skin lesion (7 classes) - 3 channels
- OCTMNIST: Retinal OCT (4 classes) - 1 channel
- PneumoniaMNIST: Pneumonia (2 classes) - 1 channel
- RetinaMNIST: Diabetic retinopathy (5 classes) - 3 channels
- BreastMNIST: Breast ultrasound (2 classes) - 1 channel
- BloodMNIST: Blood cell (8 classes) - 3 channels
- TissueMNIST: Kidney tissue (8 classes) - 1 channel
- Organ{A,C,S}MNIST: Organ slices (11 classes) - 1 channel

Usage:
    python train_medmnist2d.py --dataset pathmnist --epochs 50 --batch_size 128
    python train_medmnist2d.py --dataset chestmnist --quantizer FSQ --size 128
    python train_medmnist2d.py --dataset dermamnist --size 224
"""

import argparse
import logging
import os
import sys

import medmnist
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer
from medtokenizers.training import (
    Checkpoint,
    Combined,
    Logger,
    Trainer,
    VAEGANLoss,
    VQGANLoss,
)
from medtokenizers.training.callbacks import ReconstructionLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Seed NumPy RNG per worker for reproducible augmentation after fork."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        np.random.seed(worker_info.seed % (2**32))


DATASETS_2D = {
    "pathmnist": {
        "class": medmnist.PathMNIST,
        "info": medmnist.INFO["pathmnist"],
        "channels": 3,
        "default_size": 28,
        "n_samples": {"train": 89996, "val": 10004, "test": 7180},
    },
    "chestmnist": {
        "class": medmnist.ChestMNIST,
        "info": medmnist.INFO["chestmnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 78468, "val": 11219, "test": 22433},
    },
    "dermamnist": {
        "class": medmnist.DermaMNIST,
        "info": medmnist.INFO["dermamnist"],
        "channels": 3,
        "default_size": 28,
        "n_samples": {"train": 7007, "val": 1003, "test": 2005},
    },
    "octmnist": {
        "class": medmnist.OCTMNIST,
        "info": medmnist.INFO["octmnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 97477, "val": 10832, "test": 1000},
    },
    "pneumoniamnist": {
        "class": medmnist.PneumoniaMNIST,
        "info": medmnist.INFO["pneumoniamnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 4708, "val": 524, "test": 624},
    },
    "retinamnist": {
        "class": medmnist.RetinaMNIST,
        "info": medmnist.INFO["retinamnist"],
        "channels": 3,
        "default_size": 28,
        "n_samples": {"train": 1080, "val": 120, "test": 400},
    },
    "breastmnist": {
        "class": medmnist.BreastMNIST,
        "info": medmnist.INFO["breastmnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 546, "val": 78, "test": 156},
    },
    "bloodmnist": {
        "class": medmnist.BloodMNIST,
        "info": medmnist.INFO["bloodmnist"],
        "channels": 3,
        "default_size": 28,
        "n_samples": {"train": 11959, "val": 1712, "test": 3421},
    },
    "tissuemnist": {
        "class": medmnist.TissueMNIST,
        "info": medmnist.INFO["tissuemnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 165466, "val": 23640, "test": 47280},
    },
    "organamnist": {
        "class": medmnist.OrganAMNIST,
        "info": medmnist.INFO["organamnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 34561, "val": 6491, "test": 17778},
    },
    "organcmnist": {
        "class": medmnist.OrganCMNIST,
        "info": medmnist.INFO["organcmnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 12975, "val": 2392, "test": 8216},
    },
    "organsmnist": {
        "class": medmnist.OrganSMNIST,
        "info": medmnist.INFO["organsmnist"],
        "channels": 1,
        "default_size": 28,
        "n_samples": {"train": 13932, "val": 2452, "test": 8827},
    },
}


def normalize_to_01(data):
    """Normalize data to [0, 1] range."""
    data = data.astype(np.float32)
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val > 1e-8:
        data = (data - min_val) / (max_val - min_val)
    return data


def load_medmnist2d_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    download: bool = True,
    size: int = 28,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a MedMNIST 2D dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'pathmnist', 'chestmnist')
        data_dir: Directory to store/download data
        download: Whether to download the dataset if not present
        size: Image size (28, 64, 128, 224 for MedMNIST+)
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    data_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name.lower() not in DATASETS_2D:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASETS_2D.keys())}"
        )

    dataset_info = DATASETS_2D[dataset_name.lower()]
    dataset_class = dataset_info["class"]

    logger.info(f"Loading {dataset_name} (size={size}) from {data_dir}")

    npz_path = os.path.join(data_dir, f"{dataset_name}_{size}.npz")
    if os.path.exists(npz_path):
        logger.info(f"Loading from NPZ file: {npz_path}")
        data = np.load(npz_path)
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        images = data[images_key]
        labels = data.get(labels_key)

        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.squeeze(-1)
        images = normalize_to_01(images)

        logger.info(f"  Loaded {len(images)} samples, shape: {images.shape}")
        return images, labels

    try:
        dataset = dataset_class(
            root=data_dir,
            split=split,
            download=download,
            size=size,
        )

        if hasattr(dataset, "imgs"):
            images = dataset.imgs
            labels = dataset.labels.flatten() if hasattr(dataset, "labels") else None
        elif hasattr(dataset, "data"):
            images = dataset.data
            labels = dataset.labels.flatten() if hasattr(dataset, "labels") else None
        else:
            raise ValueError("Cannot access dataset images/labels")

        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.squeeze(-1)
        images = normalize_to_01(images)

        logger.info(f"  Loaded {len(images)} samples, shape: {images.shape}")
        return images, labels

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def create_medmnist2d_dataloaders(
    dataset_name: str,
    data_dir: str = "./data",
    download: bool = True,
    size: int = 28,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for MedMNIST 2D.

    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store data
        download: Whether to download the dataset
        size: Image size (28, 64, 128, or 224)
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_images, _ = load_medmnist2d_dataset(
        dataset_name, data_dir, download, size, "train"
    )
    val_images, _ = load_medmnist2d_dataset(dataset_name, data_dir, False, size, "val")

    # Convert to (N, C, H, W) format for PyTorch
    # MedMNIST 3-channel data comes as (N, H, W, C), 1-channel as (N, H, W)
    if train_images.ndim == 4:  # 3-channel: (N, H, W, C) -> (N, C, H, W)
        train_tensor = torch.from_numpy(train_images).permute(0, 3, 1, 2)
        val_tensor = torch.from_numpy(val_images).permute(0, 3, 1, 2)
    else:  # 1-channel: (N, H, W) -> (N, 1, H, W)
        train_tensor = torch.from_numpy(train_images).unsqueeze(1)
        val_tensor = torch.from_numpy(val_images).unsqueeze(1)

    logger.info(f"Train tensor shape: {train_tensor.shape}")
    logger.info(f"Val tensor shape: {val_tensor.shape}")

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


def create_2d_tokenizer(
    dataset_name: str,
    in_channels: int = 1,
    resolution: int = 28,
    spatial_compression: int = 8,
    tokenizer_type: str = "discrete",
    quantizer: str = "RESFSQ",
    formulation: str = "VAE",
    latent_channels: int = 4,
    z_channels: int = 64,
    embedding_dim: int = 16,
    channels: int = 64,
    **kwargs,
):
    """Create a 2D tokenizer configured for MedMNIST.

    Args:
        dataset_name: Name of the dataset (for logging)
        in_channels: Number of input channels
        resolution: Input spatial resolution
        spatial_compression: Downsampling factor
        tokenizer_type: 'discrete' or 'continuous'
        quantizer: Quantization method for discrete ('VQ', 'FSQ', 'LFQ', 'RESFSQ')
        formulation: Formulation for continuous ('VAE' or 'AE')
        latent_channels: Number of latent channels for continuous
        z_channels: Encoder output channels (before quant_conv)
        embedding_dim: Dimension of quantized embeddings
        channels: Base channel count for encoder/decoder
        **kwargs: Additional tokenizer arguments

    Returns:
        Configured tokenizer (DiscreteTokenizer or ContinuousTokenizer)
    """
    logger.info(f"Creating 2D {tokenizer_type} tokenizer for {dataset_name}")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Spatial compression: {spatial_compression}x")

    if tokenizer_type == "discrete":
        logger.info(f"  Quantizer: {quantizer}")
        quantizer_kwargs = {}

        if quantizer == "FSQ":
            levels = kwargs.pop("levels", [8, 8, 8, 5, 5, 5])
            effective_codes = np.prod(levels)
            logger.info(f"  FSQ levels: {levels}, effective codes: {effective_codes}")
            quantizer_kwargs["levels"] = levels
        elif quantizer == "RESFSQ":
            num_codebooks = kwargs.pop("num_codebooks", 2)
            levels = kwargs.pop("levels", [8, 8, 8])
            logger.info(f"  RESFSQ: {num_codebooks} codebooks, levels: {levels}")
            quantizer_kwargs["num_codebooks"] = num_codebooks
            quantizer_kwargs["levels"] = levels
        elif quantizer == "VQ":
            num_embeddings = kwargs.pop("num_embeddings", 8192)
            logger.info(f"  VQ: {num_embeddings} codebook entries")
            quantizer_kwargs["num_embeddings"] = num_embeddings
        elif quantizer == "LFQ":
            # LFQ entropy regularization is single-codebook by construction, so it
            # must always use num_codebooks=1 (see LFQuantizer in modules/quant.py).
            kwargs.pop("num_codebooks", None)
            num_codebooks = 1
            codebook_dim = kwargs.pop("codebook_dim", 12)
            codebook_size = kwargs.pop("codebook_size", 4096)
            logger.info(f"  LFQ: dim={codebook_dim}, size={codebook_size}")
            quantizer_kwargs["num_codebooks"] = num_codebooks
            quantizer_kwargs["codebook_dim"] = codebook_dim
            quantizer_kwargs["codebook_size"] = codebook_size

        model = DiscreteTokenizer(
            dim=2,
            in_channels=in_channels,
            out_channels=in_channels,
            z_channels=z_channels,
            embedding_dim=embedding_dim,
            channels=channels,
            channels_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            resolution=resolution,
            spatial_compression=spatial_compression,
            quantizer=quantizer,
            name=f"{dataset_name}_{quantizer}_tokenizer",
            **quantizer_kwargs,
            **kwargs,
        )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {num_params:,}")

        effective_vocab_size = model.get_codebook_size()
        logger.info(f"  Effective vocabulary size: {effective_vocab_size:,}")

        latent_shape = model.get_latent_shape((1, in_channels, resolution, resolution))
        logger.info(f"  Latent shape: {latent_shape}")

        return model

    else:
        logger.info(f"  Formulation: {formulation}")
        logger.info(f"  Latent channels: {latent_channels}")

        model = ContinuousTokenizer(
            dim=2,
            in_channels=in_channels,
            out_channels=in_channels,
            z_channels=z_channels,
            channels=channels,
            channels_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            resolution=resolution,
            spatial_compression=spatial_compression,
            latent_channels=latent_channels,
            formulation=formulation,
            name=f"{dataset_name}_{formulation}_tokenizer",
            **kwargs,
        )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {num_params:,}")

        latent_shape = model.get_latent_shape((1, in_channels, resolution, resolution))
        logger.info(f"  Latent shape: {latent_shape}")
        logger.info(f"  Latent channels: {model.latent_channels}")

        return model


def build_loss_fn(args, in_channels: int, dim: int, steps_per_epoch: int):
    """Build the appropriate loss function based on tokenizer type and args.

    Returns:
        Tuple of (loss_fn, disc_optimizer_or_None)
    """
    disc_optimizer = None
    disc_start_iter = args.disc_start_epoch * steps_per_epoch

    if args.tokenizer == "continuous" and args.formulation == "VAE":
        # VAE: recon + annealed KL + optional LPIPS + optional GAN
        loss_fn = VAEGANLoss(
            dim=dim,
            input_channels=in_channels,
            reconstruction_weight=args.recon_weight,
            kl_weight=args.kl_weight,
            kl_warmup_epochs=args.kl_warmup_epochs,
            perceptual_weight=args.lpips_weight,
            adversarial_weight=args.disc_weight,
            discriminator_start_iter=disc_start_iter,
            lpips_slice_stride=1,
            reconstruction_type=args.recon_type,
            use_lecam=True,
            lecam_weight=args.lecam_weight,
        )
        logger.info(
            f"  Loss: VAEGANLoss (recon_type={args.recon_type}, kl_weight={args.kl_weight}, "
            f"kl_warmup={args.kl_warmup_epochs} epochs, "
            f"lpips={args.lpips_weight}, disc={args.disc_weight})"
        )

    elif args.tokenizer == "discrete":
        # Discrete: recon + quant_loss + optional LPIPS + optional GAN
        loss_fn = VQGANLoss(
            dim=dim,
            input_channels=in_channels,
            reconstruction_weight=args.recon_weight,
            quantization_weight=args.quant_weight,
            perceptual_weight=args.lpips_weight,
            adversarial_weight=args.disc_weight,
            discriminator_start_iter=disc_start_iter,
            lpips_slice_stride=1,
            reconstruction_type=args.recon_type,
            use_lecam=True,
            lecam_weight=args.lecam_weight,
        )
        logger.info(
            f"  Loss: VQGANLoss (recon_type={args.recon_type}, quant_weight={args.quant_weight}, "
            f"lpips={args.lpips_weight}, disc={args.disc_weight})"
        )

    else:
        # AE: use VAEGANLoss with kl_weight=0 if LPIPS or disc requested,
        # otherwise plain Combined. AE model returns no kl_loss so the KL
        # term is naturally skipped inside VAEGANLoss.generator_step.
        if args.lpips_weight > 0 or args.disc_weight > 0:
            loss_fn = VAEGANLoss(
                dim=dim,
                input_channels=in_channels,
                reconstruction_weight=args.recon_weight,
                kl_weight=0.0,
                perceptual_weight=args.lpips_weight,
                adversarial_weight=args.disc_weight,
                discriminator_start_iter=disc_start_iter,
                lpips_slice_stride=1,
                reconstruction_type=args.recon_type,
                use_lecam=True,
                lecam_weight=args.lecam_weight,
            )
            logger.info(
                f"  Loss: VAEGANLoss/AE-disc (recon_type={args.recon_type}, kl_weight=0, "
                f"lpips={args.lpips_weight}, disc={args.disc_weight})"
            )
        else:
            loss_fn = Combined(
                reconstruction_weight=args.recon_weight,
                perceptual_weight=0.0,
                quantization_weight=0.0,
                reconstruction_type=args.recon_type,
            )
            logger.info(f"  Loss: Combined (recon_weight={args.recon_weight})")

    # Create discriminator optimizer if GAN is enabled
    if (
        args.disc_weight > 0
        and hasattr(loss_fn, "discriminator")
        and loss_fn.discriminator is not None
    ):
        disc_optimizer = torch.optim.AdamW(
            loss_fn.discriminator.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.5, 0.9),
        )
        logger.info(f"  Discriminator optimizer: AdamW(lr={args.lr}, betas=(0.5, 0.9))")

    return loss_fn, disc_optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a 2D tokenizer on MedMNIST datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=list(DATASETS_2D.keys()),
        help="MedMNIST dataset to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store/download data",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Download dataset if not present",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        choices=[28, 64, 128, 224],
        help="Dataset image size (64 recommended for MedMNIST+)",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        default=8,
        help="Spatial compression factor (8 for 64/128px, 4 for 28px)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Tokenizer type: discrete (VQ/FSQ) or continuous (VAE)",
    )
    parser.add_argument(
        "--quantizer",
        type=str,
        default="RESFSQ",
        choices=["VQ", "FSQ", "LFQ", "RESFSQ"],
        help="Quantization method for discrete tokenizer",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        default="VAE",
        choices=["VAE", "AE"],
        help="Formulation for continuous tokenizer",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=4,
        help="Number of latent channels for continuous tokenizer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=16,
        help="Dimension of quantized embeddings (default: 16)",
    )
    parser.add_argument(
        "--z_channels",
        type=int,
        default=64,
        help="Encoder output channels before quant_conv (default: 64)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Base channel count for encoder/decoder (default: 64)",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=1024,
        help="Number of codebook entries for VQ (default: 1024)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="FSQ/RESFSQ levels (e.g., --levels 8 5 5 5)",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=2,
        help="Number of codebooks for RESFSQ/LFQ",
    )
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=None,
        help="Codebook dimension for LFQ",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=None,
        help="Codebook size for LFQ",
    )
    parser.add_argument(
        "--commitment_loss_weight",
        type=float,
        default=None,
        help="Commitment loss weight for LFQ/VQ (default: quantizer default)",
    )
    parser.add_argument(
        "--entropy_loss_weight",
        type=float,
        default=None,
        help="Entropy loss weight for LFQ (default: quantizer default)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=5,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--recon_type",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Reconstruction loss type",
    )
    parser.add_argument(
        "--recon_weight",
        type=float,
        default=4.0,
        help="Reconstruction loss weight (SOTA default: 4.0)",
    )
    parser.add_argument(
        "--quant_weight",
        type=float,
        default=1.0,
        help="Quantization loss weight (discrete tokenizers only)",
    )
    # VAE-specific loss args
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-6,
        help="KL divergence weight for VAE (default: 1e-6, like LDM/Stable Diffusion)",
    )
    parser.add_argument(
        "--kl_warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to linearly anneal KL weight from 0 to target (default: 10)",
    )
    # Optional perceptual loss
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=0.0,
        help="LPIPS perceptual loss weight (0=disabled, typical: 0.1-1.0)",
    )
    # Optional discriminator
    parser.add_argument(
        "--disc_weight",
        type=float,
        default=0.0,
        help="Discriminator adversarial loss weight (0=disabled, typical: 0.1)",
    )
    parser.add_argument(
        "--disc_start_epoch",
        type=int,
        default=5,
        help="Epoch to start discriminator training (default: 5)",
    )
    parser.add_argument(
        "--lecam_weight",
        type=float,
        default=0.001,
        help="LeCAM regularization weight for GAN stability (default: 0.001)",
    )
    parser.add_argument(
        "--model_ema_decay",
        type=float,
        default=0.9999,
        help="Model EMA decay rate (default: 0.9999)",
    )
    parser.add_argument(
        "--use_norm",
        action="store_true",
        default=False,
        help="Use L2 normalization in VQ codebook (prevents collapse)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="medmnist-tokenizers",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or username)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from dataset and tokenizer)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Use EMA codebook updates for VQ quantizer (more stable on small datasets)",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=10,
        help="Save periodic checkpoint every N epochs regardless of val improvement",
    )

    args = parser.parse_args()

    # Create Accelerator for mixed precision, device management, and W&B logging
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision="bf16" if args.mixed_precision else "no",
    )
    device = accelerator.device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    logger.info("=" * 60)
    logger.info("MedMNIST 2D Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    if args.tokenizer == "discrete":
        logger.info(f"Quantizer: {args.quantizer}")
    else:
        logger.info(f"Formulation: {args.formulation}")
        logger.info(f"Latent channels: {args.latent_channels}")
        if args.formulation == "VAE":
            logger.info(f"KL weight: {args.kl_weight}")
            logger.info(f"KL warmup epochs: {args.kl_warmup_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_info = DATASETS_2D[args.dataset]
    logger.info(f"Dataset info: {dataset_info['info']['description'][:200]}...")
    logger.info(f"Channels: {dataset_info['channels']}")
    logger.info(f"Train samples: {dataset_info['n_samples']['train']}")
    logger.info(f"Val samples: {dataset_info['n_samples']['val']}")
    logger.info(f"Test samples: {dataset_info['n_samples']['test']}")

    logger.info("\n[1/5] Creating dataloaders...")
    train_loader, val_loader = create_medmnist2d_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        size=args.size,
        batch_size=args.batch_size,
    )
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    logger.info("\n[2/5] Creating model...")
    model_kwargs = {}
    if args.levels is not None:
        model_kwargs["levels"] = args.levels
    if args.num_codebooks is not None:
        model_kwargs["num_codebooks"] = args.num_codebooks
    if args.codebook_dim is not None:
        model_kwargs["codebook_dim"] = args.codebook_dim
    if args.codebook_size is not None:
        model_kwargs["codebook_size"] = args.codebook_size
    if args.commitment_loss_weight is not None:
        model_kwargs["commitment_loss_weight"] = args.commitment_loss_weight
    if args.entropy_loss_weight is not None:
        model_kwargs["entropy_loss_weight"] = args.entropy_loss_weight
    if args.use_ema:
        model_kwargs["use_ema"] = True
    if args.use_norm:
        model_kwargs["use_norm"] = True
    model_kwargs["num_embeddings"] = args.num_embeddings

    in_channels = dataset_info["channels"]
    model = create_2d_tokenizer(
        dataset_name=args.dataset,
        in_channels=in_channels,
        resolution=args.size,
        spatial_compression=args.spatial_compression,
        tokenizer_type=args.tokenizer,
        quantizer=args.quantizer if args.tokenizer == "discrete" else None,
        formulation=args.formulation if args.tokenizer == "continuous" else None,
        latent_channels=args.latent_channels
        if args.tokenizer == "continuous"
        else None,
        z_channels=args.z_channels,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        **model_kwargs,
    )

    logger.info("\n[3/5] Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )

    logger.info("\n[4/5] Creating loss function and callbacks...")
    loss_fn, disc_optimizer = build_loss_fn(
        args, in_channels=in_channels, dim=2, steps_per_epoch=len(train_loader)
    )

    if args.tokenizer == "discrete":
        run_name = args.name or f"{args.dataset}_{args.quantizer}"
    else:
        run_name = args.name or f"{args.dataset}_{args.formulation}"

    accelerator.init_trackers(
        args.wandb_project,
        config=vars(args),
        init_kwargs={
            "wandb": {
                "entity": args.wandb_entity,
                "name": run_name,
                "settings": wandb.Settings(start_method="fork"),
            }
        },
    )

    if args.tokenizer == "discrete":
        checkpoint_path = os.path.join(
            args.save_dir, f"{args.dataset}_{args.quantizer}"
        )
    else:
        checkpoint_path = os.path.join(
            args.save_dir, f"{args.dataset}_{args.formulation}"
        )
    callbacks = [
        Logger(),
        Checkpoint(
            filepath=checkpoint_path,
            monitor="val_total_loss"
            if isinstance(loss_fn, (VQGANLoss, VAEGANLoss))
            else "val_total",
            save_best_only=True,
            mode="min",
            save_every_n_epochs=args.save_every_n_epochs,
        ),
        ReconstructionLogger(num_samples=4, every_n_epochs=1),
    ]

    logger.info("\n[5/5] Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        loss_fn=loss_fn,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        callbacks=callbacks,
        channels_last=True,
        scheduler=scheduler,
        disc_optimizer=disc_optimizer,
        model_ema_decay=args.model_ema_decay,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    trainer.fit(
        train_loader,
        args.epochs,
        val_loader=val_loader,
        val_interval=args.val_freq,
        resume=args.resume,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    if args.tokenizer == "discrete":
        final_ckpt_dir = os.path.join(
            args.save_dir, f"{args.dataset}_{args.quantizer}_final"
        )
    else:
        final_ckpt_dir = os.path.join(
            args.save_dir, f"{args.dataset}_{args.formulation}_final"
        )
    trainer.save_checkpoint(final_ckpt_dir)
    logger.info(f"Final checkpoint: {final_ckpt_dir}")

    trainer.end_training()

    logger.info("\n" + "=" * 60)
    logger.info("Running Inference")
    logger.info("=" * 60)

    model = trainer.model
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = (
            sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        )
        sample_batch = sample_batch.to(accelerator.device)
        logger.info(f"Input shape: {sample_batch.shape}")

        if args.tokenizer == "discrete":
            indices, codes, q_loss = model.encode(sample_batch)  # type: ignore
            logger.info(f"Encoded indices shape: {indices.shape}")
            logger.info(f"Encoded codes shape: {codes.shape}")
            logger.info(f"Quantization loss: {q_loss.mean().item():.6f}")

            reconstruction = model.decode(codes)
            logger.info(f"Reconstruction shape: {reconstruction.shape}")

            recon_from_indices = model.detokenize(indices)
            logger.info(
                f"Reconstruction from indices shape: {recon_from_indices.shape}"
            )

            effective_vocab = model.get_codebook_size()
            unique_indices = indices.unique().numel()
            logger.info(f"Effective vocabulary size: {effective_vocab}")
            logger.info(f"Unique indices used: {unique_indices} / {effective_vocab}")
        else:
            latents, dist_output = model.encode(sample_batch)  # type: ignore
            logger.info(f"Encoded latents shape: {latents.shape}")
            kl_loss = dist_output[0]
            logger.info(f"KL divergence: {kl_loss.mean().item():.6f}")
            logger.info("Latent type: continuous (float32)")
            logger.info(f"Latent channels: {model.latent_channels}")

            reconstruction = model.decode(latents)
            logger.info(f"Reconstruction shape: {reconstruction.shape}")

        if args.recon_type == "l2":
            recon_error = F.mse_loss(sample_batch, reconstruction).item()
        else:
            recon_error = F.l1_loss(sample_batch, reconstruction).item()
        logger.info(f"Mean reconstruction error ({args.recon_type}): {recon_error:.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
