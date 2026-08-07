#!/usr/bin/env python3
"""Visualize reconstruction quality for ChestMNIST discrete tokenizers.

Generates two figures:
1. Reconstruction grid: Original + 4 model reconstructions (RESFSQ, VQ, FSQ, LFQ)
2. Error map grid: |original - reconstruction| heatmaps with shared colorbar

Usage:
    uv run python scripts/visualize_reconstructions.py
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
from safetensors.torch import load_file
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from medtokenizers.networks import DiscreteTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_SAMPLES = 4
DATASET_NAME = "chestmnist"
SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "medmnist",
)

# Model order for rows (top to bottom, after "Original")
MODEL_ORDER = ["RESFSQ", "VQ", "FSQ", "LFQ"]

CHECKPOINT_DIRS = {
    name: os.path.join(CHECKPOINT_BASE, f"chestmnist_{name}_final")
    for name in MODEL_ORDER
}

OUTPUT_RECON = os.path.join(CHECKPOINT_BASE, "chestmnist_v2_reconstructions.png")
OUTPUT_ERROR = os.path.join(CHECKPOINT_BASE, "chestmnist_v2_errors.png")


# ---------------------------------------------------------------------------
# Common encoder/decoder config (matches the ChestMNIST training defaults)
# ---------------------------------------------------------------------------
COMMON_CONFIG = {
    "dim": 2,
    "in_channels": 1,
    "out_channels": 1,
    "z_channels": 64,
    "channels": 64,
    "channels_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.0,
    "resolution": 64,
    "spatial_compression": 8,
    "embedding_dim": 16,
}

QUANTIZER_CONFIGS = {
    "FSQ": {
        "quantizer": "FSQ",
        "levels": [8, 8, 8, 5, 5, 5],
    },
    "RESFSQ": {
        "quantizer": "RESFSQ",
        "levels": [8, 8, 8],
        "num_codebooks": 2,
    },
    "VQ": {
        "quantizer": "VQ",
        "num_embeddings": 8192,
        "use_ema": True,
        "use_norm": True,
    },
    "LFQ": {
        "quantizer": "LFQ",
        "codebook_dim": 12,
        "codebook_size": 4096,
        "num_codebooks": 1,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_test_data() -> torch.Tensor:
    """Load ChestMNIST test split, normalize to [0,1], return (N,1,H,W) tensor."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        DATASET_NAME,
    )
    os.makedirs(data_dir, exist_ok=True)

    # Try NPZ cache first
    npz_path = os.path.join(data_dir, f"{DATASET_NAME}_{SIZE}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        images = data["test_images"]
    else:
        mn_info = medmnist.INFO[DATASET_NAME]
        DataClass = getattr(medmnist, mn_info["python_class"])
        dataset = DataClass(root=data_dir, split="test", download=True, size=SIZE)
        if hasattr(dataset, "imgs"):
            images = dataset.imgs
        elif hasattr(dataset, "data"):
            images = dataset.data
        else:
            raise ValueError(f"Cannot access images for {DATASET_NAME}")

    # Squeeze trailing singleton channel dim if present
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    # Normalize to [0, 1]
    images = images.astype(np.float32)
    min_val = images.min()
    max_val = images.max()
    if max_val - min_val > 1e-8:
        images = (images - min_val) / (max_val - min_val)

    # (N, H, W) -> (N, 1, H, W)
    tensor = torch.from_numpy(images).unsqueeze(1)
    return tensor


def load_model(name: str) -> DiscreteTokenizer:
    """Load a discrete tokenizer model from its checkpoint."""
    ckpt_dir = CHECKPOINT_DIRS[name]
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found in {ckpt_dir}")

    cfg = {**COMMON_CONFIG, **QUANTIZER_CONFIGS[name]}
    cfg["name"] = f"chestmnist_{name}_tokenizer"

    model = DiscreteTokenizer(**cfg)
    state_dict = load_file(safetensors_path, device=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def reconstruct(model: DiscreteTokenizer, samples: torch.Tensor) -> torch.Tensor:
    """Run samples through model and return reconstructions."""
    with torch.inference_mode():
        output = model(samples.to(DEVICE))
        if isinstance(output, dict):
            recon = output["reconstructions"]
        else:
            recon = output.reconstructions
    return recon.cpu().clamp(0, 1)


def compute_metrics(
    originals: np.ndarray, reconstructions: np.ndarray
) -> tuple[float, float]:
    """Compute mean PSNR and SSIM over a batch of images.

    Args:
        originals: (N, H, W) array in [0, 1]
        reconstructions: (N, H, W) array in [0, 1]

    Returns:
        (mean_psnr, mean_ssim)
    """
    psnr_vals = []
    ssim_vals = []
    for orig, recon in zip(originals, reconstructions):
        psnr_vals.append(psnr(orig, recon, data_range=1.0))
        ssim_vals.append(ssim(orig, recon, data_range=1.0))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {DATASET_NAME} test data...")
    all_data = load_test_data()
    print(f"  Test set shape: {all_data.shape}")

    # Pick random samples
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(all_data), size=NUM_SAMPLES, replace=False)
    samples = all_data[indices]  # (4, 1, 64, 64)
    originals_np = samples[:, 0].numpy()  # (4, 64, 64)
    print(f"  Selected sample indices: {indices.tolist()}")

    # Run each model
    reconstructions = {}  # name -> (4, 64, 64) numpy
    metrics = {}  # name -> (psnr, ssim)

    for name in MODEL_ORDER:
        print(f"Loading {name} model...")
        model = load_model(name)
        recon = reconstruct(model, samples)  # (4, 1, 64, 64) tensor
        recon_np = recon[:, 0].numpy()  # (4, 64, 64)
        reconstructions[name] = recon_np
        m_psnr, m_ssim = compute_metrics(originals_np, recon_np)
        metrics[name] = (m_psnr, m_ssim)
        print(f"  {name}: PSNR={m_psnr:.2f} dB, SSIM={m_ssim:.4f}")
        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Figure 1: Reconstruction grid
    # -----------------------------------------------------------------------
    print("\nGenerating reconstruction grid...")
    nrows = 1 + len(MODEL_ORDER)  # Original + 4 models
    ncols = NUM_SAMPLES

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0), squeeze=False
    )

    row_labels = ["Original"] + [
        f"{name}\nPSNR={metrics[name][0]:.1f} / SSIM={metrics[name][1]:.3f}"
        for name in MODEL_ORDER
    ]

    for col in range(ncols):
        # Original row
        axes[0, col].imshow(originals_np[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Model rows
        for row_idx, name in enumerate(MODEL_ORDER, start=1):
            axes[row_idx, col].imshow(
                reconstructions[name][col], cmap="gray", vmin=0, vmax=1
            )
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])

    # Row labels on the left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(
            label, fontsize=9, rotation=0, labelpad=80, va="center"
        )

    fig.suptitle("ChestMNIST Discrete Tokenizer Reconstructions", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.12, 0.0, 1.0, 0.95])
    fig.savefig(OUTPUT_RECON, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_RECON}")

    # -----------------------------------------------------------------------
    # Figure 2: Error map grid
    # -----------------------------------------------------------------------
    print("Generating error map grid...")

    # Compute all errors to find global max for shared colorbar
    errors = {}
    global_max = 0.0
    for name in MODEL_ORDER:
        err = np.abs(originals_np - reconstructions[name])
        errors[name] = err
        global_max = max(global_max, err.max())

    fig = plt.figure(figsize=(ncols * 2.0 + 0.8, len(MODEL_ORDER) * 2.0))
    gs = gridspec.GridSpec(
        len(MODEL_ORDER),
        ncols + 1,
        width_ratios=[1] * ncols + [0.05],
        wspace=0.05,
        hspace=0.15,
    )

    im = None
    for row_idx, name in enumerate(MODEL_ORDER):
        for col in range(ncols):
            ax = fig.add_subplot(gs[row_idx, col])
            im = ax.imshow(
                errors[name][col],
                cmap="hot",
                vmin=0,
                vmax=global_max,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                label = (
                    f"{name}\nPSNR={metrics[name][0]:.1f}\nSSIM={metrics[name][1]:.3f}"
                )
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=75, va="center")

    # Shared colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cbar_ax, label="|Original - Reconstruction|")

    fig.suptitle("ChestMNIST Reconstruction Error Maps", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.12, 0.0, 1.0, 0.95])
    fig.savefig(OUTPUT_ERROR, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_ERROR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
