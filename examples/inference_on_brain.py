"""Brain MRI inference with sliding window processing.

Demonstrates:
- Loading medical images using medrs (NIfTI format)
- Sliding window inference for large volumes
- Reconstruction quality metrics
- Saving results

Works with any tokenizer (VAE, VQ-VAE, etc.) and any checkpoint.
"""

import argparse
from pathlib import Path

import medrs
import numpy as np
import torch

from medtokenizers import (
    ContinuousTokenizer,
    DiscreteTokenizer,
    MAISITokenizer,
    example_volume_path,
)
from medtokenizers.preprocessing import save_nifti


def load_nifti_volume(nifti_path: str) -> tuple[torch.Tensor, dict]:
    """Load and preprocess NIfTI volume using medrs.

    Args:
        nifti_path: Path to .nii or .nii.gz file

    Returns:
        volume: Normalized tensor (1, 1, H, W, D)
        metadata: Dictionary with original image info for saving
    """
    print(f"Loading {nifti_path}...")
    img = medrs.load(nifti_path)
    tensor = img.to_torch_with_dtype_and_device(dtype=torch.float32)

    # Get metadata for later saving
    metadata = {
        "spacing": tuple(img.spacing) if hasattr(img, "spacing") else (1.0, 1.0, 1.0),
        "original_shape": tuple(tensor.shape),
        "source_path": nifti_path,
    }

    print(f"  Shape: {tensor.shape}")
    print(f"  Range: [{tensor.min():.2f}, {tensor.max():.2f}]")
    print(f"  Spacing: {metadata['spacing']}")

    # Normalize to [0, 1]
    volume = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    # Convert to tensor: (H, W, D) -> (1, 1, H, W, D)
    volume = volume.unsqueeze(0).unsqueeze(0)

    return volume, metadata


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction quality metrics.

    Args:
        original: Original volume
        reconstructed: Reconstructed volume

    Returns:
        Dictionary of metrics (MSE, PSNR, MAE)
    """
    mse = ((original - reconstructed) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    mae = (original - reconstructed).abs().mean().item()

    return {"mse": mse, "psnr": psnr, "mae": mae}


def save_nifti_volume(volume: torch.Tensor, metadata: dict, output_path: str):
    """Save volume as a NIfTI file.

    Args:
        volume: Tensor (1, 1, H, W, D)
        metadata: Dictionary with spacing info
        output_path: Output path
    """
    volume_np = volume.squeeze().cpu().numpy()
    spacing = metadata.get("spacing", (1.0, 1.0, 1.0))
    save_nifti(volume_np, output_path, spacing=spacing)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Brain MRI inference with sliding windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(example_volume_path()),
        help="Path to input NIfTI file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (None = random init for testing)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="maisi",
        choices=["maisi", "continuous", "discrete"],
        help="Type of tokenizer to use",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[96, 96, 64],
        metavar=("H", "W", "D"),
        help="ROI size for sliding window",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between windows (0.0-0.99)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu, None=auto)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--save-original", action="store_true", help="Also save the normalized original"
    )

    args = parser.parse_args()

    # Setup
    print("\n" + "=" * 70)
    print("Brain MRI Inference with Sliding Windows")
    print("=" * 70 + "\n")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    print("Loading model...")
    if args.model is not None:
        print(f"  Loading from: {args.model}")
        if args.model_type == "maisi":
            model = MAISITokenizer.from_pretrained(args.model)
        elif args.model_type == "continuous":
            model = ContinuousTokenizer.from_pretrained(args.model)
        else:
            model = DiscreteTokenizer.from_pretrained(args.model)
    else:
        print("  Creating new model (random initialization)")
        if args.model_type == "maisi":
            model = MAISITokenizer()
        elif args.model_type == "continuous":
            model = ContinuousTokenizer()
        else:
            model = DiscreteTokenizer()

    model = model.to(device)
    model.eval()
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Spatial compression: {model.spatial_compression}x\n")

    # Load volume using medrs
    print("Loading volume...")
    volume, metadata = load_nifti_volume(args.input)
    volume = volume.to(device)
    print()

    # Run inference
    print("Running sliding window inference...")
    print(f"  ROI size: {tuple(args.roi_size)}")
    print(f"  Overlap: {args.overlap}")

    with torch.no_grad():
        reconstructed = model.reconstruct(
            volume, roi_size=tuple(args.roi_size), overlap=args.overlap
        )

    print(f"  Output shape: {reconstructed.shape}\n")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(volume, reconstructed)
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  MAE:  {metrics['mae']:.6f}\n")

    # Save results
    print("Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save reconstruction
    input_name = Path(args.input).stem
    recon_path = output_dir / f"{input_name}_reconstructed.nii"
    save_nifti_volume(reconstructed, metadata, str(recon_path))

    # Optionally save original
    if args.save_original:
        orig_path = output_dir / f"{input_name}_normalized.nii"
        save_nifti_volume(volume, metadata, str(orig_path))

    # Save metrics
    metrics_path = output_dir / f"{input_name}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Reconstruction Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Model: {args.model or 'random init'}\n")
        f.write(f"ROI size: {tuple(args.roi_size)}\n")
        f.write(f"Overlap: {args.overlap}\n")
        f.write("\n")
        f.write(f"MSE:  {metrics['mse']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"MAE:  {metrics['mae']:.6f}\n")
    print(f"  Saved metrics to: {metrics_path}")

    print("\n" + "=" * 70)
    print("Inference complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
