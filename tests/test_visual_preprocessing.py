"""Visual tests for preprocessing and augmentations.

This script generates visualizations of the preprocessing pipeline
and augmentations applied to the bundled simulated brain volume.

Run with: python -m pytest tests/test_visual_preprocessing.py -v -s
Or directly: python tests/test_visual_preprocessing.py

Generates output images in tests/visual_outputs/ for inspection.
"""

import shutil
import tempfile
from pathlib import Path

import medrs
import numpy as np
import pytest
import torch

from medtokenizers import example_volume_path

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "visual_outputs"


def _nifti_suffix(path: Path) -> str:
    """Return ``.nii.gz`` or ``.nii`` so copies stay readable by medrs."""
    return ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"


def _percentile_normalize(
    data: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Normalize data using percentile scaling."""
    p_low = np.percentile(data, lower_percentile)
    p_high = np.percentile(data, upper_percentile)
    normalized = (data - p_low) / (p_high - p_low + 1e-8)
    normalized = normalized * (target_max - target_min) + target_min
    return np.clip(normalized, target_min, target_max).astype(np.float32)


def save_slices_as_png(
    tensor: torch.Tensor,
    output_path: str,
    title: str = "",
    show_stats: bool = True,
):
    """Save middle slices of a 3D volume as a PNG image.

    Args:
        tensor: 3D or 5D tensor (if 5D, uses first sample)
        output_path: Path to save the image
        title: Title for the figure
        show_stats: Whether to show intensity statistics
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed - skipping visual tests")
        return

    # Handle different tensor shapes
    if tensor.dim() == 5:
        tensor = tensor[0, 0]  # Remove batch and channel dims
    elif tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu().contiguous()
    h, w, d = tensor.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Middle slices along each axis
    # NIfTI data is (X, Y, Z) - extract slices and transpose for proper display
    slices = [
        ("Axial (Z)", tensor[:, :, d // 2].numpy().T, "gray"),
        ("Coronal (Y)", tensor[:, w // 2, :].numpy().T, "gray"),
        ("Sagittal (X)", tensor[h // 2, :, :].numpy().T, "gray"),
    ]

    for ax, (slice_name, slice_data, cmap) in zip(
        axes[0],
        slices,
    ):
        im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{slice_name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Intensity histograms
    flat = tensor.flatten().numpy()
    axes[1, 0].hist(flat, bins=100, color="blue", alpha=0.7)
    axes[1, 0].set_title("Intensity Histogram (Full)")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Count")

    # Focus on [0, 1] range
    mask = (flat >= 0) & (flat <= 1)
    axes[1, 1].hist(flat[mask], bins=100, color="green", alpha=0.7)
    axes[1, 1].set_title("Intensity Histogram [0, 1] range")
    axes[1, 1].set_xlabel("Intensity")

    # Stats text
    stats_text = f"""Statistics:
Shape: {tuple(tensor.shape)}
Min: {tensor.min().item():.4f}
Max: {tensor.max().item():.4f}
Mean: {tensor.mean().item():.4f}
Std: {tensor.std().item():.4f}
% in [0,1]: {100 * mask.sum() / len(flat):.1f}%
"""
    axes[1, 2].text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_augmentation_grid(
    original: torch.Tensor,
    augmented_list: list[tuple[str, torch.Tensor]],
    output_path: str,
):
    """Save a grid showing original and multiple augmentations.

    Args:
        original: Original tensor
        augmented_list: List of (name, tensor) tuples
        output_path: Path to save the image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed - skipping visual tests")
        return

    n_augs = len(augmented_list) + 1
    cols = min(4, n_augs)
    rows = (n_augs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    # Flatten axes for easier indexing
    axes_flat = [ax for row in axes for ax in row]

    # Get middle slice from original
    if original.dim() == 5:
        orig_slice = original[0, 0, :, :, original.shape[-1] // 2]
    elif original.dim() == 4:
        orig_slice = original[0, :, :, original.shape[-1] // 2]
    else:
        orig_slice = original[:, :, original.shape[-1] // 2]

    axes_flat[0].imshow(orig_slice.T.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes_flat[0].set_title("Original")
    axes_flat[0].axis("off")

    for i, (name, aug_tensor) in enumerate(augmented_list):
        if aug_tensor.dim() == 5:
            aug_slice = aug_tensor[0, 0, :, :, aug_tensor.shape[-1] // 2]
        elif aug_tensor.dim() == 4:
            aug_slice = aug_tensor[0, :, :, aug_tensor.shape[-1] // 2]
        else:
            aug_slice = aug_tensor[:, :, aug_tensor.shape[-1] // 2]

        axes_flat[i + 1].imshow(aug_slice.T.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_flat[i + 1].set_title(name)
        axes_flat[i + 1].axis("off")

    # Hide unused axes
    for i in range(n_augs, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


class TestVisualPreprocessing:
    """Visual tests for preprocessing pipeline."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(exist_ok=True)

    @pytest.fixture
    def brain_image_path(self):
        """Get path to the bundled simulated brain volume."""
        return str(example_volume_path())

    def test_visualize_raw_loading(self, brain_image_path):
        """Visualize the raw loaded brain image."""
        img = medrs.load(brain_image_path)
        data = img.to_numpy()  # Use numpy, not torch

        # Normalize just for visualization
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        tensor = torch.from_numpy(data)

        save_slices_as_png(
            tensor,
            str(OUTPUT_DIR / "01_raw_brain.png"),
            title="Raw Brain Image (min-max normalized for visualization)",
        )
        print(f"Saved: {OUTPUT_DIR / '01_raw_brain.png'}")

    def test_visualize_percentile_normalization(self, brain_image_path):
        """Visualize different percentile normalization settings."""
        img = medrs.load(brain_image_path)
        data = img.to_numpy()

        settings = [
            (0.0, 100.0, "0-100%"),
            (0.5, 99.5, "0.5-99.5%"),
            (1.0, 99.0, "1-99%"),
            (2.0, 98.0, "2-98%"),
        ]

        for lower, upper, name in settings:
            normalized = _percentile_normalize(data, lower, upper, 0.0, 1.0)
            tensor = torch.from_numpy(normalized)

            save_slices_as_png(
                tensor,
                str(
                    OUTPUT_DIR
                    / f"02_percentile_{name.replace('-', '_').replace('%', '')}.png"
                ),
                title=f"Percentile Normalization: {name}",
            )
        print(f"Saved percentile visualizations to {OUTPUT_DIR}")

    def test_visualize_resampling(self, brain_image_path):
        """Visualize the effect of resampling."""
        img = medrs.load(brain_image_path)
        original_shape = img.shape
        spacing = tuple(img.spacing) if hasattr(img, "spacing") else (1.0, 1.0, 1.0)

        # Load and normalize using numpy
        data = img.to_numpy()
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

        # Original
        orig_tensor = torch.from_numpy(normalized)
        save_slices_as_png(
            orig_tensor,
            str(OUTPUT_DIR / "03a_resampling_original.png"),
            title=f"Original: shape={original_shape}, spacing={spacing}",
        )

        # Resample to different sizes using medrs
        for target_size in [64, 128, 192]:
            resampled = img.resample_to_shape([target_size, target_size, target_size])
            resampled_data = resampled.to_numpy()
            resampled_norm = _percentile_normalize(resampled_data, 0.5, 99.5, 0.0, 1.0)
            tensor = torch.from_numpy(resampled_norm)

            save_slices_as_png(
                tensor,
                str(
                    OUTPUT_DIR
                    / f"03b_resampled_{target_size}x{target_size}x{target_size}.png"
                ),
                title=f"Resampled to {target_size}^3",
            )
        print(f"Saved resampling visualizations to {OUTPUT_DIR}")

    def test_visualize_cropping(self, brain_image_path):
        """Visualize different crop sizes and positions."""
        img = medrs.load(brain_image_path)
        data = img.to_numpy()
        shape = data.shape

        # Normalize first
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

        # Center crops of different sizes using numpy slicing
        crops = []
        for crop_size in [32, 64, 96]:
            center = [s // 2 for s in shape]
            start = [max(0, c - crop_size // 2) for c in center]
            end = [
                min(s, st + crop_size)
                for s, st in zip(
                    shape,
                    start,
                )
            ]

            cropped = normalized[
                start[0] : end[0], start[1] : end[1], start[2] : end[2]
            ]
            tensor = torch.from_numpy(cropped)
            crops.append((f"Center {crop_size}^3", tensor))

        # Save grid
        orig_tensor = torch.from_numpy(normalized)
        save_augmentation_grid(
            orig_tensor,
            crops,
            str(OUTPUT_DIR / "04_cropping_sizes.png"),
        )
        print(f"Saved: {OUTPUT_DIR / '04_cropping_sizes.png'}")

    def test_visualize_augmentations(self, brain_image_path):
        """Visualize different augmentations."""
        img = medrs.load(brain_image_path)
        data = img.to_numpy()

        # Load and normalize
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)
        tensor = torch.from_numpy(normalized)

        # Add channel dimension
        tensor = tensor.unsqueeze(0)

        augmentations = []

        # Flip augmentations
        for axis, name in [(1, "Flip X"), (2, "Flip Y"), (3, "Flip Z")]:
            flipped = torch.flip(tensor.clone(), dims=[axis])
            augmentations.append((name, flipped))

        # Intensity scaling
        for scale in [0.8, 1.2]:
            scaled = (tensor.clone() * scale).clamp(0, 1)
            augmentations.append((f"Scale {scale}", scaled))

        # Gamma correction
        for gamma in [0.7, 1.5]:
            gamma_corrected = torch.pow(tensor.clone().clamp(0.001, 1), gamma)
            augmentations.append((f"Gamma {gamma}", gamma_corrected))

        # 90 degree rotations
        for k in [1, 2, 3]:
            rotated = torch.rot90(tensor.clone(), k=k, dims=[1, 2])
            augmentations.append((f"Rot90 k={k}", rotated))

        save_augmentation_grid(
            tensor,
            augmentations,
            str(OUTPUT_DIR / "05_augmentations.png"),
        )
        print(f"Saved: {OUTPUT_DIR / '05_augmentations.png'}")

    def test_visualize_full_pipeline(self, brain_image_path):
        """Visualize the complete MAISI preprocessing pipeline."""
        from medtokenizers.preprocessing import preprocess_for_maisi

        # Run MAISI preprocessing
        tensor, metadata = preprocess_for_maisi(
            brain_image_path,
            target_spacing=(1.0, 1.0, 1.0),
            percentile_lower=0.5,
            percentile_upper=99.5,
            divisible_k=16,
        )

        save_slices_as_png(
            tensor,
            str(OUTPUT_DIR / "06_maisi_preprocessed.png"),
            title=f"MAISI Preprocessed\nOriginal: {metadata['original_shape']}, "
            f"Spacing: {metadata['original_spacing']}\n"
            f"Final: {tensor.shape[2:]}, Padding: {metadata['padding'][:2]}",
        )
        print(f"Saved: {OUTPUT_DIR / '06_maisi_preprocessed.png'}")

    def test_visualize_data_loader_output(self, brain_image_path):
        """Visualize actual data loader output with augmentations."""
        from scripts.data_loading import get_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy brain image to temp directory (need multiple files for split)
            brain_path = Path(brain_image_path)
            for i in range(3):  # Create 3 copies for train/val split
                dest_path = Path(tmpdir) / f"brain_{i}{_nifti_suffix(brain_path)}"
                shutil.copy(brain_path, dest_path)

            # Create loader with augmentation
            train_loader, _ = get_loaders(
                batch_size=1,
                data_dir=tmpdir,
                augment=True,
                reslice_prob=1.0,
                resize_threshold=128,
                num_workers=0,
            )

            # Get multiple samples (same image with different augmentations)
            samples = []
            for i, batch in enumerate(train_loader):
                if i >= 4:
                    break
                samples.append((f"Sample {i + 1}", batch["image"]))

            if samples:
                save_augmentation_grid(
                    samples[0][1],
                    samples[1:],
                    str(OUTPUT_DIR / "07_dataloader_augmented.png"),
                )
                print(f"Saved: {OUTPUT_DIR / '07_dataloader_augmented.png'}")


class TestVisualCropTraining:
    """Visual tests for MAISI-style crop training."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(exist_ok=True)

    @pytest.fixture
    def brain_image_path(self):
        """Get path to the bundled simulated brain volume."""
        return str(example_volume_path())

    def test_visualize_random_crops(self, brain_image_path):
        """Visualize random crops from the brain image."""
        from scripts.data_loading import get_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy brain image to temp directory (need multiple files for split)
            brain_path = Path(brain_image_path)
            for i in range(3):  # Create 3 copies for train/val split
                dest = Path(tmpdir) / f"brain_{i}{_nifti_suffix(brain_path)}"
                shutil.copy(brain_path, dest)

            train_loader, _ = get_loaders(
                batch_size=1,
                data_dir=tmpdir,
                crop_size=64,
                crops_per_volume=8,
                augment=True,
                num_workers=0,
            )

            crops = []
            for i, batch in enumerate(train_loader):
                if i >= 6:
                    break
                crops.append((f"Crop {i + 1}", batch["image"]))

            if crops:
                save_augmentation_grid(
                    crops[0][1],
                    crops[1:],
                    str(OUTPUT_DIR / "08_random_crops.png"),
                )
                print(f"Saved: {OUTPUT_DIR / '08_random_crops.png'}")


def run_all_visual_tests():
    """Run all visual tests and generate output images."""
    print("Running visual preprocessing tests...")
    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run pytest on this file
    import pytest

    result = pytest.main([__file__, "-v", "-s", "--tb=short"])
    return result


if __name__ == "__main__":
    run_all_visual_tests()
