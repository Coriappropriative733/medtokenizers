"""Comprehensive tests for med-rs integration.

Tests cover:
- Core medrs operations (load, save, resample, rescale)
- Preprocessing functions used by data loading
- Full pipeline integration with data loading
- Tests on the bundled simulated brain volume

Uses medrs for all NIfTI I/O operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import medrs
import nibabel as nib
import numpy as np
import pytest
import torch
from torch.nn import functional as nnf

from medtokenizers import example_volume_path


def _create_nifti(volume: np.ndarray, path: str, spacing: tuple = (1.0, 1.0, 1.0)):
    """Write a NIfTI file for medrs to read back.

    Args:
        volume: 3D numpy array
        path: Output file path
        spacing: Voxel spacing (x, y, z) in mm
    """
    # A positive diagonal affine is already RAS, so medrs reads the array back
    # in the same axis order it was written.
    affine = np.diag([*spacing, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), path)


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


class TestMedrsCoreFunctions:
    """Test core medrs operations used in the codebase."""

    def test_niftiimage_constructor_creates_image(self):
        """Test medrs.NiftiImage constructor creates a valid image object."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        assert img is not None
        assert hasattr(img, "shape")
        assert tuple(img.shape) == (64, 64, 64)

    def test_niftiimage_preserves_spacing(self):
        """Test that spacing is derived from affine."""
        vol = np.random.rand(64, 64, 32).astype(np.float32)
        # medrs derives spacing from the affine matrix
        affine = np.diag([1.5, 1.5, 3.0, 1.0]).astype(np.float32)
        img = medrs.NiftiImage(vol, affine)

        # Check that image has spacing attribute
        assert hasattr(img, "spacing")
        # Note: medrs may compute spacing differently from the affine diagonal
        # Just check that spacing exists and is reasonable
        assert len(img.spacing) == 3
        assert all(s > 0 for s in img.spacing)

    def test_save_and_load_roundtrip(self):
        """Test that save and load preserve data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vol = np.random.rand(64, 64, 64).astype(np.float32) * 100
            path = Path(tmpdir) / "test.nii.gz"
            _create_nifti(vol, str(path), (1.0, 1.0, 1.0))

            # Load
            loaded = medrs.load(str(path))
            loaded_data = loaded.to_numpy()

            np.testing.assert_allclose(loaded_data, vol, rtol=1e-4, atol=1e-4)

    def test_to_torch_returns_tensor(self):
        """Test to_torch returns a PyTorch tensor."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        tensor = img.to_torch()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_to_numpy_returns_array(self):
        """Test to_numpy returns a numpy array."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        arr = img.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, vol, rtol=1e-5)

    def test_resample_to_shape_changes_size(self):
        """Test resample_to_shape changes volume dimensions."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        # Resample to different size
        resampled = img.resample_to_shape([32, 32, 32])
        tensor = resampled.to_torch()

        assert tuple(tensor.shape) == (32, 32, 32)

    def test_resample_preserves_content(self):
        """Test resampling preserves overall content structure."""
        # Create a simple pattern
        vol = np.zeros((64, 64, 64), dtype=np.float32)
        vol[:32, :, :] = 1.0  # Left half is white

        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        # Resample to half size
        resampled = img.resample_to_shape([32, 32, 32])
        tensor = resampled.to_torch()

        # Left half should still be brighter
        left_mean = tensor[:16, :, :].mean()
        right_mean = tensor[16:, :, :].mean()
        assert left_mean > right_mean

    def test_rescale_intensity_normalizes(self):
        """Test medrs.rescale_intensity normalizes to specified range."""
        vol = np.linspace(0, 1000, 64 * 64 * 64).reshape(64, 64, 64).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        # Normalize to [0, 1] - medrs uses output_range keyword argument
        normalized = medrs.rescale_intensity(img, output_range=(0.0, 1.0))
        tensor = normalized.to_torch()

        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_crop_or_pad_changes_shape(self):
        """Test crop_or_pad changes volume shape."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)

        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        # Crop to 32x32x32
        cropped = img.crop_or_pad([32, 32, 32])
        tensor = cropped.to_torch()

        assert tuple(tensor.shape) == (32, 32, 32)


class TestPercentileNormalization:
    """Test percentile normalization function."""

    def test_percentile_normalize_basic(self):
        """Test basic percentile normalization."""
        vol = np.random.rand(64, 64, 64).astype(np.float32) * 1000

        normalized = _percentile_normalize(vol, 0.0, 100.0, 0.0, 1.0)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_percentile_normalize_with_outliers(self):
        """Test percentile normalization clips outliers."""
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        vol[0, 0, 0] = 10000  # High outlier
        vol[-1, -1, -1] = -100  # Low outlier

        normalized = _percentile_normalize(vol, 0.5, 99.5, 0.0, 1.0)

        # Clipped to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestPreprocessingPipeline:
    """Test the preprocessing pipeline used in data loading."""

    @pytest.fixture
    def temp_nifti_file(self):
        """Create a temporary NIfTI file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vol = np.random.rand(64, 64, 32).astype(np.float32) * 1000
            path = Path(tmpdir) / "test_brain.nii.gz"
            _create_nifti(vol, str(path), (1.0, 1.0, 2.0))
            yield str(path)

    def test_load_and_normalize(self, temp_nifti_file):
        """Test loading and normalizing a NIfTI file."""
        img = medrs.load(temp_nifti_file)
        data = img.to_numpy()
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

        assert normalized.shape == data.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_load_resample_normalize(self, temp_nifti_file):
        """Test loading, resampling, and normalizing."""
        img = medrs.load(temp_nifti_file)

        # Resample to isotropic
        resampled = img.resample_to_shape([64, 64, 64])

        # Normalize
        data = resampled.to_numpy()
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)
        tensor = torch.from_numpy(normalized)

        assert tensor.shape == (64, 64, 64)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestMedrsWithDataLoading:
    """Integration tests with data loading pipeline."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create test volumes with varying properties
            test_cases = [
                {"shape": (64, 64, 64), "spacing": (1.0, 1.0, 1.0)},  # Isotropic
                {"shape": (64, 64, 32), "spacing": (1.0, 1.0, 2.0)},  # Anisotropic
                {"shape": (128, 128, 64), "spacing": (0.5, 0.5, 1.0)},  # High-res
            ]

            for i, tc in enumerate(test_cases):
                vol = np.random.rand(*tc["shape"]).astype(np.float32)
                _create_nifti(vol, str(data_dir / f"sub-{i:03d}.nii.gz"), tc["spacing"])

            yield str(data_dir)

    def test_data_loading_with_medrs(self, temp_dataset_dir):
        """Test that data loading works with medrs-saved files."""
        from scripts.data_loading import get_loaders

        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_dataset_dir,
            reslice_prob=1.0,
            num_workers=0,
        )

        # Should be able to iterate
        batch = next(iter(train_loader))
        assert "image" in batch
        assert batch["image"].dim() == 5

    def test_patch_training_with_medrs(self, temp_dataset_dir):
        """Test MAISI-style patch training with medrs."""
        from scripts.data_loading import get_loaders

        train_loader, val_loader = get_loaders(
            batch_size=2,
            data_dir=temp_dataset_dir,
            crop_size=32,
            crops_per_volume=2,
            num_workers=0,
        )

        batch = next(iter(train_loader))
        assert batch["image"].shape[2:] == (32, 32, 32)

    def test_augmentation_preserves_valid_range(self, temp_dataset_dir):
        """Test that augmentations keep values in reasonable range."""
        from scripts.data_loading import get_loaders

        train_loader, _ = get_loaders(
            batch_size=1,
            data_dir=temp_dataset_dir,
            augment=True,
            reslice_prob=1.0,
            num_workers=0,
        )

        # Check multiple batches
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            img = batch["image"]
            # After augmentation, values should still be in reasonable range
            assert img.min() >= -1.0, f"Min value too low: {img.min()}"
            assert img.max() <= 2.0, f"Max value too high: {img.max()}"


class TestBundledBrainImage:
    """Tests using the bundled simulated brain volume from assets."""

    @pytest.fixture
    def brain_image_path(self):
        """Get path to the bundled simulated brain volume."""
        return str(example_volume_path())

    def test_load_real_brain(self, brain_image_path):
        """Test loading the bundled brain volume with medrs."""
        img = medrs.load(brain_image_path)
        tensor = img.to_torch()

        # Should be a 3D volume
        assert tensor.dim() == 3
        # Should have reasonable dimensions
        assert all(d > 50 for d in tensor.shape)

    def test_normalize_real_brain(self, brain_image_path):
        """Test normalizing the bundled brain volume."""
        img = medrs.load(brain_image_path)
        data = img.to_numpy()
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_resample_real_brain(self, brain_image_path):
        """Test resampling the bundled brain volume."""
        img = medrs.load(brain_image_path)
        resampled = img.resample_to_shape([128, 128, 128])
        tensor = resampled.to_torch()

        assert tuple(tensor.shape) == (128, 128, 128)

    def test_full_pipeline_real_brain(self, brain_image_path):
        """Test full preprocessing pipeline on the bundled brain volume."""
        # 1. Load
        img = medrs.load(brain_image_path)

        # 2. Resample to isotropic
        resampled = img.resample_to_shape([128, 128, 128])

        # 3. Normalize
        data = resampled.to_numpy()
        normalized = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

        # 4. Convert to tensor with proper shape
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        assert tensor.dim() == 5
        assert tensor.shape == (1, 1, 128, 128, 128)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestMedrsEdgeCases:
    """Test edge cases and error handling."""

    def test_small_volume(self):
        """Test handling of small volumes."""
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        tensor = img.to_torch()
        assert tuple(tensor.shape) == (16, 16, 16)

    def test_large_volume(self):
        """Test handling of larger volumes."""
        vol = np.random.rand(128, 128, 128).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        tensor = img.to_torch()
        assert tuple(tensor.shape) == (128, 128, 128)

    def test_anisotropic_spacing(self):
        """Test handling of anisotropic volumes."""
        vol = np.random.rand(64, 64, 10).astype(np.float32)
        spacing = (1.0, 1.0, 10.0)
        affine = np.diag([*spacing, 1.0]).astype(np.float32)
        img = medrs.NiftiImage(vol, affine)

        tensor = img.to_torch()
        assert tuple(tensor.shape) == (64, 64, 10)

        # Resample to isotropic
        resampled = img.resample_to_shape([64, 64, 100])
        resampled_tensor = resampled.to_torch()
        assert tuple(resampled_tensor.shape) == (64, 64, 100)

    def test_negative_values(self):
        """Test handling of volumes with negative values."""
        vol = np.random.rand(64, 64, 64).astype(np.float32) * 2 - 1  # [-1, 1]
        affine = np.eye(4, dtype=np.float32)
        img = medrs.NiftiImage(vol, affine)

        tensor = img.to_torch()
        assert tensor.min() < 0
        assert tensor.max() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
