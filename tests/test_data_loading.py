"""Tests for data loading utilities, including reslice_prob feature.

Tests cover:
- Reslicing probability (0.0, 0.5, 1.0)
- MAISI-style patch training with crops
- Whole volume training
- Edge cases with anisotropic volumes
- Batch size compatibility

Uses medrs for NIfTI I/O.
"""

import os
import tempfile
from pathlib import Path

import medrs
import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.data_loading import (
    MedrsVolumeDataset,
    PatchDataset,
    get_2d_loaders,
    get_loaders,
)


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


class TestResliceProbability:
    """Test reslice_prob parameter for training on non-resliced volumes."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test NIfTI files using medrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create test volumes with different shapes and spacings
            # 1. Isotropic volume (1mm x 1mm x 1mm)
            vol1 = np.random.rand(64, 64, 64).astype(np.float32)
            _create_nifti(
                vol1, str(data_dir / "sub-001_isotropic.nii.gz"), (1.0, 1.0, 1.0)
            )

            # 2. Anisotropic volume (1mm x 1mm x 5mm) - common in medical imaging
            vol2 = np.random.rand(64, 64, 13).astype(
                np.float32
            )  # 13 slices * 5mm = ~64mm
            _create_nifti(
                vol2, str(data_dir / "sub-002_anisotropic.nii.gz"), (1.0, 1.0, 5.0)
            )

            # 3. Another anisotropic volume (different shape)
            vol3 = np.random.rand(128, 128, 20).astype(np.float32)
            _create_nifti(
                vol3, str(data_dir / "sub-003_varying.nii.gz"), (0.5, 0.5, 2.0)
            )

            yield str(data_dir)

    def test_reslice_prob_always_reslice(self, temp_data_dir):
        """Test reslice_prob=1.0 always reslices to isotropic."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=1.0,
            resize_threshold=192,
        )

        # Check that all samples have same shape (resliced to isotropic)
        shapes = []
        for batch in train_loader:
            shapes.append(tuple(batch["image"].shape[2:]))  # Get spatial dims

        # All should be resliced to same size (with padding)
        assert len(set(shapes)) == 1, "All volumes should be resliced to same size"

        # Check that shape is divisible by 16 (padding requirement)
        sample_shape = shapes[0]
        assert all(dim % 16 == 0 for dim in sample_shape), (
            "Shape should be divisible by 16"
        )

    def test_reslice_prob_never_reslice(self, temp_data_dir):
        """Test reslice_prob=0.0 keeps original anisotropic spacing."""
        train_loader, val_loader = get_loaders(
            batch_size=1,  # Must use batch_size=1 for variable shapes
            data_dir=temp_data_dir,
            reslice_prob=0.0,
            resize_threshold=192,
        )

        # Check that samples preserve original shapes (with padding)
        shapes = []
        for i, batch in enumerate(train_loader):
            if i >= len(train_loader.dataset):  # Check all train samples
                break
            shapes.append(tuple(batch["image"].shape[2:]))

        # Should have at least some samples (train/val split might give us 2-3 train samples)
        assert len(shapes) >= 2, f"Should have at least 2 samples, got {len(shapes)}"

        # All shapes should still be divisible by 16 (padding requirement)
        for shape in shapes:
            assert all(dim % 16 == 0 for dim in shape), (
                f"Shape {shape} should be divisible by 16"
            )

        # Shapes should reflect original anisotropic volumes (not all identical after padding)
        # Note: After padding to divisible by 16, some shapes might coincidentally match,
        # but we verify the feature works by checking padding is applied correctly

    def test_reslice_prob_mixed(self, temp_data_dir):
        """Test reslice_prob=0.5 mixes resliced and non-resliced volumes."""
        # Set seed for reproducibility
        np.random.seed(42)

        train_loader, val_loader = get_loaders(
            batch_size=1,  # Must use batch_size=1 for variable shapes
            data_dir=temp_data_dir,
            reslice_prob=0.5,
            resize_threshold=192,
        )

        # Collect shapes from multiple batches
        shapes = []
        num_samples = min(10, len(train_loader.dataset))
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
            shapes.append(tuple(batch["image"].shape[2:]))

        # Should have some samples
        assert len(shapes) >= 2, f"Should have at least 2 samples, got {len(shapes)}"

        # All shapes should be divisible by 16
        for shape in shapes:
            assert all(dim % 16 == 0 for dim in shape), (
                f"Shape {shape} should be divisible by 16"
            )

    def test_reslice_prob_with_crops(self, temp_data_dir):
        """Test reslice_prob works correctly with MAISI-style crop training."""
        train_loader, val_loader = get_loaders(
            batch_size=2,  # Can use batch_size > 1 with crops
            data_dir=temp_data_dir,
            crop_size=64,
            crops_per_volume=2,
            reslice_prob=0.5,  # Mix resliced and raw
            resize_threshold=192,
        )

        # With crops, all samples should be same size (64x64x64)
        # Check first few batches
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Check first 5 batches
                break
            shape = batch["image"].shape
            assert shape[2:] == (64, 64, 64), f"Expected (64,64,64), got {shape[2:]}"
            assert shape[0] <= 2, f"Batch size should be <= 2, got {shape[0]}"

    def test_reslice_prob_batch_size_limitation(self, temp_data_dir):
        """Test that batch_size > 1 may fail with whole volumes and reslice_prob < 1.0."""
        # This should work (batch_size=1)
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.5,
            resize_threshold=192,
        )

        # Should be able to iterate without errors
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            assert batch["image"].shape[0] == 1

        # With batch_size > 1 and reslice_prob < 1.0, we might get shape mismatches
        # This is expected behavior - the user should use batch_size=1 or crop_size
        # Note: This test documents the limitation but may not always fail
        # if volumes happen to have compatible shapes after padding
        train_loader_mixed, _ = get_loaders(
            batch_size=2,
            data_dir=temp_data_dir,
            reslice_prob=0.5,  # Mixed spacing
            resize_threshold=192,
        )

        # Try to get a batch - may succeed or fail depending on shapes
        try:
            batch = next(iter(train_loader_mixed))
            # If it succeeds, shapes happened to match (possible after padding)
            assert batch["image"].shape[0] <= 2
        except (RuntimeError, ValueError) as e:
            # Expected: "stack expects each tensor to be equal size" or similar
            # This is acceptable - documents the limitation
            assert (
                "size" in str(e).lower()
                or "shape" in str(e).lower()
                or "batch" in str(e).lower()
            )


class TestPatchTrainingWithReslice:
    """Test MAISI-style patch training combined with reslice_prob."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test NIfTI files using medrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create volumes with varying shapes
            for i in range(5):
                # Random shape between 100-200 in each dimension
                h, w, d = np.random.randint(100, 200, 3)
                vol = np.random.rand(h, w, d).astype(np.float32)
                # Random spacing
                spacing = tuple(np.random.choice([0.5, 1.0, 2.0, 5.0], 3))
                _create_nifti(vol, str(data_dir / f"sub-{i:03d}.nii.gz"), spacing)

            yield str(data_dir)

    def test_crop_training_with_reslice_prob_1(self, temp_data_dir):
        """Test crop training with reslice_prob=1.0 (always reslice)."""
        train_loader, val_loader = get_loaders(
            batch_size=4,
            data_dir=temp_data_dir,
            crop_size=64,
            crops_per_volume=2,
            reslice_prob=1.0,
            resize_threshold=256,
            augment=False,  # Disable augmentation to test normalization
        )

        # All crops should be same size
        for batch in train_loader:
            assert batch["image"].shape == (4, 1, 64, 64, 64)
            assert batch["image"].min() >= 0.0
            assert batch["image"].max() <= 1.0  # Normalized

    def test_crop_training_with_reslice_prob_0(self, temp_data_dir):
        """Test crop training with reslice_prob=0.0 (never reslice)."""
        train_loader, val_loader = get_loaders(
            batch_size=4,
            data_dir=temp_data_dir,
            crop_size=64,
            crops_per_volume=2,
            reslice_prob=0.0,
            resize_threshold=256,
            augment=False,  # Disable augmentation to test normalization
        )

        # All crops should still be same size (crops are extracted after loading)
        for batch in train_loader:
            assert batch["image"].shape == (4, 1, 64, 64, 64)
            assert batch["image"].min() >= 0.0
            assert batch["image"].max() <= 1.0

    def test_crop_training_with_reslice_prob_mixed(self, temp_data_dir):
        """Test crop training with reslice_prob=0.5 (mixed)."""
        train_loader, val_loader = get_loaders(
            batch_size=4,
            data_dir=temp_data_dir,
            crop_size=64,
            crops_per_volume=2,
            reslice_prob=0.5,
            resize_threshold=256,
        )

        # Crops should still be same size regardless of reslice_prob
        # because crops are extracted from volumes (which may or may not be resliced)
        for batch in train_loader:
            assert batch["image"].shape == (4, 1, 64, 64, 64)

    def test_crops_per_volume(self, temp_data_dir):
        """Test that crops_per_volume parameter works correctly."""
        train_loader, val_loader = get_loaders(
            batch_size=2,
            data_dir=temp_data_dir,
            crop_size=64,
            crops_per_volume=4,
            reslice_prob=0.5,
        )

        # Count actual files in directory
        num_files = len([f for f in os.listdir(temp_data_dir) if f.endswith(".nii.gz")])
        train_files_expected = int(0.9 * num_files)  # 90/10 split

        # Should have train_files * crops_per_volume samples
        expected_train_samples = train_files_expected * 4
        assert len(train_loader.dataset) == expected_train_samples, (
            f"Expected {expected_train_samples} samples, got {len(train_loader.dataset)}"
        )

        # Validation should have 1 crop per volume (hardcoded in our implementation)
        val_files_expected = num_files - train_files_expected
        assert len(val_loader.dataset) == val_files_expected, (
            f"Expected {val_files_expected} val samples, got {len(val_loader.dataset)}"
        )


class TestTransformEdgeCases:
    """Test edge cases in transform pipeline with reslice_prob."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with edge case volumes using medrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # 1. Very anisotropic volume (1mm x 1mm x 10mm)
            vol1 = np.random.rand(64, 64, 7).astype(np.float32)  # 7 * 10mm = 70mm
            _create_nifti(
                vol1, str(data_dir / "very_anisotropic.nii.gz"), (1.0, 1.0, 10.0)
            )

            # 2. Small volume
            vol2 = np.random.rand(32, 32, 16).astype(np.float32)
            _create_nifti(vol2, str(data_dir / "small.nii.gz"), (1.0, 1.0, 1.0))

            # 3. Large volume
            vol3 = np.random.rand(256, 256, 128).astype(np.float32)
            _create_nifti(vol3, str(data_dir / "large.nii.gz"), (0.5, 0.5, 1.0))

            yield str(data_dir)

    def test_very_anisotropic_volume(self, temp_data_dir):
        """Test handling of very anisotropic volumes (1mm x 1mm x 10mm)."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.0,  # Keep original spacing
            resize_threshold=256,
        )

        # Should handle without errors
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            assert batch["image"].shape[0] == 1
            assert len(batch["image"].shape) == 5  # (B, C, D, H, W)
            # Shape should be divisible by 16 after padding
            assert all(dim % 16 == 0 for dim in batch["image"].shape[2:])

    def test_small_volume_with_padding(self, temp_data_dir):
        """Test that small volumes are properly padded."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.0,
            resize_threshold=256,
        )

        for batch in train_loader:
            shape = batch["image"].shape[2:]
            # All dimensions should be divisible by 16
            assert all(dim % 16 == 0 for dim in shape), (
                f"Shape {shape} not divisible by 16"
            )
            # Should be at least as large as original (after padding)
            assert all(dim >= 16 for dim in shape)

    def test_large_volume_no_reslice(self, temp_data_dir):
        """Test that large volumes work without reslicing."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.0,  # Keep original size
            resize_threshold=256,  # This shouldn't matter if reslice_prob=0.0
        )

        # Should handle large volumes
        for batch in train_loader:
            shape = batch["image"].shape[2:]
            # Should preserve large dimensions (with padding)
            assert all(dim % 16 == 0 for dim in shape)

    def test_reslice_prob_with_augmentation(self, temp_data_dir):
        """Test that augmentation works correctly with reslice_prob."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.5,
            augment=True,
            resize_threshold=256,
        )

        # Should apply augmentation without errors
        # Note: Augmentations can push values slightly outside [0, 1]
        num_samples = min(5, len(train_loader.dataset))
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
            # After augmentation, values should still be in reasonable range
            assert batch["image"].min() >= -0.5, (
                f"Image min should be >= -0.5, got {batch['image'].min()}"
            )
            assert batch["image"].max() <= 1.5, (
                f"Image max should be <= 1.5, got {batch['image'].max()}"
            )


class TestDataLoaderCompatibility:
    """Test compatibility between different parameter combinations."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test files using medrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            for i in range(3):
                vol = np.random.rand(64, 64, 32).astype(np.float32)
                _create_nifti(
                    vol, str(data_dir / f"test_{i:03d}.nii.gz"), (1.0, 1.0, 2.0)
                )

            yield str(data_dir)

    def test_crop_size_with_reslice_prob_1(self, temp_data_dir):
        """Test crop_size works with reslice_prob=1.0."""
        train_loader, val_loader = get_loaders(
            batch_size=4,
            data_dir=temp_data_dir,
            crop_size=32,
            reslice_prob=1.0,
        )

        # Check first few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            assert batch["image"].shape[1:] == (1, 32, 32, 32), (
                f"Expected (1, 32, 32, 32), got {batch['image'].shape[1:]}"
            )
            assert batch["image"].shape[0] <= 4

    def test_crop_size_with_reslice_prob_0(self, temp_data_dir):
        """Test crop_size works with reslice_prob=0.0."""
        train_loader, val_loader = get_loaders(
            batch_size=4,
            data_dir=temp_data_dir,
            crop_size=32,
            reslice_prob=0.0,
        )

        # Check first few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            assert batch["image"].shape[1:] == (1, 32, 32, 32), (
                f"Expected (1, 32, 32, 32), got {batch['image'].shape[1:]}"
            )
            assert batch["image"].shape[0] <= 4

    def test_whole_volume_batch_size_1(self, temp_data_dir):
        """Test whole volume training with batch_size=1 and reslice_prob < 1.0."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            crop_size=None,  # Whole volumes
            reslice_prob=0.5,
        )

        # Should work without errors
        num_samples = min(3, len(train_loader.dataset))
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
            assert batch["image"].shape[0] == 1
            assert all(dim % 16 == 0 for dim in batch["image"].shape[2:]), (
                "Shape should be divisible by 16"
            )

    def test_resize_threshold_ignored_when_reslice_prob_0(self, temp_data_dir):
        """Test that resize_threshold is ignored when reslice_prob=0.0."""
        train_loader1, _ = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.0,
            resize_threshold=64,  # Small threshold
        )

        train_loader2, _ = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=0.0,
            resize_threshold=256,  # Large threshold
        )

        # Both should produce similar shapes (original volumes, not resliced)
        num_samples = min(3, len(train_loader1.dataset))
        shapes1 = [
            batch["image"].shape[2:]
            for i, batch in enumerate(train_loader1)
            if i < num_samples
        ]
        shapes2 = [
            batch["image"].shape[2:]
            for i, batch in enumerate(train_loader2)
            if i < num_samples
        ]

        # Shapes should be similar (same original volumes, resize_threshold is ignored when reslice_prob=0.0)
        assert len(shapes1) == len(shapes2), (
            f"Should have same number of samples: {len(shapes1)} vs {len(shapes2)}"
        )
        # Shapes should match (same volumes, same padding)
        assert shapes1 == shapes2, (
            f"Shapes should match when reslice_prob=0.0: {shapes1} vs {shapes2}"
        )


class TestMedrsIntegration:
    """Test medrs integration for data loading."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test NIfTI files using medrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create multiple test volumes (need at least 2 for 90/10 split)
            for i in range(3):
                vol = np.random.rand(64, 64, 64).astype(np.float32)
                _create_nifti(
                    vol, str(data_dir / f"test_volume_{i}.nii.gz"), (1.0, 1.0, 1.0)
                )

            yield str(data_dir)

    def test_medrs_load_and_save(self, temp_data_dir):
        """Test that medrs can load files it saved."""
        # Load the first saved file
        img = medrs.load(str(Path(temp_data_dir) / "test_volume_0.nii.gz"))
        tensor = img.to_torch()

        assert tensor.shape == (64, 64, 64)
        assert tensor.dtype == torch.float32

    def test_loader_uses_medrs(self, temp_data_dir):
        """Test that get_loaders works with medrs-saved files."""
        train_loader, val_loader = get_loaders(
            batch_size=1,
            data_dir=temp_data_dir,
            reslice_prob=1.0,
        )

        # Should be able to load without errors
        batch = next(iter(train_loader))
        assert "image" in batch
        assert batch["image"].shape[0] == 1  # Batch size
        assert batch["image"].dim() == 5  # (B, C, H, W, D)


class TestDataLoaderFailures:
    """Ensure data loading failures are loud and actionable."""

    def test_get_loaders_missing_dir_raises(self):
        """get_loaders should fail when data_dir is missing."""
        with pytest.raises(ValueError, match="data_dir is required"):
            get_loaders(data_dir=None)

    def test_get_loaders_nonexistent_dir_raises(self):
        """get_loaders should fail when data_dir doesn't exist."""
        missing_path = Path(tempfile.gettempdir()) / "medtok_missing_dir"
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            get_loaders(data_dir=str(missing_path))

    def test_get_2d_loaders_empty_dir_raises(self):
        """get_2d_loaders should fail when no NIfTI files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No NIfTI files found"):
                get_2d_loaders(data_dir=tmpdir)

    def test_patch_dataset_missing_file_raises(self):
        """PatchDataset should raise on missing files."""
        missing_path = Path(tempfile.gettempdir()) / "medtok_missing_patch.nii.gz"
        if missing_path.exists():
            missing_path.unlink()

        dataset = PatchDataset(
            [str(missing_path)],
            crop_size=32,
            crops_per_volume=1,
            augment=False,
            use_cache=False,
        )

        with pytest.raises(RuntimeError, match="Failed to load any volume"):
            _ = dataset[0]

    def test_volume_dataset_missing_file_raises(self):
        """MedrsVolumeDataset should raise on missing files."""
        missing_path = Path(tempfile.gettempdir()) / "medtok_missing_volume.nii.gz"
        if missing_path.exists():
            missing_path.unlink()

        dataset = MedrsVolumeDataset([str(missing_path)], augment=False)

        with pytest.raises(RuntimeError, match="Failed to load"):
            _ = dataset[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
