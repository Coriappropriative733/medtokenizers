"""Tests for preprocessing utilities."""

import medrs
import numpy as np
import pytest
import torch

from medtokenizers.preprocessing import load_nifti, save_nifti
from medtokenizers.training.preprocessing import (
    CenterCrop3d,
    HounsfieldNormalizer,
    RandomCrop3d,
)


class TestHounsfieldNormalizer:
    """Tests for HounsfieldNormalizer."""

    def test_default_normalization(self):
        normalizer = HounsfieldNormalizer()
        ct = torch.tensor([[-1000.0, 0.0, 500.0, 1000.0]])

        normalized = normalizer(ct)

        assert torch.allclose(normalized, torch.tensor([[0.0, 0.5, 0.75, 1.0]]))

    def test_custom_hu_range(self):
        normalizer = HounsfieldNormalizer(hu_min=-500, hu_max=500)
        ct = torch.tensor([[-500.0, 0.0, 500.0]])

        normalized = normalizer(ct)

        assert torch.allclose(normalized, torch.tensor([[0.0, 0.5, 1.0]]))

    def test_clipping(self):
        normalizer = HounsfieldNormalizer(clip=True)
        ct = torch.tensor([[-2000.0, 0.0, 2000.0]])

        normalized = normalizer(ct)

        # -2000 clipped to -1000 -> 0.0
        # 2000 clipped to 1000 -> 1.0
        assert torch.allclose(normalized, torch.tensor([[0.0, 0.5, 1.0]]))

    def test_no_clipping(self):
        normalizer = HounsfieldNormalizer(clip=False)
        ct = torch.tensor([[-2000.0, 0.0, 2000.0]])

        normalized = normalizer(ct)

        # -2000 -> -0.5, 0 -> 0.5, 2000 -> 1.5
        expected = (ct - (-1000.0)) / 2000.0
        assert torch.allclose(normalized, expected)

    def test_custom_output_range(self):
        normalizer = HounsfieldNormalizer(output_range=(-1.0, 1.0))
        ct = torch.tensor([[-1000.0, 0.0, 1000.0]])

        normalized = normalizer(ct)

        assert torch.allclose(normalized, torch.tensor([[-1.0, 0.0, 1.0]]))

    def test_denormalization(self):
        normalizer = HounsfieldNormalizer()
        ct_original = torch.randn(2, 1, 16, 16) * 500  # Random HU values

        normalized = normalizer(ct_original)
        denormalized = normalizer.denormalize(normalized)

        # After clipping, should match within clipped range
        ct_clipped = torch.clamp(ct_original, -1000, 1000)
        assert torch.allclose(denormalized, ct_clipped, atol=1e-4)

    def test_denormalization_custom_range(self):
        normalizer = HounsfieldNormalizer(output_range=(-1.0, 1.0))
        ct_original = torch.tensor([[-1000.0, 0.0, 500.0, 1000.0]])

        normalized = normalizer(ct_original)
        denormalized = normalizer.denormalize(normalized)

        assert torch.allclose(denormalized, ct_original, atol=1e-4)

    def test_2d_input(self):
        normalizer = HounsfieldNormalizer()
        ct = torch.randn(2, 1, 64, 64) * 500

        normalized = normalizer(ct)

        assert normalized.shape == ct.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_3d_input(self):
        normalizer = HounsfieldNormalizer()
        ct = torch.randn(1, 1, 16, 32, 32) * 500

        normalized = normalizer(ct)

        assert normalized.shape == ct.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_batch_processing(self):
        normalizer = HounsfieldNormalizer()
        ct_batch = torch.randn(4, 1, 32, 32) * 500

        normalized = normalizer(ct_batch)

        assert normalized.shape[0] == 4
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_boundary_values(self):
        normalizer = HounsfieldNormalizer()
        ct = torch.tensor([[-1000.0, 1000.0]])

        normalized = normalizer(ct)

        assert torch.allclose(normalized, torch.tensor([[0.0, 1.0]]))

    def test_multichannel_input(self):
        normalizer = HounsfieldNormalizer()
        ct = torch.randn(1, 3, 32, 32) * 500  # 3 channels

        normalized = normalizer(ct)

        assert normalized.shape == ct.shape


class TestPreprocessingEdgeCases:
    """Edge case tests for preprocessing utilities."""

    def test_normalizer_zero_range(self):
        # When hu_min == hu_max, division by zero produces nan/inf
        normalizer = HounsfieldNormalizer(hu_min=0, hu_max=0)
        ct = torch.zeros(1, 1, 16, 16)

        # Division by zero produces nan/inf but doesn't crash
        normalized = normalizer(ct)
        assert torch.isnan(normalized).any() or torch.isinf(normalized).any()

    def test_normalizer_reversed_range(self):
        # When hu_max < hu_min
        normalizer = HounsfieldNormalizer(hu_min=1000, hu_max=-1000)
        ct = torch.tensor([[0.0]])

        normalized = normalizer(ct)

        # Should produce inverted normalization
        assert normalized.shape == ct.shape

    def test_normalizer_gradient_flow(self):
        normalizer = HounsfieldNormalizer()
        # Use values within the normalization range to avoid clipping
        ct_data = torch.randn(1, 1, 16, 16) * 200
        ct = ct_data.clone().requires_grad_(True)

        normalized = normalizer(ct)
        loss = normalized.sum()
        loss.backward()

        # Check that gradients flow back to the input (ct is a leaf tensor)
        assert ct.grad is not None
        # For values in [-1000, 1000], gradient should be 1/2000
        assert torch.allclose(ct.grad, torch.ones_like(ct) / 2000.0)

    def test_normalizer_device_consistency(self):
        if torch.cuda.is_available():
            normalizer = HounsfieldNormalizer().cuda()
            ct = torch.randn(1, 1, 16, 16, device="cuda") * 500

            normalized = normalizer(ct)

            assert normalized.device.type == "cuda"


class TestRandomCrop3d:
    """Tests for RandomCrop3d."""

    def test_basic_crop(self):
        cropper = RandomCrop3d(crop_size=32)
        volume = torch.randn(2, 1, 64, 64, 64)

        cropped = cropper(volume)

        assert cropped.shape == (2, 1, 32, 32, 32)

    def test_crop_with_tuple_size(self):
        cropper = RandomCrop3d(crop_size=(16, 24, 32))
        volume = torch.randn(1, 1, 64, 64, 64)

        cropped = cropper(volume)

        assert cropped.shape == (1, 1, 16, 24, 32)

    def test_crop_deterministic_values(self):
        """Test that crop extracts actual values from volume."""
        # Create volume with known values
        volume = torch.arange(8).view(1, 1, 2, 2, 2).float()
        cropper = RandomCrop3d(crop_size=1)

        cropped = cropper(volume)

        # Should extract a single voxel with a value in [0, 7]
        assert cropped.shape == (1, 1, 1, 1, 1)
        assert cropped.item() in range(8)

    def test_crop_preserves_channels(self):
        cropper = RandomCrop3d(crop_size=16)
        volume = torch.randn(1, 3, 32, 32, 32)

        cropped = cropper(volume)

        assert cropped.shape == (1, 3, 16, 16, 16)

    def test_crop_size_equals_volume(self):
        cropper = RandomCrop3d(crop_size=32)
        volume = torch.randn(1, 1, 32, 32, 32)

        cropped = cropper(volume)

        assert cropped.shape == (1, 1, 32, 32, 32)
        assert torch.allclose(cropped, volume)

    def test_crop_too_large_raises_error(self):
        cropper = RandomCrop3d(crop_size=64)
        volume = torch.randn(1, 1, 32, 32, 32)

        with pytest.raises(ValueError, match="smaller than crop size"):
            cropper(volume)

    def test_crop_with_padding(self):
        cropper = RandomCrop3d(crop_size=32, padding=4)
        volume = torch.randn(1, 1, 32, 32, 32)

        cropped = cropper(volume)

        # After padding, volume is 40x40x40, crop should succeed
        assert cropped.shape == (1, 1, 32, 32, 32)

    def test_crop_pad_if_needed(self):
        cropper = RandomCrop3d(crop_size=64, pad_if_needed=True)
        volume = torch.randn(1, 1, 32, 32, 32)

        cropped = cropper(volume)

        assert cropped.shape == (1, 1, 64, 64, 64)

    def test_crop_randomness(self):
        """Test that multiple crops are different (probabilistically)."""
        cropper = RandomCrop3d(crop_size=32)
        volume = torch.randn(1, 1, 128, 128, 128)

        crop1 = cropper(volume)
        crop2 = cropper(volume)

        # With very high probability, different random crops should differ
        # (They could be the same, but extremely unlikely for 128³→32³)
        assert not torch.allclose(crop1, crop2)

    def test_crop_gradient_flow(self):
        cropper = RandomCrop3d(crop_size=16)
        volume = torch.randn(1, 1, 32, 32, 32, requires_grad=True)

        cropped = cropper(volume)
        loss = cropped.sum()
        loss.backward()

        # Gradients should flow back (some voxels will have grad=1, others=0)
        assert volume.grad is not None
        assert volume.grad.sum() > 0


class TestCenterCrop3d:
    """Tests for CenterCrop3d."""

    def test_basic_center_crop(self):
        cropper = CenterCrop3d(crop_size=32)
        volume = torch.randn(1, 1, 64, 64, 64)

        cropped = cropper(volume)

        assert cropped.shape == (1, 1, 32, 32, 32)

    def test_center_crop_with_tuple(self):
        cropper = CenterCrop3d(crop_size=(16, 24, 32))
        volume = torch.randn(1, 1, 64, 64, 64)

        cropped = cropper(volume)

        assert cropped.shape == (1, 1, 16, 24, 32)

    def test_center_crop_deterministic(self):
        """Test that center crop is deterministic."""
        cropper = CenterCrop3d(crop_size=32)
        volume = torch.randn(2, 3, 64, 64, 64)

        crop1 = cropper(volume)
        crop2 = cropper(volume)

        assert torch.allclose(crop1, crop2)

    def test_center_crop_extracts_center(self):
        """Test that center crop extracts the center region."""
        # Create volume with known center value
        volume = torch.zeros(1, 1, 10, 10, 10)
        volume[:, :, 4:6, 4:6, 4:6] = 1.0  # Center 2x2x2 is 1

        cropper = CenterCrop3d(crop_size=2)
        cropped = cropper(volume)

        # Should extract the center 2x2x2
        assert torch.allclose(cropped, torch.ones(1, 1, 2, 2, 2))

    def test_center_crop_preserves_channels(self):
        cropper = CenterCrop3d(crop_size=16)
        volume = torch.randn(1, 4, 32, 32, 32)

        cropped = cropper(volume)

        assert cropped.shape == (1, 4, 16, 16, 16)

    def test_center_crop_size_equals_volume(self):
        cropper = CenterCrop3d(crop_size=32)
        volume = torch.randn(1, 1, 32, 32, 32)

        cropped = cropper(volume)

        assert torch.allclose(cropped, volume)

    def test_center_crop_too_large_raises_error(self):
        cropper = CenterCrop3d(crop_size=64)
        volume = torch.randn(1, 1, 32, 32, 32)

        with pytest.raises(ValueError, match="smaller than crop size"):
            cropper(volume)

    def test_center_crop_gradient_flow(self):
        cropper = CenterCrop3d(crop_size=16)
        volume = torch.randn(1, 1, 32, 32, 32, requires_grad=True)

        cropped = cropper(volume)
        loss = cropped.sum()
        loss.backward()

        assert volume.grad is not None
        # Center region should have gradients
        assert volume.grad[:, :, 8:24, 8:24, 8:24].sum() > 0


class TestNiftiRoundTrip:
    """Cover the NIfTI write path, which medrs cannot serve on its own."""

    def test_save_nifti_then_load_with_medrs(self, tmp_path):
        volume = np.random.rand(8, 9, 10).astype(np.float32)
        path = tmp_path / "volume.nii.gz"

        save_nifti(volume, path, spacing=(1.5, 1.5, 2.0))

        assert path.exists()
        img = medrs.load(str(path))
        assert tuple(img.to_numpy().shape) == volume.shape
        np.testing.assert_allclose(img.to_numpy(), volume, rtol=1e-5, atol=1e-5)

    def test_save_nifti_accepts_tensor_and_squeezes_batch_dims(self, tmp_path):
        volume = torch.rand(1, 1, 8, 9, 10)
        path = tmp_path / "tensor.nii.gz"

        save_nifti(volume, path)

        assert tuple(medrs.load(str(path)).to_numpy().shape) == (8, 9, 10)

    def test_load_nifti_returns_volume_and_spacing(self, tmp_path):
        volume = np.random.rand(8, 9, 10).astype(np.float32)
        path = tmp_path / "volume.nii.gz"
        save_nifti(volume, path, spacing=(1.5, 1.5, 2.0))

        loaded, metadata = load_nifti(path)

        assert loaded.shape == volume.shape
        assert metadata["original_shape"] == volume.shape
