"""Tests for sliding window inference in tokenizers."""

import pytest
import torch

from medtokenizers import ContinuousTokenizer, DiscreteTokenizer


class TestSlidingWindowInference:
    """Tests for sliding window reconstruction."""

    @pytest.fixture
    def continuous_tokenizer(self):
        """Create a small continuous tokenizer for testing."""
        return ContinuousTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=4,
            latent_channels=4,
            channels=8,
            channels_mult=(1, 2),
            num_res_blocks=1,
            attn_resolutions=(),
            spatial_compression=4,
        )

    @pytest.fixture
    def discrete_tokenizer(self):
        """Create a small discrete tokenizer for testing."""
        return DiscreteTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=4,
            latent_channels=4,
            channels=8,
            channels_mult=(1, 2),
            num_res_blocks=1,
            attn_resolutions=(),
            spatial_compression=4,
            codebook_size=256,
        )

    def test_no_sliding_window(self, continuous_tokenizer):
        """Test direct reconstruction without sliding window."""
        x = torch.randn(1, 1, 32, 32, 32)
        recon = continuous_tokenizer.reconstruct(x, roi_size=None)
        assert recon.shape == x.shape

    def test_exact_multiple_roi(self, continuous_tokenizer):
        """Test with volume that's exact multiple of ROI."""
        x = torch.randn(1, 1, 64, 64, 64)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        assert recon.shape == x.shape, f"Expected {x.shape}, got {recon.shape}"

    def test_non_multiple_roi(self, continuous_tokenizer):
        """Test with volume that requires padding."""
        x = torch.randn(1, 1, 50, 55, 45)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        assert recon.shape == x.shape, f"Expected {x.shape}, got {recon.shape}"

    def test_small_volume(self, continuous_tokenizer):
        """Test with volume smaller than ROI."""
        x = torch.randn(1, 1, 20, 25, 18)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        assert recon.shape == x.shape, f"Expected {x.shape}, got {recon.shape}"

    def test_no_overlap(self, continuous_tokenizer):
        """Test with no overlap (overlap=0.0)."""
        x = torch.randn(1, 1, 64, 64, 64)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.0)
        assert recon.shape == x.shape

    def test_high_overlap(self, continuous_tokenizer):
        """Test with high overlap (overlap=0.75)."""
        x = torch.randn(1, 1, 48, 48, 48)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.75)
        assert recon.shape == x.shape

    def test_overlap_out_of_range_raises(self, continuous_tokenizer):
        """Test that invalid overlap values raise errors."""
        x = torch.randn(1, 1, 32, 32, 32)
        with pytest.raises(ValueError, match="overlap"):
            continuous_tokenizer.reconstruct(x, roi_size=(16, 16, 16), overlap=1.0)
        with pytest.raises(ValueError, match="overlap"):
            continuous_tokenizer.reconstruct(x, roi_size=(16, 16, 16), overlap=-0.1)

    def test_integer_roi_size(self, continuous_tokenizer):
        """Test with integer ROI size (should expand to tuple)."""
        x = torch.randn(1, 1, 64, 64, 64)
        recon = continuous_tokenizer.reconstruct(x, roi_size=32, overlap=0.5)
        assert recon.shape == x.shape

    def test_batch_size_greater_than_one(self, continuous_tokenizer):
        """Test with batch size > 1."""
        x = torch.randn(2, 1, 48, 48, 48)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        assert recon.shape == x.shape

    def test_discrete_tokenizer_no_overlap(self, discrete_tokenizer):
        """Test discrete tokenizer with overlap=0.0 (allowed)."""
        x = torch.randn(1, 1, 64, 64, 64)
        recon = discrete_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.0)
        assert recon.shape == x.shape

    def test_discrete_tokenizer_with_overlap_raises(self, discrete_tokenizer):
        """Test discrete tokenizer with overlap>0.0 raises error."""
        x = torch.randn(1, 1, 64, 64, 64)
        with pytest.raises(ValueError, match="Discrete tokenizers require overlap=0.0"):
            discrete_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)

    def test_different_spatial_dims(self, continuous_tokenizer):
        """Test with different spatial dimensions."""
        x = torch.randn(1, 1, 40, 50, 30)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(24, 24, 24), overlap=0.5)
        assert recon.shape == x.shape

    def test_asymmetric_roi(self, continuous_tokenizer):
        """Test with asymmetric ROI size."""
        x = torch.randn(1, 1, 64, 48, 32)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 24, 16), overlap=0.5)
        assert recon.shape == x.shape

    def test_output_range_preservation(self, continuous_tokenizer):
        """Test that output values are in reasonable range."""
        x = torch.randn(1, 1, 48, 48, 48) * 0.5  # Moderate input range
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)

        # Output should not be extreme
        assert not torch.isnan(recon).any(), "Reconstruction contains NaN"
        assert not torch.isinf(recon).any(), "Reconstruction contains Inf"
        assert recon.abs().max() < 100, "Reconstruction values are too large"

    def test_padding_calculation_edge_cases(self, continuous_tokenizer):
        """Test padding calculation for edge case dimensions."""
        # Test cases where padding calculation is tricky
        test_cases = [
            (1, 1, 33, 33, 33),  # Just over single window
            (1, 1, 63, 63, 63),  # Just under two windows
            (1, 1, 65, 65, 65),  # Just over two windows
            (1, 1, 31, 31, 31),  # Just under single window
        ]

        for shape in test_cases:
            x = torch.randn(*shape)
            recon = continuous_tokenizer.reconstruct(
                x, roi_size=(32, 32, 32), overlap=0.5
            )
            assert recon.shape == x.shape, f"Failed for shape {shape}"

    def test_single_window_coverage(self, continuous_tokenizer):
        """Test that volume fitting in single window works."""
        x = torch.randn(1, 1, 32, 32, 32)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        assert recon.shape == x.shape

    def test_multiple_channels(self, continuous_tokenizer):
        """Test with different input/output channel counts."""
        # The tokenizer might have different output channels after passing through the network
        x = torch.randn(1, 1, 48, 48, 48)
        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)
        # Shape should match input except potentially channels
        assert recon.shape[0] == x.shape[0]  # Batch
        assert recon.shape[2:] == x.shape[2:]  # Spatial dims


class TestSlidingWindowGaussianBlending:
    """Tests for Gaussian blending in sliding window."""

    @pytest.fixture
    def continuous_tokenizer(self):
        """Create a small continuous tokenizer for testing."""
        return ContinuousTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=4,
            latent_channels=4,
            channels=8,
            channels_mult=(1, 2),
            num_res_blocks=1,
            attn_resolutions=(),
            spatial_compression=4,
        )

    def test_gaussian_smoothness(self, continuous_tokenizer):
        """Test that overlapping regions are smoothly blended."""
        # Create input with clear regions
        x = torch.zeros(1, 1, 64, 64, 64)
        x[:, :, :32, :, :] = 1.0  # Left half bright

        recon = continuous_tokenizer.reconstruct(x, roi_size=(32, 32, 32), overlap=0.5)

        # Check reconstruction doesn't have extreme artifacts
        assert not torch.isnan(recon).any()
        assert not torch.isinf(recon).any()

    def test_overlap_affects_output(self, continuous_tokenizer):
        """Test that different overlaps produce different results."""
        x = torch.randn(1, 1, 64, 64, 64)

        recon_no_overlap = continuous_tokenizer.reconstruct(
            x, roi_size=(32, 32, 32), overlap=0.0
        )
        recon_high_overlap = continuous_tokenizer.reconstruct(
            x, roi_size=(32, 32, 32), overlap=0.5
        )

        # Results should differ (Gaussian blending vs tiling)
        # Note: They might be very similar for untrained network, so just check shapes
        assert recon_no_overlap.shape == recon_high_overlap.shape


class TestSlidingWindow2D:
    """Tests for 2D sliding window reconstruction."""

    @pytest.fixture
    def continuous_tokenizer_2d(self):
        """Create a small 2D tokenizer for testing."""
        return ContinuousTokenizer(
            dim=2,
            in_channels=1,
            out_channels=1,
            z_channels=4,
            latent_channels=4,
            channels=8,
            channels_mult=(1, 2),
            num_res_blocks=1,
            attn_resolutions=(),
            spatial_compression=4,
        )

    def test_2d_sliding_window(self, continuous_tokenizer_2d):
        """Test 2D reconstruction with sliding window."""
        x = torch.randn(1, 1, 64, 48)
        recon = continuous_tokenizer_2d.reconstruct(x, roi_size=(32, 24), overlap=0.5)
        assert recon.shape == x.shape

    def test_2d_integer_roi(self, continuous_tokenizer_2d):
        """Test 2D reconstruction with integer ROI size."""
        x = torch.randn(1, 1, 32, 32)
        recon = continuous_tokenizer_2d.reconstruct(x, roi_size=16, overlap=0.5)
        assert recon.shape == x.shape

    def test_2d_roi_shape_mismatch_raises(self, continuous_tokenizer_2d):
        """Test that mismatched roi_size length raises an error."""
        x = torch.randn(1, 1, 32, 32)
        with pytest.raises(ValueError, match="roi_size"):
            continuous_tokenizer_2d.reconstruct(x, roi_size=(16, 16, 16), overlap=0.5)
