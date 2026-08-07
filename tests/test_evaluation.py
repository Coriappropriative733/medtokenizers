"""Tests for evaluation functionality."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from medtokenizers.evaluation import (
    SimpleDataset,
    TokenizerEvaluator,
    compute_all_metrics,
    compute_codebook_usage,
    compute_compression_ratio,
    compute_mae,
    compute_mse,
    compute_perplexity,
    compute_psnr,
    compute_ssim,
)
from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer


class TestBasicMetrics:
    """Test basic reconstruction metrics."""

    def test_compute_mse(self):
        """Test MSE computation."""
        # Perfect reconstruction
        x = torch.randn(4, 1, 32, 32, 32)
        mse = compute_mse(x, x)
        assert mse == pytest.approx(0.0, abs=1e-6)

        # Different inputs
        y = torch.randn(4, 1, 32, 32, 32)
        mse = compute_mse(x, y)
        assert mse > 0

        # Test with numpy
        x_np = x.numpy()
        y_np = y.numpy()
        mse_np = compute_mse(x_np, y_np)
        assert isinstance(mse_np, float)

    def test_compute_mae(self):
        """Test MAE computation."""
        # Perfect reconstruction
        x = torch.randn(4, 1, 32, 32, 32)
        mae = compute_mae(x, x)
        assert mae == pytest.approx(0.0, abs=1e-6)

        # Different inputs
        y = torch.randn(4, 1, 32, 32, 32)
        mae = compute_mae(x, y)
        assert mae > 0

    def test_compute_psnr(self):
        """Test PSNR computation."""
        # Perfect reconstruction should give infinite PSNR
        x = torch.randn(4, 1, 32, 32, 32)
        psnr = compute_psnr(x, x, data_range=1.0)
        assert psnr == float("inf")

        # Different inputs should give finite PSNR
        y = x + 0.1 * torch.randn_like(x)
        psnr = compute_psnr(x, y, data_range=1.0)
        assert 0 < psnr < 100

    def test_compute_ssim(self):
        """Test SSIM computation."""
        # Test 2D images
        x_2d = torch.randn(4, 1, 64, 64)
        ssim_2d = compute_ssim(x_2d, x_2d, data_range=1.0)
        assert 0.99 < ssim_2d <= 1.0

        # Test 3D volumes
        x_3d = torch.randn(4, 1, 32, 32, 32)
        ssim_3d = compute_ssim(x_3d, x_3d, data_range=1.0)
        assert 0.99 < ssim_3d <= 1.0

        # Different inputs should have lower SSIM
        y = x_3d + 0.2 * torch.randn_like(x_3d)
        ssim = compute_ssim(x_3d, y, data_range=1.0)
        assert 0 < ssim < 1.0

    def test_metrics_with_mask(self):
        """Test metrics with mask."""
        x = torch.randn(4, 1, 32, 32, 32)
        y = torch.randn(4, 1, 32, 32, 32)
        mask = torch.ones_like(x)
        mask[:, :, :16, :, :] = 0  # Mask out half

        mse_masked = compute_mse(x, y, mask=mask)
        mse_full = compute_mse(x, y)

        assert mse_masked != mse_full


class TestDiscreteMetrics:
    """Test discrete tokenizer metrics."""

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        # Uniform distribution should give perplexity ≈ codebook_size
        codebook_size = 1024
        num_samples = 10000
        indices = torch.randint(0, codebook_size, (num_samples,))

        perplexity = compute_perplexity(indices, codebook_size)
        # Should be close to codebook_size for uniform distribution
        assert 900 < perplexity < 1100

        # Single code should give perplexity = 1
        indices_single = torch.zeros(num_samples, dtype=torch.long)
        perplexity_single = compute_perplexity(indices_single, codebook_size)
        assert perplexity_single == pytest.approx(1.0, abs=0.1)

    def test_compute_perplexity_large_codebook(self):
        """Test perplexity with large codebook (OOM guard via unique-based counting)."""
        # LFQ-like codebook: 2^16 = 65536 entries, but only sparse usage
        large_codebook_size = 2**16
        num_samples = 1000
        # Only use a small subset of codes (sparse usage)
        indices = torch.randint(0, 100, (num_samples,))

        # This should not OOM because we use unique-based counting
        perplexity = compute_perplexity(indices, large_codebook_size)
        assert 50 < perplexity < 110  # ~100 codes used uniformly

    def test_compute_codebook_usage(self):
        """Test codebook usage computation."""
        codebook_size = 1024

        # All codes used
        indices = torch.arange(codebook_size).repeat(10)
        usage = compute_codebook_usage(indices, codebook_size)
        assert usage == pytest.approx(100.0, abs=0.1)

        # Half of codes used
        indices = torch.randint(0, codebook_size // 2, (10000,))
        usage = compute_codebook_usage(indices, codebook_size)
        assert 40 < usage < 60  # Should be around 50%

        # Single code used
        indices = torch.zeros(1000, dtype=torch.long)
        usage = compute_codebook_usage(indices, codebook_size)
        assert usage == pytest.approx(100.0 / codebook_size, abs=0.1)


class TestCompressionRatio:
    """Test compression ratio computation."""

    def test_compute_compression_ratio(self):
        """Test compression ratio computation."""
        # 8x spatial compression, 1 channel -> 4 channels
        input_shape = (8, 1, 256, 256, 256)
        latent_shape = (8, 4, 32, 32, 32)

        ratio = compute_compression_ratio(input_shape, latent_shape)

        # Expected: (8*1*256*256*256) / (8*4*32*32*32) = 128
        expected_ratio = (256 * 256 * 256) / (4 * 32 * 32 * 32)
        assert ratio == pytest.approx(expected_ratio, rel=0.01)


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""

    def test_continuous_tokenizer_metrics(self):
        """Test all metrics for continuous tokenizer."""
        recon = torch.randn(4, 1, 32, 32, 32)
        target = torch.randn(4, 1, 32, 32, 32)

        metrics = compute_all_metrics(recon, target, data_range=1.0)

        # Check that all basic metrics are computed
        assert metrics.mse is not None
        assert metrics.mae is not None
        assert metrics.psnr is not None
        assert metrics.ssim is not None

        # Discrete metrics should be None
        assert metrics.perplexity is None
        assert metrics.codebook_usage is None

    def test_discrete_tokenizer_metrics(self):
        """Test all metrics for discrete tokenizer."""
        recon = torch.randn(4, 1, 32, 32, 32)
        target = torch.randn(4, 1, 32, 32, 32)
        indices = torch.randint(0, 1024, (4, 32, 32, 32))

        metrics = compute_all_metrics(
            recon, target, data_range=1.0, indices=indices, codebook_size=1024
        )

        # Check that all metrics are computed
        assert metrics.mse is not None
        assert metrics.mae is not None
        assert metrics.psnr is not None
        assert metrics.ssim is not None
        assert metrics.perplexity is not None
        assert metrics.codebook_usage is not None


class TestTokenizerEvaluator:
    """Test TokenizerEvaluator class."""

    @pytest.fixture
    def continuous_model(self):
        """Create a small continuous tokenizer for testing."""
        model = ContinuousTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=32,
            latent_channels=4,
            channels=16,
            channels_mult=(1, 2),
            num_res_blocks=1,
            spatial_compression=4,
            formulation="VAE",
        )
        model.eval()
        return model

    @pytest.fixture
    def discrete_model(self):
        """Create a small discrete tokenizer for testing."""
        model = DiscreteTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=32,
            embedding_dim=6,
            channels=16,
            channels_mult=(1, 2),
            num_res_blocks=1,
            spatial_compression=4,
            quantizer="FSQ",
            levels=[8, 8, 8],
        )
        model.eval()
        return model

    def test_evaluator_continuous(self, continuous_model):
        """Test evaluator with continuous tokenizer."""
        evaluator = TokenizerEvaluator(continuous_model, device="cpu", data_range=1.0)

        # Create test data
        images = torch.randn(8, 1, 32, 32, 32)

        # Evaluate batch
        results = evaluator.evaluate_batch(images)

        assert "metrics" in results
        assert "compression_ratio" in results
        assert "reconstructions" in results
        assert results["metrics"].mse is not None
        assert results["metrics"].psnr is not None
        assert results["compression_ratio"] > 1.0

    def test_evaluator_discrete(self, discrete_model):
        """Test evaluator with discrete tokenizer."""
        evaluator = TokenizerEvaluator(discrete_model, device="cpu", data_range=1.0)

        # Create test data
        images = torch.randn(8, 1, 32, 32, 32)

        # Evaluate batch
        results = evaluator.evaluate_batch(images)

        assert "metrics" in results
        assert "compression_ratio" in results
        assert "indices" in results
        assert results["metrics"].mse is not None
        assert results["metrics"].perplexity is not None
        assert results["metrics"].codebook_usage is not None

    def test_evaluator_dataloader(self, continuous_model):
        """Test evaluator with DataLoader."""
        evaluator = TokenizerEvaluator(continuous_model, device="cpu")

        # Create dataset and dataloader
        images = torch.randn(32, 1, 32, 32, 32)
        dataset = SimpleDataset(images)
        loader = DataLoader(dataset, batch_size=8)

        # Evaluate
        results = evaluator.evaluate(loader, num_samples=16)

        assert results["num_samples"] == 16
        assert results["avg_metrics"].mse is not None
        assert results["avg_metrics"].psnr is not None

    def test_save_load_results(self, continuous_model, tmp_path):
        """Test saving and loading results."""
        evaluator = TokenizerEvaluator(continuous_model, device="cpu")

        # Create and evaluate data
        images = torch.randn(16, 1, 32, 32, 32)
        dataset = SimpleDataset(images)
        loader = DataLoader(dataset, batch_size=8)
        results = evaluator.evaluate(loader)

        # Save results
        save_path = tmp_path / "results.json"
        TokenizerEvaluator.save_results(results, save_path)

        # Load results
        loaded_results = TokenizerEvaluator.load_results(save_path)

        assert loaded_results["num_samples"] == results["num_samples"]
        assert loaded_results["avg_metrics"].mse == pytest.approx(
            results["avg_metrics"].mse, rel=1e-5
        )


class TestSimpleDataset:
    """Test SimpleDataset class."""

    def test_dataset_torch(self):
        """Test dataset with torch tensors."""
        images = torch.randn(100, 1, 32, 32, 32)
        dataset = SimpleDataset(images)

        assert len(dataset) == 100
        sample = dataset[0]
        assert sample.shape == (1, 32, 32, 32)

    def test_dataset_numpy(self):
        """Test dataset with numpy arrays."""
        images = np.random.randn(100, 1, 32, 32, 32).astype(np.float32)
        dataset = SimpleDataset(images)

        assert len(dataset) == 100
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)

    def test_dataset_with_masks(self):
        """Test dataset with masks."""
        images = torch.randn(100, 1, 32, 32, 32)
        masks = torch.ones(100, 1, 32, 32, 32)
        dataset = SimpleDataset(images, masks)

        assert len(dataset) == 100
        sample_img, sample_mask = dataset[0]
        assert sample_img.shape == (1, 32, 32, 32)
        assert sample_mask.shape == (1, 32, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
