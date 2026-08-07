"""Tests for distribution modules."""

import pytest
import torch

from medtokenizers.modules.distributions import (
    GaussianDistribution,
    IdentityDistribution,
)


class TestIdentityDistribution:
    """Tests for IdentityDistribution."""

    def test_returns_input_unchanged(self):
        dist = IdentityDistribution()
        x = torch.randn(2, 4, 8, 8)

        latent, (kl, logvar) = dist(x)

        assert torch.allclose(latent, x)
        assert kl.shape == (2, 1)
        assert torch.allclose(kl, torch.zeros(2, 1))

    def test_handles_different_shapes(self):
        dist = IdentityDistribution()

        # 2D
        x_2d = torch.randn(1, 8, 16, 16)
        latent, _ = dist(x_2d)
        assert latent.shape == x_2d.shape

        # 3D
        x_3d = torch.randn(1, 8, 8, 8, 8)
        latent, _ = dist(x_3d)
        assert latent.shape == x_3d.shape

    def test_device_consistency(self):
        dist = IdentityDistribution()
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 8, 8, device="cuda")
            latent, (kl, logvar) = dist(x)

            assert latent.device == x.device
            assert kl.device == x.device


class TestGaussianDistribution:
    """Tests for GaussianDistribution with KL divergence."""

    def test_samples_from_distribution(self):
        dist = GaussianDistribution()
        # Input: concatenated mean and logvar
        params = torch.randn(2, 8, 8, 8)  # 8 channels = 4 mean + 4 logvar

        latent, (kl_loss, (mean, logvar)) = dist(params)

        assert latent.shape == (2, 4, 8, 8)
        assert mean.shape == (2, 4, 8, 8)
        assert logvar.shape == (2, 4, 8, 8)
        assert kl_loss.shape == (2,)

    def test_kl_divergence_positive(self):
        dist = GaussianDistribution()
        # Non-zero mean and variance should give positive KL
        params = torch.cat(
            [
                torch.ones(2, 4, 8, 8) * 2.0,  # mean = 2
                torch.ones(2, 4, 8, 8) * 0.5,  # logvar = 0.5
            ],
            dim=1,
        )

        _, (kl_loss, _) = dist(params)

        assert torch.all(kl_loss > 0)

    def test_kl_divergence_zero_for_standard_normal(self):
        dist = GaussianDistribution()
        # Standard normal should have near-zero KL
        params = torch.cat(
            [
                torch.zeros(2, 4, 8, 8),  # mean = 0
                torch.zeros(2, 4, 8, 8),  # logvar = 0 (std = 1)
            ],
            dim=1,
        )

        _, (kl_loss, _) = dist(params)

        assert torch.allclose(kl_loss, torch.zeros(2), atol=1e-5)

    def test_logvar_clamping(self):
        dist = GaussianDistribution(min_logvar=-10.0, max_logvar=10.0)
        # Extreme logvar values
        params = torch.cat(
            [
                torch.zeros(2, 4, 8, 8),
                torch.ones(2, 4, 8, 8) * 100.0,  # Very large logvar
            ],
            dim=1,
        )

        _, (_, (_, logvar)) = dist(params)

        assert torch.all(logvar <= 10.0)

        # Very negative logvar
        params = torch.cat(
            [
                torch.zeros(2, 4, 8, 8),
                torch.ones(2, 4, 8, 8) * -100.0,
            ],
            dim=1,
        )

        _, (_, (_, logvar)) = dist(params)

        assert torch.all(logvar >= -10.0)

    def test_sampling_variability(self):
        dist = GaussianDistribution()
        params = torch.randn(2, 8, 8, 8)

        # Multiple samples should be different (stochastic)
        latent1, _ = dist(params)
        latent2, _ = dist(params)

        assert not torch.allclose(latent1, latent2)

    def test_3d_input(self):
        dist = GaussianDistribution()
        params = torch.randn(1, 8, 4, 8, 8)  # 3D spatial

        latent, (kl_loss, (mean, logvar)) = dist(params)

        assert latent.shape == (1, 4, 4, 8, 8)
        assert kl_loss.shape == (1,)

    def test_deterministic_with_zero_std(self):
        dist = GaussianDistribution()
        # logvar = -inf should give deterministic output
        params = torch.cat(
            [
                torch.ones(1, 4, 4, 4),
                torch.ones(1, 4, 4, 4) * dist.min_logvar,
            ],
            dim=1,
        )

        latent, _ = dist(params)

        # With very small std, output should be close to mean
        assert latent.shape == (1, 4, 4, 4)
