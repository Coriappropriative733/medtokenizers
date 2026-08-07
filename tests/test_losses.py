"""Tests for loss functions."""

import pytest
import torch
import torch.nn as nn
from conftest import requires_network

from medtokenizers.training.losses import (
    Adversarial,
    Combined,
    CombinedPerceptual,
    R1Penalty,
    Reconstruction,
    VAEGANLoss,
    VGGFeatureLoss,
    VQGANLoss,
)


class TestReconstruction:
    """Tests for Reconstruction loss."""

    def test_l1_loss(self):
        loss_fn = Reconstruction(loss_type="l1")
        recon = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)

        loss = loss_fn(recon, target)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_l2_loss(self):
        loss_fn = Reconstruction(loss_type="l2")
        recon = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)

        loss = loss_fn(recon, target)

        assert loss >= 0

    def test_zero_loss_for_perfect_reconstruction(self):
        loss_fn = Reconstruction(loss_type="l1")
        x = torch.randn(2, 1, 16, 16)

        loss = loss_fn(x, x)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_3d_input(self):
        loss_fn = Reconstruction(loss_type="l1")
        recon = torch.randn(1, 1, 8, 16, 16)
        target = torch.randn(1, 1, 8, 16, 16)

        loss = loss_fn(recon, target)

        assert loss.ndim == 0


class TestAdversarial:
    """Tests for Adversarial loss."""

    @pytest.fixture
    def logits(self):
        return {
            "real": [torch.randn(2, 1, 8, 8) + 1.0 for _ in range(3)],
            "fake": [torch.randn(2, 1, 8, 8) - 1.0 for _ in range(3)],
        }

    def test_hinge_discriminator_loss(self, logits):
        loss_fn = Adversarial(loss_type="hinge")

        disc_loss = loss_fn.discriminator_loss(logits["real"], logits["fake"])

        assert disc_loss >= 0
        assert disc_loss.ndim == 0

    def test_hinge_generator_loss(self, logits):
        loss_fn = Adversarial(loss_type="hinge")

        gen_loss = loss_fn.generator_loss(logits["fake"])

        assert gen_loss.ndim == 0

    def test_non_saturating_loss(self, logits):
        loss_fn = Adversarial(loss_type="non_saturating")

        disc_loss = loss_fn.discriminator_loss(logits["real"], logits["fake"])
        gen_loss = loss_fn.generator_loss(logits["fake"])

        assert disc_loss >= 0
        assert gen_loss.ndim == 0

    def test_vanilla_loss(self, logits):
        loss_fn = Adversarial(loss_type="vanilla")

        disc_loss = loss_fn.discriminator_loss(logits["real"], logits["fake"])
        gen_loss = loss_fn.generator_loss(logits["fake"])

        assert disc_loss >= 0
        assert gen_loss >= 0

    def test_single_scale(self):
        loss_fn = Adversarial(loss_type="hinge")
        real = [torch.randn(2, 1, 8, 8)]
        fake = [torch.randn(2, 1, 8, 8)]

        disc_loss = loss_fn.discriminator_loss(real, fake)

        assert disc_loss.ndim == 0


class TestR1Penalty:
    """Tests for R1 gradient penalty."""

    def test_computes_penalty(self):
        penalty = R1Penalty(weight=10.0)

        real_images = torch.randn(2, 1, 16, 16, requires_grad=True)

        # Create logits that depend on real_images
        conv = nn.Conv2d(1, 1, 3, padding=1)
        real_logits = [conv(real_images) for _ in range(2)]

        loss = penalty(real_images, real_logits)

        assert loss >= 0
        assert loss.ndim == 0

    def test_zero_penalty_for_zero_gradients(self):
        penalty = R1Penalty(weight=10.0)

        real_images = torch.zeros(2, 1, 16, 16, requires_grad=True)

        # Create constant logits (no gradient)
        with torch.no_grad():
            conv = nn.Conv2d(1, 1, 3, padding=1)
            for param in conv.parameters():
                param.fill_(0.0)

        real_logits = [conv(real_images) for _ in range(2)]

        loss = penalty(real_images, real_logits)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-4)


@requires_network
class TestVGGFeatureLoss:
    """Tests for VGGFeatureLoss perceptual loss (requires VGG weights download)."""

    def test_2d_input(self):
        loss_fn = VGGFeatureLoss(dim=2, weight=1.0)
        recon = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)

        loss = loss_fn(recon, target)

        assert loss >= 0
        assert loss.ndim == 0

    def test_3d_input(self):
        loss_fn = VGGFeatureLoss(dim=3, weight=1.0)
        recon = torch.randn(1, 1, 4, 64, 64)  # Small depth for speed
        target = torch.randn(1, 1, 4, 64, 64)

        loss = loss_fn(recon, target)

        assert loss >= 0
        assert loss.ndim == 0

    def test_rgb_input(self):
        loss_fn = VGGFeatureLoss(dim=2, weight=1.0)
        recon = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 64, 64)

        loss = loss_fn(recon, target)

        assert loss >= 0

    def test_zero_loss_for_identical_images(self):
        loss_fn = VGGFeatureLoss(dim=2, weight=1.0)
        x = torch.randn(1, 1, 64, 64)

        loss = loss_fn(x, x)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


@requires_network
class TestCombinedPerceptual:
    """Tests for CombinedPerceptual loss (requires VGG weights download)."""

    def test_vgg_feature_key_in_loss_dict(self):
        loss_fn = CombinedPerceptual(
            dim=2,
            input_channels=1,
            vgg_weight=0.5,
            lpips_slice_stride=1,
        )
        recon = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)

        loss, loss_dict = loss_fn(recon, target)

        assert loss >= 0
        assert "reconstruction" in loss_dict
        assert "vgg_feature" in loss_dict
        assert "lpips" not in loss_dict
        assert "total" in loss_dict


class TestCombined:
    """Tests for Combined loss."""

    def test_default_perceptual_weight_is_zero(self):
        """Default weight is 0 so VGG is not loaded and no perceptual term is added."""
        loss_fn = Combined()
        assert loss_fn.perceptual_weight == 0.0
        assert loss_fn.has_perceptual is False

    def test_without_quantization_no_perceptual(self):
        """With perceptual_weight=0 (offline), no perceptual term is computed."""
        loss_fn = Combined(
            reconstruction_weight=1.0,
            perceptual_weight=0.0,
            quantization_weight=1.0,
        )
        recon = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)

        loss, loss_dict = loss_fn(recon, target)

        assert loss >= 0
        assert "reconstruction" in loss_dict
        assert "perceptual" not in loss_dict
        assert "total" in loss_dict
        assert "quantization" not in loss_dict

    @requires_network
    def test_with_perceptual(self):
        """With perceptual_weight>0, VGG is loaded and a perceptual term is added."""
        loss_fn = Combined(
            reconstruction_weight=1.0,
            perceptual_weight=0.1,
            quantization_weight=1.0,
        )
        assert loss_fn.has_perceptual is True

        recon = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)

        loss, loss_dict = loss_fn(recon, target)

        assert loss >= 0
        assert "reconstruction" in loss_dict
        assert "perceptual" in loss_dict
        assert "total" in loss_dict

    def test_with_quantization(self):
        loss_fn = Combined()
        recon = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        quant_loss = torch.rand(2)

        loss, loss_dict = loss_fn(recon, target, quant_loss)

        assert "quantization" in loss_dict


@requires_network
class TestVQGANLoss:
    """Tests for VQGANLoss (requires VGG weights download)."""

    def test_generator_step_before_warmup(self):
        loss_fn = VQGANLoss(
            dim=2,
            input_channels=1,
            discriminator_start_iter=1000,
        )

        real = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)

        loss, loss_dict = loss_fn.generator_step(real, recon)

        assert loss >= 0
        assert "reconstruction_loss" in loss_dict
        assert "adversarial_loss" not in loss_dict  # Before warmup

    def test_generator_step_after_warmup(self):
        loss_fn = VQGANLoss(
            dim=2,
            input_channels=1,
            discriminator_start_iter=0,
            discriminator_config={"num_scales": 2, "ndf": 16},
        )

        real = torch.randn(1, 1, 64, 64)
        recon = torch.randn(1, 1, 64, 64)

        loss, loss_dict = loss_fn.generator_step(real, recon)

        assert "adversarial_loss" in loss_dict  # After warmup

    def test_discriminator_step(self):
        loss_fn = VQGANLoss(
            dim=2, input_channels=1, discriminator_config={"num_scales": 2, "ndf": 16}
        )

        real = torch.randn(1, 1, 64, 64)
        recon = torch.randn(1, 1, 64, 64)

        loss, loss_dict = loss_fn.discriminator_step(real, recon)

        assert loss >= 0
        assert "discriminator_loss" in loss_dict


@requires_network
class TestVAEGANLoss:
    """Tests for VAEGANLoss (requires VGG weights download)."""

    def test_generator_step_with_kl(self):
        loss_fn = VAEGANLoss(
            dim=2,
            input_channels=1,
            discriminator_start_iter=1000,
        )

        real = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)
        kl_loss = torch.rand(1)
        mean = torch.randn(1, 4, 8, 8)
        logvar = torch.randn(1, 4, 8, 8)

        loss, loss_dict = loss_fn.generator_step(real, recon, kl_loss, (mean, logvar))

        assert loss >= 0
        assert "kl_loss" in loss_dict
        assert "latent_std" in loss_dict
