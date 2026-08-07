"""Compound loss functions combining multiple objectives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from ..discriminators import MultiScale
from .adversarial import Adversarial, LeCAM, R1Penalty
from .perceptual import VGGFeatureLoss


class _GANLossBase(nn.Module):
    """Shared scaffolding for adversarial reconstruction losses.

    Holds the discriminator/perceptual setup, the reconstruction loss helper,
    and the default discriminator step shared by :class:`VQGANLoss` and
    :class:`VAEGANLoss`. Subclasses provide their own ``generator_step`` and may
    override ``discriminator_step`` to add extra penalties (e.g. R1).
    """

    def _init_discriminator(
        self,
        dim: int,
        input_channels: int,
        adversarial_weight: float,
        discriminator_config: Optional[dict],
        use_lecam: bool,
        lecam_weight: float,
    ) -> None:
        """Instantiate the discriminator and its auxiliary losses.

        When ``adversarial_weight <= 0`` the discriminator is skipped entirely
        (saves memory) and all adversarial sub-losses are disabled.
        """
        self.has_discriminator = adversarial_weight > 0
        if self.has_discriminator:
            disc_config = discriminator_config or {}
            self.discriminator = MultiScale(
                dim=dim, input_channels=input_channels, **disc_config
            )
            self.adversarial_loss = Adversarial(loss_type="hinge")

            self.use_lecam = use_lecam
            if use_lecam:
                self.lecam_loss = LeCAM(weight=lecam_weight)
        else:
            self.discriminator = None
            self.use_lecam = False

    def _init_perceptual(
        self,
        dim: int,
        perceptual_weight: float,
        lpips_slice_stride: int,
    ) -> None:
        """Instantiate the perceptual loss network when its weight is positive.

        When ``perceptual_weight <= 0`` the VGG feature network is not loaded
        (saves memory).
        """
        self.has_perceptual = perceptual_weight > 0
        if self.has_perceptual:
            self.perceptual_loss = VGGFeatureLoss(
                dim=dim,
                weight=perceptual_weight,
                slice_stride=lpips_slice_stride,
            )

    def _recon_loss(
        self,
        reconstructions: torch.Tensor,
        real_images: torch.Tensor,
    ) -> torch.Tensor:
        if self.reconstruction_type == "l2":
            return F.mse_loss(reconstructions, real_images)
        return F.l1_loss(reconstructions, real_images)

    def discriminator_step(
        self,
        real_images: Float[torch.Tensor, "batch channels ..."],
        reconstructions: Float[torch.Tensor, "batch channels ..."],
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        if not self.has_discriminator:
            return torch.tensor(0.0, device=real_images.device), {}

        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(reconstructions.detach())

        disc_loss = self.adversarial_loss.discriminator_loss(real_logits, fake_logits)

        loss_dict = {"discriminator_loss": disc_loss.item()}

        if self.use_lecam:
            lecam_loss = self.lecam_loss(real_logits, fake_logits)
            disc_loss = disc_loss + lecam_loss
            loss_dict["lecam_loss"] = lecam_loss.item()

        return disc_loss, loss_dict


class VQGANLoss(_GANLossBase):
    """VQGAN-style loss combining reconstruction, adversarial, and perceptual objectives.

    Loss = recon_weight * recon_loss + quant_weight * quant_loss
           + perceptual_weight * VGGFeatureLoss (if > 0)
           + adversarial_weight * GAN (if > 0, after discriminator_start_iter)

    The perceptual term is a raw VGG16 feature-space L1 loss (VGGFeatureLoss),
    not the calibrated LPIPS metric (see evaluation.compute_lpips for that).

    When adversarial_weight=0, discriminator is not instantiated (saves memory).
    When perceptual_weight=0, the VGG feature network is not loaded (saves memory).
    """

    def __init__(
        self,
        dim: int = 2,
        input_channels: int = 1,
        reconstruction_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        perceptual_weight: float = 1.0,
        quantization_weight: float = 1.0,
        use_r1_penalty: bool = False,
        r1_weight: float = 10.0,
        r1_penalty_interval: int = 16,
        discriminator_start_iter: int = 5000,
        discriminator_config: Optional[dict] = None,
        lpips_slice_stride: int = 1,
        reconstruction_type: str = "l1",
        use_lecam: bool = True,
        lecam_weight: float = 0.001,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.quantization_weight = quantization_weight
        self.discriminator_start_iter = discriminator_start_iter
        self.r1_penalty_interval = r1_penalty_interval
        self.reconstruction_type = reconstruction_type
        self.iteration = 0

        # Discriminator (optional — skipped when adversarial_weight <= 0)
        self._init_discriminator(
            dim=dim,
            input_channels=input_channels,
            adversarial_weight=adversarial_weight,
            discriminator_config=discriminator_config,
            use_lecam=use_lecam,
            lecam_weight=lecam_weight,
        )
        if self.has_discriminator:
            self.use_r1_penalty = use_r1_penalty
            if use_r1_penalty:
                self.r1_penalty = R1Penalty(weight=r1_weight)
        else:
            self.use_r1_penalty = False

        # Perceptual loss (optional — skipped when perceptual_weight <= 0)
        self._init_perceptual(
            dim=dim,
            perceptual_weight=perceptual_weight,
            lpips_slice_stride=lpips_slice_stride,
        )

    def discriminator_step(
        self,
        real_images: Float[torch.Tensor, "batch channels ..."],
        reconstructions: Float[torch.Tensor, "batch channels ..."],
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        if not self.has_discriminator:
            return torch.tensor(0.0, device=real_images.device), {}

        if self.use_r1_penalty:
            real_images.requires_grad_(True)

        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(reconstructions.detach())

        disc_loss = self.adversarial_loss.discriminator_loss(real_logits, fake_logits)

        loss_dict = {"discriminator_loss": disc_loss.item()}

        if self.use_r1_penalty and self.iteration % self.r1_penalty_interval == 0:
            r1_loss = self.r1_penalty(real_images, real_logits)
            disc_loss = disc_loss + r1_loss
            loss_dict["r1_penalty"] = r1_loss.item()

        if self.use_lecam:
            lecam_loss = self.lecam_loss(real_logits, fake_logits)
            disc_loss = disc_loss + lecam_loss
            loss_dict["lecam_loss"] = lecam_loss.item()

        return disc_loss, loss_dict

    def generator_step(
        self,
        real_images: Float[torch.Tensor, "batch channels ..."],
        reconstructions: Float[torch.Tensor, "batch channels ..."],
        quant_loss: Optional[Float[torch.Tensor, "..."]] = None,
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        self.iteration += 1

        recon_loss = self._recon_loss(reconstructions, real_images)
        total_loss = self.reconstruction_weight * recon_loss

        loss_dict = {"reconstruction_loss": recon_loss.item()}

        if self.has_perceptual:
            perceptual_loss = self.perceptual_loss(reconstructions, real_images)
            total_loss += perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss.item()

        if quant_loss is not None:
            quant_loss_value = quant_loss.mean()
            total_loss += self.quantization_weight * quant_loss_value
            loss_dict["quantization_loss"] = quant_loss_value.item()

        if self.has_discriminator and self.iteration >= self.discriminator_start_iter:
            fake_logits = self.discriminator(reconstructions)
            adv_loss = self.adversarial_loss.generator_loss(fake_logits)
            total_loss += self.adversarial_weight * adv_loss
            loss_dict["adversarial_loss"] = adv_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


class VAEGANLoss(_GANLossBase):
    """VAE-GAN loss for continuous latent space with KL divergence.

    Loss = recon_weight * L1 + kl_weight(epoch) * KL
           + perceptual_weight * VGGFeatureLoss (if > 0)
           + adversarial_weight * GAN (if > 0, after discriminator_start_iter)

    KL annealing: when kl_warmup_epochs > 0, linearly ramps kl_weight from 0
    to the target value over the first N epochs. This prevents posterior collapse
    early in training and is standard best practice for VAE training.

    The perceptual term is a raw VGG16 feature-space L1 loss (VGGFeatureLoss),
    not the calibrated LPIPS metric (see evaluation.compute_lpips for that).

    When adversarial_weight=0, discriminator is not instantiated (saves memory).
    When perceptual_weight=0, the VGG feature network is not loaded (saves memory).
    """

    def __init__(
        self,
        dim: int = 2,
        input_channels: int = 1,
        reconstruction_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        perceptual_weight: float = 1.0,
        kl_weight: float = 1e-6,
        discriminator_start_iter: int = 5000,
        discriminator_config: Optional[dict] = None,
        target_std_min: float = 0.9,
        target_std_max: float = 1.1,
        lpips_slice_stride: int = 1,
        kl_warmup_epochs: int = 0,
        reconstruction_type: str = "l1",
        use_lecam: bool = True,
        lecam_weight: float = 0.001,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self._kl_weight_target = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.discriminator_start_iter = discriminator_start_iter
        self.target_std_min = target_std_min
        self.target_std_max = target_std_max
        self.reconstruction_type = reconstruction_type
        self.iteration = 0

        # Discriminator (optional — skipped when adversarial_weight <= 0)
        self._init_discriminator(
            dim=dim,
            input_channels=input_channels,
            adversarial_weight=adversarial_weight,
            discriminator_config=discriminator_config,
            use_lecam=use_lecam,
            lecam_weight=lecam_weight,
        )

        # Perceptual loss (optional — skipped when perceptual_weight <= 0)
        self._init_perceptual(
            dim=dim,
            perceptual_weight=perceptual_weight,
            lpips_slice_stride=lpips_slice_stride,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update KL weight based on current epoch (linear warmup annealing).

        If kl_warmup_epochs > 0, linearly ramps kl_weight from 0 to the target
        over the first kl_warmup_epochs. After warmup, kl_weight stays at target.
        """
        if self.kl_warmup_epochs > 0 and epoch < self.kl_warmup_epochs:
            progress = (epoch + 1) / self.kl_warmup_epochs
            self.kl_weight = self._kl_weight_target * progress
        else:
            self.kl_weight = self._kl_weight_target

    def generator_step(
        self,
        real_images: Float[torch.Tensor, "batch channels ..."],
        reconstructions: Float[torch.Tensor, "batch channels ..."],
        kl_loss: Optional[Float[torch.Tensor, "batch"]] = None,
        posteriors: Optional[tuple] = None,
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        self.iteration += 1

        recon_loss = self._recon_loss(reconstructions, real_images)
        total_loss = self.reconstruction_weight * recon_loss

        loss_dict = {"reconstruction_loss": recon_loss.item()}

        if self.has_perceptual:
            perceptual_loss = self.perceptual_loss(reconstructions, real_images)
            total_loss += perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss.item()

        if kl_loss is not None:
            kl_loss_value = kl_loss.mean()
            total_loss += self.kl_weight * kl_loss_value
            loss_dict["kl_loss"] = kl_loss_value.item()
            loss_dict["kl_weight"] = self.kl_weight

            if posteriors is not None:
                mean, logvar = posteriors
                std = torch.exp(0.5 * logvar).mean().item()
                loss_dict["latent_std"] = std

        if self.has_discriminator and self.iteration >= self.discriminator_start_iter:
            fake_logits = self.discriminator(reconstructions)
            adv_loss = self.adversarial_loss.generator_loss(fake_logits)
            total_loss += self.adversarial_weight * adv_loss
            loss_dict["adversarial_loss"] = adv_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
