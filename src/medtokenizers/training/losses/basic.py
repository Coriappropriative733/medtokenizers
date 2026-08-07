"""Basic loss functions for tokenizer training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float

from ...modules.utils import jaxtyped_compile_safe
from .compound import _GANLossBase
from .perceptual import VGGFeatureLoss


class Reconstruction(nn.Module):
    """Reconstruction loss (L1, L2, or smooth L1)."""

    def __init__(self, loss_type: str = "l1", reduction: str = "mean"):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        reconstruction: Float[torch.Tensor, "batch channels ..."],
        target: Float[torch.Tensor, "batch channels ..."],
    ) -> Float[torch.Tensor, ""]:
        if self.loss_type == "l1":
            return F.l1_loss(reconstruction, target, reduction=self.reduction)
        elif self.loss_type == "l2":
            return F.mse_loss(reconstruction, target, reduction=self.reduction)
        return F.smooth_l1_loss(reconstruction, target, reduction=self.reduction)


class Combined(nn.Module):
    """Weighted combination of reconstruction, perceptual, and quantization losses.

    The perceptual term is a raw VGG16 feature-space L1 loss (VGGFeatureLoss),
    not the calibrated LPIPS metric (see evaluation.compute_lpips for that).
    When perceptual_weight <= 0, the VGG feature network is not loaded (saves
    memory) and no perceptual term is computed.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        quantization_weight: float = 1.0,
        reconstruction_type: str = "l1",
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.quantization_weight = quantization_weight

        self.reconstruction = Reconstruction(loss_type=reconstruction_type)

        self.has_perceptual = perceptual_weight > 0
        if self.has_perceptual:
            self.perceptual = VGGFeatureLoss(weight=1.0)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        reconstruction: Float[torch.Tensor, "batch channels ..."],
        target: Float[torch.Tensor, "batch channels ..."],
        quant_loss: Optional[Float[torch.Tensor, "..."]] = None,
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        recon_loss = self.reconstruction(reconstruction, target)
        total_loss = self.reconstruction_weight * recon_loss

        loss_dict = {
            "reconstruction": recon_loss.item(),
        }

        if self.has_perceptual:
            perceptual_loss = self.perceptual(reconstruction, target)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            loss_dict["perceptual"] = perceptual_loss.item()

        if quant_loss is not None:
            quant_loss_value = quant_loss.mean()
            total_loss += self.quantization_weight * quant_loss_value
            loss_dict["quantization"] = quant_loss_value.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


class CombinedPerceptual(_GANLossBase):
    """Combined L1 + perceptual + Gram style loss for high-quality reconstruction.

    Supports two perceptual loss modes:
    - VGGFeatureLoss (raw VGG16 feature-space L1, not calibrated LPIPS): Slower
      but captures semantic similarity. For the real LPIPS metric, see
      evaluation.compute_lpips.
    - SSIM3D: Much faster, captures structural similarity (better for medical imaging)

    The discriminator setup and discriminator step are shared with the compound
    losses via :class:`_GANLossBase`.
    """

    def __init__(
        self,
        dim: int = 3,
        input_channels: int = 1,
        reconstruction_weight: float = 1.0,
        vgg_weight: float = 0.0,
        gram_weight: float = 0.0,
        quantization_weight: float = 0.0,
        reconstruction_type: str = "l1",
        lpips_slice_stride: int = 4,
        ssim_weight: float = 0.0,
        use_ssim_instead_of_lpips: bool = False,
        adversarial_weight: float = 0.0,
        discriminator_start_iter: int = 5000,
        discriminator_config: Optional[dict] = None,
        use_lecam: bool = True,
        lecam_weight: float = 0.001,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.vgg_weight = vgg_weight
        self.gram_weight = gram_weight
        self.quantization_weight = quantization_weight
        self.ssim_weight = ssim_weight
        self.use_ssim_instead_of_lpips = use_ssim_instead_of_lpips
        self.adversarial_weight = adversarial_weight
        self.discriminator_start_iter = discriminator_start_iter
        self.iteration = 0

        self.reconstruction = Reconstruction(loss_type=reconstruction_type)

        from .perceptual import SSIM3D, GramLoss, VGGFeatureLoss

        # Only instantiate the VGG feature loss if we're actually using it
        # (saves memory). NOTE: this is raw VGG16 feature L1, not calibrated
        # LPIPS (see evaluation.compute_lpips for the calibrated metric).
        if not use_ssim_instead_of_lpips:
            self.vgg_feature = VGGFeatureLoss(
                dim=dim, weight=1.0, slice_stride=lpips_slice_stride
            )
        else:
            self.vgg_feature = None
        self.gram = GramLoss(dim=dim)
        self.ssim = SSIM3D() if (ssim_weight > 0 or use_ssim_instead_of_lpips) else None

        # Discriminator (optional — skipped when adversarial_weight <= 0)
        self._init_discriminator(
            dim=dim,
            input_channels=input_channels,
            adversarial_weight=adversarial_weight,
            discriminator_config=discriminator_config,
            use_lecam=use_lecam,
            lecam_weight=lecam_weight,
        )

    def set_weights(
        self, vgg_weight: float, gram_weight: float, ssim_weight: float | None = None
    ) -> None:
        self.vgg_weight = vgg_weight
        self.gram_weight = gram_weight
        if ssim_weight is not None:
            self.ssim_weight = ssim_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        quant_loss: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = self.reconstruction(reconstruction, target)
        total_loss = self.reconstruction_weight * recon_loss

        loss_dict = {
            "reconstruction": recon_loss.item(),
        }

        # SSIM loss (fast, good for medical imaging)
        if self.use_ssim_instead_of_lpips and self.vgg_weight > 0:
            # Use SSIM in place of LPIPS when use_ssim_instead_of_lpips=True
            ssim_loss = self.ssim(reconstruction, target)
            total_loss = total_loss + self.vgg_weight * ssim_loss
            loss_dict["ssim"] = ssim_loss.item()
        elif self.vgg_weight > 0 and self.vgg_feature is not None:
            # Raw VGG16 feature-space L1 (slower)
            vgg_feature_loss = self.vgg_feature(reconstruction, target)
            total_loss = total_loss + self.vgg_weight * vgg_feature_loss
            loss_dict["vgg_feature"] = vgg_feature_loss.item()

        # Additional standalone SSIM if ssim_weight > 0
        if (
            self.ssim_weight > 0
            and self.ssim is not None
            and not self.use_ssim_instead_of_lpips
        ):
            ssim_loss = self.ssim(reconstruction, target)
            total_loss = total_loss + self.ssim_weight * ssim_loss
            loss_dict["ssim"] = ssim_loss.item()

        if self.gram_weight > 0:
            gram_loss = self.gram(reconstruction, target)
            total_loss = total_loss + self.gram_weight * gram_loss
            loss_dict["gram_style"] = gram_loss.item()

        if quant_loss is not None and self.quantization_weight > 0:
            quant_loss_value = quant_loss.mean()
            total_loss = total_loss + self.quantization_weight * quant_loss_value
            loss_dict["quantization"] = quant_loss_value.item()

        # Adversarial (generator) loss
        self.iteration += 1
        if self.has_discriminator and self.iteration >= self.discriminator_start_iter:
            fake_logits = self.discriminator(reconstruction)
            adv_loss = self.adversarial_loss.generator_loss(fake_logits)
            total_loss = total_loss + self.adversarial_weight * adv_loss
            loss_dict["adversarial_loss"] = adv_loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
