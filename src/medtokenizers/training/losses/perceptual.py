"""Perceptual loss functions."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float
from torchvision import models

from ...modules.utils import jaxtyped_compile_safe


def _gram_matrix_2d(input: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for 2D inputs.

    Args:
        input: Input tensor of shape (batch, channels, height, width)

    Returns:
        Gram matrix of shape (batch * channels, batch * channels)
    """
    batch, channels, height, width = input.shape
    features = input.view(batch * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * height * width)


def _gram_matrix_3d(input: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for 3D inputs.

    Args:
        input: Input tensor of shape (batch, channels, depth, height, width)

    Returns:
        Gram matrix of shape (batch * channels, batch * channels)
    """
    batch, channels, depth, height, width = input.shape
    features = input.view(batch * channels, depth * height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * depth * height * width)


class GramLoss(nn.Module):
    """Gram matrix loss for style transfer and texture synthesis.

    Computes the L1 loss between Gram matrices of input and target tensors.
    Useful for capturing texture and style information in reconstructions.
    """

    def __init__(self, dim: int = 2, reduction: str = "mean"):
        """Initialize Gram loss.

        Args:
            dim: Spatial dimensionality (2 for 2D, 3 for 3D)
            reduction: Reduction method for loss ('mean' or 'sum')
        """
        super().__init__()
        if dim not in [2, 3]:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2 or 3.")
        self.dim = dim
        self.reduction = reduction

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        input: Float[torch.Tensor, "batch channels ..."],
        target: Float[torch.Tensor, "batch channels ..."],
    ) -> Float[torch.Tensor, ""]:
        """Compute Gram matrix loss.

        Args:
            input: Reconstructed tensor
            target: Target tensor

        Returns:
            Scalar loss value
        """
        if self.dim == 2:
            gram_input = _gram_matrix_2d(input)
            gram_target = _gram_matrix_2d(target)
        else:
            gram_input = _gram_matrix_3d(input)
            gram_target = _gram_matrix_3d(target)

        return F.l1_loss(gram_input, gram_target, reduction=self.reduction)


class VGGFeatureLoss(nn.Module):
    """Raw VGG16 feature-space L1 loss (NOT true LPIPS).

    This computes the L1 distance between intermediate VGG16 feature maps of
    the reconstruction and target. Despite the historical name, it is **not**
    Learned Perceptual Image Patch Similarity: it uses raw VGG features with no
    learned linear calibration weights on top of them, so the resulting values
    are not on the calibrated LPIPS scale.

    For the real, calibrated LPIPS metric, use
    ``medtokenizers.evaluation.compute_lpips`` (backed by the ``lpips`` package).

    For 3D inputs, applies VGG slice-wise with optional stride to reduce compute.
    """

    def __init__(
        self,
        dim: int = 2,
        weight: float = 1.0,
        layers: Optional[list[int]] = None,
        slice_stride: int = 1,
    ):
        if layers is None:
            layers = [3, 8, 15, 22]
        super().__init__()
        self.weight = weight
        self.dim = dim
        self.layers = layers
        self.slice_stride = max(1, slice_stride)

        vgg_features = models.vgg16(weights="IMAGENET1K_V1").features
        self.feature_blocks = nn.ModuleList()

        prev_layer = 0
        for layer_idx in layers:
            block = nn.Sequential(
                *list(vgg_features.children())[prev_layer : layer_idx + 1]
            )
            self.feature_blocks.append(block)
            prev_layer = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from VGG."""
        features = []
        for block in self.feature_blocks:
            x = block(x)
            features.append(x)
        return features

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        reconstruction: Float[torch.Tensor, "batch channels ..."],
        target: Float[torch.Tensor, "batch channels ..."],
    ) -> Float[torch.Tensor, ""]:
        if self.dim == 3:
            batch, channels, depth, height, width = reconstruction.shape

            loss: torch.Tensor = torch.tensor(0.0, device=reconstruction.device)
            slice_count = 0
            for d in range(0, depth, self.slice_stride):
                recon_slice = reconstruction[:, :, d, :, :]
                target_slice = target[:, :, d, :, :]

                if channels == 1:
                    recon_slice = recon_slice.repeat(1, 3, 1, 1)
                    target_slice = target_slice.repeat(1, 3, 1, 1)

                with torch.no_grad():
                    target_features = self._extract_features(target_slice)

                recon_features = self._extract_features(recon_slice)

                for tf, rf in zip(
                    target_features,
                    recon_features,
                ):
                    loss = loss + F.l1_loss(rf, tf)

                slice_count += 1

            slice_count = max(1, slice_count)
            return self.weight * loss / slice_count
        else:
            if reconstruction.shape[1] == 1:
                reconstruction = reconstruction.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)

            with torch.no_grad():
                target_features = self._extract_features(target)

            recon_features = self._extract_features(reconstruction)

            loss: torch.Tensor = torch.tensor(0.0, device=reconstruction.device)
            for tf, rf in zip(
                target_features,
                recon_features,
            ):
                loss = loss + F.l1_loss(rf, tf)

            return self.weight * loss


class SSIM3D(nn.Module):
    """3D SSIM - single pass over entire volume."""

    def __init__(
        self,
        window_size: int = 5,
        sigma: float = 1.0,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad = window_size // 2

        coords = torch.arange(window_size).float() - window_size // 2
        kernel_1d = torch.exp(-coords.pow(2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_3d = (
            kernel_1d[:, None, None]
            * kernel_1d[None, :, None]
            * kernel_1d[None, None, :]
        )
        self.register_buffer("kernel", kernel_3d[None, None])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        kernel = self.kernel.expand(c, -1, -1, -1, -1).to(x.dtype)

        mu_x = F.conv3d(x, kernel, padding=self.pad, groups=c)
        mu_y = F.conv3d(y, kernel, padding=self.pad, groups=c)

        sigma_x_sq = F.conv3d(x * x, kernel, padding=self.pad, groups=c) - mu_x * mu_x
        sigma_y_sq = F.conv3d(y * y, kernel, padding=self.pad, groups=c) - mu_y * mu_y
        sigma_xy = F.conv3d(x * y, kernel, padding=self.pad, groups=c) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x * mu_x + mu_y * mu_y + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2)
        )

        return 1.0 - ssim.mean()
