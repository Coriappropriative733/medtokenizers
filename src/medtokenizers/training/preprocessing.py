"""Medical imaging preprocessing utilities.

Based on MAISI (Guo et al., 2024) preprocessing strategies.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float

from ..modules.utils import jaxtyped_compile_safe


class HounsfieldNormalizer(nn.Module):
    """Normalize CT Hounsfield Units for medical imaging.

    MAISI strategy:
    - Clips HU values to [-1000, 1000] range
    - Normalizes to [0, 1] for stable training

    Example usage:
        normalizer = HounsfieldNormalizer()
        ct_normalized = normalizer(ct_volume)
    """

    def __init__(
        self,
        hu_min: float = -1000.0,  # Air
        hu_max: float = 1000.0,  # Dense bone
        clip: bool = True,
        output_range: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.clip = clip
        self.output_min, self.output_max = output_range

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, ct_volume: Float[torch.Tensor, "batch channels ..."]
    ) -> Float[torch.Tensor, "batch channels ..."]:
        """Normalize CT volume."""
        if self.clip:
            ct_volume = torch.clamp(ct_volume, self.hu_min, self.hu_max)

        normalized = (ct_volume - self.hu_min) / (self.hu_max - self.hu_min)

        # Scale to output range if different from [0, 1]
        if self.output_min != 0.0 or self.output_max != 1.0:
            normalized = (
                normalized * (self.output_max - self.output_min) + self.output_min
            )

        return normalized

    def denormalize(
        self, normalized_volume: Float[torch.Tensor, "batch channels ..."]
    ) -> Float[torch.Tensor, "batch channels ..."]:
        """Convert normalized values back to Hounsfield Units."""
        # Unscale from output range
        if self.output_min != 0.0 or self.output_max != 1.0:
            normalized_volume = (normalized_volume - self.output_min) / (
                self.output_max - self.output_min
            )

        # Denormalize
        ct_volume = normalized_volume * (self.hu_max - self.hu_min) + self.hu_min
        return ct_volume


class RandomCrop3d(nn.Module):
    """Extract random 3D crops from medical volumes for MAISI-style training.

    This enables efficient training on large volumes by extracting smaller
    random patches during data loading, reducing memory requirements.

    Example usage:
        cropper = RandomCrop3d(crop_size=(64, 64, 64))
        batch_crops = cropper(full_volumes)  # (B, C, 512, 512, 512) → (B, C, 64, 64, 64)
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int, int],
        padding: Optional[int] = None,
        pad_if_needed: bool = False,
        fill: float = 0.0,
    ):
        """Initialize random cropping.

        Args:
            crop_size: Size of crop (H, W, D). If int, uses same size for all dims.
            padding: Optional padding to add before cropping
            pad_if_needed: If True, pad the volume if smaller than crop_size
            fill: Value to use for padding
        """
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, volume: Float[torch.Tensor, "batch channels height width depth"]
    ) -> Float[torch.Tensor, "batch channels crop_h crop_w crop_d"]:
        """Extract random crop from volume.

        Args:
            volume: Input volume (B, C, H, W, D)

        Returns:
            cropped: Random crop (B, C, crop_h, crop_w, crop_d)
        """
        if self.padding is not None:
            volume = torch.nn.functional.pad(
                volume, [self.padding] * 6, mode="constant", value=self.fill
            )

        _, _, h, w, d = volume.shape
        crop_h, crop_w, crop_d = self.crop_size

        # Pad if needed
        if self.pad_if_needed:
            if h < crop_h:
                padding = (0, 0, 0, 0, 0, crop_h - h)
                volume = torch.nn.functional.pad(
                    volume, padding, mode="constant", value=self.fill
                )
                h = crop_h
            if w < crop_w:
                padding = (0, 0, 0, crop_w - w, 0, 0)
                volume = torch.nn.functional.pad(
                    volume, padding, mode="constant", value=self.fill
                )
                w = crop_w
            if d < crop_d:
                padding = (0, crop_d - d, 0, 0, 0, 0)
                volume = torch.nn.functional.pad(
                    volume, padding, mode="constant", value=self.fill
                )
                d = crop_d

        if h < crop_h or w < crop_w or d < crop_d:
            raise ValueError(
                f"Volume size ({h}, {w}, {d}) is smaller than crop size {self.crop_size}. "
                "Set pad_if_needed=True to automatically pad."
            )

        # Random crop position
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        front = torch.randint(0, d - crop_d + 1, (1,)).item()

        return volume[
            :, :, top : top + crop_h, left : left + crop_w, front : front + crop_d
        ]


class CenterCrop3d(nn.Module):
    """Extract center crop from 3D medical volumes.

    Useful for validation/testing with consistent crop positions.

    Example usage:
        cropper = CenterCrop3d(crop_size=(128, 128, 128))
        center_crop = cropper(full_volume)
    """

    def __init__(self, crop_size: int | tuple[int, int, int]):
        """Initialize center cropping.

        Args:
            crop_size: Size of crop (H, W, D). If int, uses same size for all dims.
        """
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = tuple(crop_size)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, volume: Float[torch.Tensor, "batch channels height width depth"]
    ) -> Float[torch.Tensor, "batch channels crop_h crop_w crop_d"]:
        """Extract center crop from volume.

        Args:
            volume: Input volume (B, C, H, W, D)

        Returns:
            cropped: Center crop (B, C, crop_h, crop_w, crop_d)
        """
        _, _, h, w, d = volume.shape
        crop_h, crop_w, crop_d = self.crop_size

        if h < crop_h or w < crop_w or d < crop_d:
            raise ValueError(
                f"Volume size ({h}, {w}, {d}) is smaller than crop size {self.crop_size}."
            )

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        front = (d - crop_d) // 2

        return volume[
            :, :, top : top + crop_h, left : left + crop_w, front : front + crop_d
        ]
