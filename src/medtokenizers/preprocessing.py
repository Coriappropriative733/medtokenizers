"""Preprocessing utilities for medical image tokenization.

Implements NVIDIA MAISI-style preprocessing for consistent inference.
Uses medrs for high-performance NIfTI I/O.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional, Union

import medrs
import numpy as np
import torch
from torch.nn import functional as nnf


def example_volume_path() -> Path:
    """Return the path to the bundled simulated T1-weighted brain volume.

    The volume is a BrainWeb phantom shipped inside the package so examples and
    tests work from an installed wheel, not only from a source checkout. See
    ``medtokenizers/assets/README.md`` for provenance and citation terms: the
    volume is not covered by this project's MIT licence.
    """
    return Path(
        str(resources.files("medtokenizers") / "assets" / "t1w_brainweb.nii.gz")
    )


def percentile_normalize(
    volume: np.ndarray,
    lower: float = 0.0,
    upper: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
    clip: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Normalize intensity to percentile range.

    Args:
        volume: Input volume
        lower: Lower percentile (default: 0.0)
        upper: Upper percentile (default: 99.5)
        b_min: Output minimum value (default: 0.0)
        b_max: Output maximum value (default: 1.0)
        clip: Whether to clip values outside range (default: False)

    Returns:
        Normalized volume, lower_percentile_value, upper_percentile_value
    """
    lower_val = np.percentile(volume, lower)
    upper_val = np.percentile(volume, upper)

    # Scale to [b_min, b_max]
    normalized = (volume - lower_val) / (upper_val - lower_val + 1e-8)
    normalized = normalized * (b_max - b_min) + b_min

    if clip:
        normalized = np.clip(normalized, b_min, b_max)

    return normalized, lower_val, upper_val


def resample_to_spacing(
    volume: torch.Tensor,
    src_spacing: tuple[float, float, float],
    tgt_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = "trilinear",
) -> torch.Tensor:
    """Resample volume to target spacing.

    Args:
        volume: Input volume tensor (B, C, D, H, W)
        src_spacing: Source voxel spacing (z, y, x)
        tgt_spacing: Target voxel spacing (z, y, x)
        mode: Interpolation mode (default: "trilinear")

    Returns:
        Resampled volume tensor
    """
    # Calculate target size
    src_size = np.array(volume.shape[2:])
    zoom_factors = np.array(src_spacing) / np.array(tgt_spacing)
    tgt_size = (src_size * zoom_factors).astype(int)

    # Resample
    resampled = nnf.interpolate(
        volume,
        size=tuple(tgt_size),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    return resampled


def pad_divisible(
    volume: torch.Tensor,
    k: int = 4,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad volume to be divisible by k.

    Args:
        volume: Input volume tensor (B, C, D, H, W)
        k: Divisibility factor (default: 4)

    Returns:
        Padded volume, padding amounts (d_front, d_back, h_front, h_back, w_front, w_back)
    """
    d, h, w = volume.shape[2:]

    # Calculate padding needed
    pad_d = (k - d % k) % k
    pad_h = (k - h % k) % k
    pad_w = (k - w % k) % k

    # Pad symmetrically
    pad_d_front = pad_d // 2
    pad_d_back = pad_d - pad_d_front
    pad_h_front = pad_h // 2
    pad_h_back = pad_h - pad_h_front
    pad_w_front = pad_w // 2
    pad_w_back = pad_w - pad_w_front

    padded = nnf.pad(
        volume,
        (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_d_front, pad_d_back),
        mode="constant",
        value=0,
    )

    padding = (
        pad_d_front,
        pad_d_back,
        pad_h_front,
        pad_h_back,
        pad_w_front,
        pad_w_back,
    )
    return padded, padding


def unpad(
    volume: torch.Tensor,
    padding: tuple[int, ...],
) -> torch.Tensor:
    """Remove padding from volume.

    Args:
        volume: Padded volume tensor (B, C, D, H, W)
        padding: Padding amounts (d_front, d_back, h_front, h_back, w_front, w_back)

    Returns:
        Unpadded volume tensor
    """
    pad_d_front, pad_d_back, pad_h_front, pad_h_back, pad_w_front, pad_w_back = padding

    d, h, w = volume.shape[2:]
    d_end = d - pad_d_back if pad_d_back > 0 else d
    h_end = h - pad_h_back if pad_h_back > 0 else h
    w_end = w - pad_w_back if pad_w_back > 0 else w

    unpadded = volume[
        :,
        :,
        pad_d_front:d_end,
        pad_h_front:h_end,
        pad_w_front:w_end,
    ]

    return unpadded


def load_nifti(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """Load a NIfTI file using medrs.

    Args:
        path: Path to NIfTI file

    Returns:
        volume: Volume data as numpy array
        metadata: Dictionary with spacing, shape, etc.
    """
    img = medrs.load(str(path))

    # Get volume data
    tensor = img.to_torch_with_dtype_and_device(dtype=torch.float32)
    volume = tensor.numpy()

    metadata = {
        "original_shape": volume.shape,
        "spacing": tuple(img.spacing) if hasattr(img, "spacing") else (1.0, 1.0, 1.0),
    }

    return volume, metadata


def save_nifti(
    volume: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    spacing: Optional[tuple[float, float, float]] = None,
) -> None:
    """Save a volume as a NIfTI file.

    medrs reads NIfTI but offers no array-to-image constructor, so writing goes
    through nibabel.

    Args:
        volume: Volume data (numpy array or torch tensor)
        path: Output path
        spacing: Voxel spacing (optional)
    """
    import nibabel as nib

    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Ensure 3D
    while volume.ndim > 3:
        volume = volume.squeeze(0)

    # Create affine matrix from spacing
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    affine = np.diag([*spacing, 1.0]).astype(np.float32)

    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(path))


def preprocess_for_maisi(
    nifti_path: Union[str, Path],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile_lower: float = 0.0,
    percentile_upper: float = 99.5,
    divisible_k: int = 4,
) -> tuple[torch.Tensor, dict]:
    """NVIDIA MAISI-style preprocessing pipeline.

    Uses medrs for efficient NIfTI loading.

    Args:
        nifti_path: Path to input NIfTI image
        target_spacing: Target voxel spacing in mm (default: 1mm^3)
        percentile_lower: Lower percentile for normalization (default: 0.0)
        percentile_upper: Upper percentile for normalization (default: 99.5)
        divisible_k: Pad to be divisible by k (default: 4)

    Returns:
        Preprocessed volume tensor (1, 1, D, H, W), metadata dict
    """
    # Load volume using medrs
    img = medrs.load(str(nifti_path))
    tensor = img.to_torch()
    volume = tensor.numpy()

    # Get spacing from medrs image
    original_spacing = (
        tuple(img.spacing) if hasattr(img, "spacing") else (1.0, 1.0, 1.0)
    )

    # Store original info
    metadata = {
        "original_shape": volume.shape,
        "original_spacing": original_spacing,
        "source_path": str(nifti_path),
    }

    # 1. Reorient to RAS (medrs handles this if properly oriented)
    # For simplicity, assume input is already RAS or handled by medrs

    # 2. Percentile normalization
    volume_normalized, lower_val, upper_val = percentile_normalize(
        volume,
        lower=percentile_lower,
        upper=percentile_upper,
        b_min=0.0,
        b_max=1.0,
        clip=False,
    )
    metadata["percentile_lower_val"] = lower_val
    metadata["percentile_upper_val"] = upper_val

    # Convert to tensor
    volume_tensor = (
        torch.from_numpy(volume_normalized).float().unsqueeze(0).unsqueeze(0)
    )

    # 3. Resample to target spacing (1mm^3)
    src_spacing = metadata["original_spacing"]
    if src_spacing != target_spacing:
        volume_tensor = resample_to_spacing(
            volume_tensor,
            src_spacing=src_spacing,
            tgt_spacing=target_spacing,
            mode="trilinear",
        )
        metadata["resampled"] = True
        metadata["target_spacing"] = target_spacing
        metadata["resampled_shape"] = tuple(volume_tensor.shape[2:])
    else:
        metadata["resampled"] = False

    # 4. Pad to be divisible by k
    volume_padded, padding = pad_divisible(volume_tensor, k=divisible_k)
    metadata["padding"] = padding
    metadata["padded_shape"] = tuple(volume_padded.shape[2:])

    return volume_padded, metadata


def postprocess_from_maisi(
    reconstruction: torch.Tensor,
    metadata: dict,
    denormalize: bool = True,
) -> np.ndarray:
    """Reverse MAISI preprocessing to get back to original space.

    Args:
        reconstruction: Reconstructed volume tensor (1, 1, D, H, W)
        metadata: Metadata dict from preprocess_for_maisi
        denormalize: Whether to denormalize intensities (default: True)

    Returns:
        Volume in original space as numpy array
    """
    # 1. Remove padding
    volume = unpad(reconstruction, metadata["padding"])

    # 2. Resample back to original spacing if needed
    if metadata["resampled"]:
        volume = nnf.interpolate(
            volume,
            size=metadata["original_shape"],
            mode="trilinear",
            align_corners=False,
        )

    # 3. Denormalize intensities
    volume_np = volume.cpu().numpy()[0, 0]

    if denormalize:
        lower_val = metadata["percentile_lower_val"]
        upper_val = metadata["percentile_upper_val"]
        volume_np = volume_np * (upper_val - lower_val) + lower_val

    return volume_np


__all__ = [
    "percentile_normalize",
    "resample_to_spacing",
    "pad_divisible",
    "unpad",
    "load_nifti",
    "save_nifti",
    "preprocess_for_maisi",
    "postprocess_from_maisi",
]
