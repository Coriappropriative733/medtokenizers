"""Evaluation metrics for medical image tokenizers.

This module provides evaluation metrics for both continuous and discrete
tokenizers including:
- Reconstruction quality: PSNR, SSIM, MSE, MAE
- Perceptual quality: LPIPS
- Discrete tokenizer metrics: Perplexity, codebook usage, entropy
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Module-level cache for LPIPS models (expensive to load)
_LPIPS_CACHE: Dict[Tuple[str, str], object] = {}


@lru_cache(maxsize=8)
def _create_gaussian_window(
    size: int, sigma: float, ndim: int, device: str, dtype_str: str
) -> torch.Tensor:
    """Create and cache Gaussian window for SSIM computation.

    Cached by (size, sigma, ndim, device, dtype) to avoid recreation.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))

    if ndim == 2:
        window = g.view(1, -1) * g.view(-1, 1)
    else:  # 3D
        window = g.view(1, 1, -1) * g.view(1, -1, 1) * g.view(-1, 1, 1)

    window = window / window.sum()

    # Convert dtype string back to torch dtype
    dtype = getattr(torch, dtype_str)
    return window.to(device=device, dtype=dtype)


def _get_lpips_model(net: str, device: str) -> object:
    """Get cached LPIPS model, loading only once per (net, device) combination.

    LPIPS models are expensive to load (~100-500ms) as they require loading
    VGG/AlexNet weights. This function caches them for reuse.
    """
    cache_key = (net, device)
    if cache_key not in _LPIPS_CACHE:
        import lpips

        _LPIPS_CACHE[cache_key] = lpips.LPIPS(net=net).to(device)
    return _LPIPS_CACHE[cache_key]


def clear_metric_caches() -> None:
    """Clear all cached metric computation resources.

    Useful for freeing GPU memory after evaluation or when switching devices.
    """
    _LPIPS_CACHE.clear()
    _create_gaussian_window.cache_clear()


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Reconstruction metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    # Perceptual metrics
    lpips: Optional[float] = None

    # Discrete tokenizer metrics
    perplexity: Optional[float] = None
    codebook_usage: Optional[float] = None
    entropy: Optional[float] = None

    # Efficiency metrics
    compression_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __repr__(self) -> str:
        """Pretty print metrics."""
        lines = ["Evaluation Metrics:"]
        lines.append("-" * 40)
        for key, value in self.to_dict().items():
            if value is not None:
                lines.append(f"  {key:20s}: {value:.4f}")
        return "\n".join(lines)


def compute_mse(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> float:
    """Compute Mean Squared Error.

    Args:
        reconstruction: Reconstructed images (B, C, H, W) or (B, C, H, W, D)
        target: Target images (B, C, H, W) or (B, C, H, W, D)
        mask: Optional binary mask to compute metric only on masked region

    Returns:
        MSE value

    Example:
        >>> recon = torch.randn(8, 1, 128, 128, 128)
        >>> target = torch.randn(8, 1, 128, 128, 128)
        >>> mse = compute_mse(recon, target)
    """
    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        reconstruction = reconstruction * mask
        target = target * mask

    mse = F.mse_loss(reconstruction, target, reduction="mean")
    return mse.item()


def compute_mae(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> float:
    """Compute Mean Absolute Error.

    Args:
        reconstruction: Reconstructed images (B, C, H, W) or (B, C, H, W, D)
        target: Target images (B, C, H, W) or (B, C, H, W, D)
        mask: Optional binary mask to compute metric only on masked region

    Returns:
        MAE value

    Example:
        >>> recon = torch.randn(8, 1, 128, 128, 128)
        >>> target = torch.randn(8, 1, 128, 128, 128)
        >>> mae = compute_mae(recon, target)
    """
    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        reconstruction = reconstruction * mask
        target = target * mask

    mae = F.l1_loss(reconstruction, target, reduction="mean")
    return mae.item()


def compute_psnr(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        reconstruction: Reconstructed images (B, C, H, W) or (B, C, H, W, D)
        target: Target images (B, C, H, W) or (B, C, H, W, D)
        data_range: Maximum possible pixel value (default: 1.0 for normalized images)
        mask: Optional binary mask to compute metric only on masked region

    Returns:
        PSNR value in dB

    Example:
        >>> recon = torch.randn(8, 1, 128, 128, 128)
        >>> target = torch.randn(8, 1, 128, 128, 128)
        >>> psnr = compute_psnr(recon, target, data_range=1.0)
    """
    mse = compute_mse(reconstruction, target, mask)
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return float(psnr)


def compute_ssim(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    window_size: int = 11,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Uses a Gaussian window-based approach to measure structural similarity
    between images. Works for both 2D and 3D images.

    Args:
        reconstruction: Reconstructed images (B, C, H, W) or (B, C, H, W, D)
        target: Target images (B, C, H, W) or (B, C, H, W, D)
        data_range: Maximum possible pixel value (default: 1.0)
        window_size: Size of the Gaussian window (default: 11)
        mask: Optional binary mask to compute metric only on masked region

    Returns:
        SSIM value (between -1 and 1, where 1 is perfect similarity)

    Example:
        >>> recon = torch.randn(8, 1, 128, 128, 128)
        >>> target = torch.randn(8, 1, 128, 128, 128)
        >>> ssim = compute_ssim(recon, target, data_range=1.0)
    """
    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    reconstruction = reconstruction.float()
    target = target.float()

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        reconstruction = reconstruction * mask
        target = target * mask

    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Determine if 2D or 3D
    is_3d = reconstruction.ndim == 5
    ndim = 3 if is_3d else 2

    # Get cached Gaussian window (avoids recreation on each call)
    device_str = str(reconstruction.device)
    dtype_str = str(reconstruction.dtype).split(".")[-1]  # e.g., "float32"
    window = _create_gaussian_window(window_size, 1.5, ndim, device_str, dtype_str)

    # Expand window to match channel dimension
    channel = reconstruction.size(1)
    if is_3d:
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1, -1)
        conv_fn = F.conv3d
    else:
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
        conv_fn = F.conv2d

    # Compute SSIM
    mu1 = conv_fn(reconstruction, window, padding=window_size // 2, groups=channel)
    mu2 = conv_fn(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    padding = window_size // 2
    sigma1_sq = (
        conv_fn(reconstruction**2, window, padding=padding, groups=channel) - mu1_sq
    )
    sigma2_sq = conv_fn(target**2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = (
        conv_fn(reconstruction * target, window, padding=padding, groups=channel)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def compute_lpips(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    net: str = "alex",
    device: Optional[str] = None,
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Note: This requires the lpips package to be installed:
        pip install lpips

    Args:
        reconstruction: Reconstructed images (B, C, H, W)
        target: Target images (B, C, H, W)
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to use for computation

    Returns:
        LPIPS value (lower is better, typically 0-1)

    Example:
        >>> recon = torch.randn(8, 1, 128, 128)
        >>> target = torch.randn(8, 1, 128, 128)
        >>> lpips_val = compute_lpips(recon, target, net='alex')
    """
    try:
        import lpips as _  # noqa: F401 - Check if lpips is available
    except ImportError as err:
        raise ImportError(
            "lpips package not found. Install with: pip install lpips"
        ) from err

    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    reconstruction = reconstruction.float()
    target = target.float()

    # LPIPS only works with 2D images
    if reconstruction.ndim == 5:
        raise ValueError(
            "LPIPS only supports 2D images. For 3D volumes, compute LPIPS "
            "on 2D slices and average."
        )

    # LPIPS expects 3-channel RGB images, so repeat grayscale if needed
    if reconstruction.size(1) == 1:
        reconstruction = reconstruction.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

    # Get device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get cached LPIPS model (avoids expensive reload on each call)
    loss_fn = _get_lpips_model(net, device)
    reconstruction = reconstruction.to(device)
    target = target.to(device)

    # Compute LPIPS
    with torch.no_grad():
        lpips_val = loss_fn(reconstruction, target)

    return lpips_val.mean().item()


def compute_perplexity(
    indices: Union[torch.Tensor, np.ndarray], codebook_size: int
) -> float:
    """Compute codebook perplexity for discrete tokenizers.

    Perplexity measures how well the codebook is being utilized.
    Higher perplexity indicates better codebook usage.

    Args:
        indices: Discrete token indices (B, H, W) or (B, H, W, D)
        codebook_size: Size of the codebook

    Returns:
        Perplexity value

    Example:
        >>> indices = torch.randint(0, 1024, (8, 32, 32, 32))
        >>> perplexity = compute_perplexity(indices, codebook_size=1024)
    """
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    # Flatten indices
    indices_flat = indices.flatten().to(torch.int64)

    # Use unique-based counting to avoid OOM for large codebooks (e.g., LFQ with 2^d entries)
    unique_indices, counts = torch.unique(indices_flat, return_counts=True)
    probs = counts.float() / counts.sum()

    # Compute perplexity
    entropy = -(probs * torch.log(probs)).sum()
    perplexity = torch.exp(entropy)

    return perplexity.item()


def compute_codebook_usage(
    indices: Union[torch.Tensor, np.ndarray], codebook_size: int
) -> float:
    """Compute codebook usage percentage for discrete tokenizers.

    Measures what fraction of the codebook is actually used.

    Args:
        indices: Discrete token indices (B, H, W) or (B, H, W, D)
        codebook_size: Size of the codebook

    Returns:
        Usage percentage (0-100)

    Example:
        >>> indices = torch.randint(0, 1024, (8, 32, 32, 32))
        >>> usage = compute_codebook_usage(indices, codebook_size=1024)
        >>> print(f"Codebook usage: {usage:.1f}%")
    """
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    # Flatten and count unique indices
    unique_indices = torch.unique(indices.flatten())
    usage = (len(unique_indices) / codebook_size) * 100

    return usage


def compute_compression_ratio(
    input_shape: tuple,
    latent_shape: tuple,
    input_dtype_bits: int = 32,
    latent_dtype_bits: int = 32,
) -> float:
    """Compute compression ratio.

    Args:
        input_shape: Input tensor shape
        latent_shape: Latent tensor shape
        input_dtype_bits: Bits per element in input (default: 32 for float32)
        latent_dtype_bits: Bits per element in latent (default: 32 for float32)

    Returns:
        Compression ratio

    Example:
        >>> input_shape = (8, 1, 256, 256, 256)
        >>> latent_shape = (8, 4, 32, 32, 32)
        >>> ratio = compute_compression_ratio(input_shape, latent_shape)
    """
    input_size = np.prod(input_shape) * input_dtype_bits
    latent_size = np.prod(latent_shape) * latent_dtype_bits
    ratio = input_size / latent_size
    return float(ratio)


def compute_all_metrics(
    reconstruction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
    codebook_size: Optional[int] = None,
    compute_lpips_metric: bool = False,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> EvaluationMetrics:
    """Compute all available metrics.

    Args:
        reconstruction: Reconstructed images
        target: Target images
        data_range: Maximum possible pixel value
        indices: Discrete token indices (for discrete tokenizers)
        codebook_size: Size of codebook (for discrete tokenizers)
        compute_lpips_metric: Whether to compute LPIPS (requires lpips package)
        mask: Optional binary mask

    Returns:
        EvaluationMetrics object with all computed metrics

    Example:
        >>> # For continuous tokenizer
        >>> metrics = compute_all_metrics(recon, target, data_range=1.0)
        >>>
        >>> # For discrete tokenizer
        >>> metrics = compute_all_metrics(
        ...     recon, target,
        ...     indices=indices,
        ...     codebook_size=1024
        ... )
    """
    metrics = EvaluationMetrics()

    # Reconstruction metrics
    metrics.mse = compute_mse(reconstruction, target, mask)
    metrics.mae = compute_mae(reconstruction, target, mask)
    metrics.psnr = compute_psnr(reconstruction, target, data_range, mask)
    metrics.ssim = compute_ssim(reconstruction, target, data_range, mask=mask)

    # Perceptual metrics (only for 2D)
    if compute_lpips_metric:
        try:
            if isinstance(reconstruction, np.ndarray):
                reconstruction_tensor = torch.from_numpy(reconstruction)
            else:
                reconstruction_tensor = reconstruction

            if reconstruction_tensor.ndim == 4:  # 2D images
                metrics.lpips = compute_lpips(reconstruction, target)
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Could not compute LPIPS: {e}")

    # Discrete tokenizer metrics
    if indices is not None and codebook_size is not None:
        metrics.perplexity = compute_perplexity(indices, codebook_size)
        metrics.codebook_usage = compute_codebook_usage(indices, codebook_size)

    return metrics
