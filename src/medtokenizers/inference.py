"""Inference utilities for medical image tokenizers."""

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch

from medtokenizers.networks import (
    ContinuousTokenizer,
    DiscreteTokenizer,
    RAETokenizer,
    TiTokTokenizer,
)

# Efficient dtype mappings for storage
INDICES_DTYPE = np.int16  # Supports vocab up to 32767 (sufficient for all quantizers)
LATENTS_DTYPE = np.float16  # Half precision for continuous latents


def load_tokenizer(
    model_name_or_path: Union[str, Path], device: Optional[str] = None, **kwargs
) -> Union[ContinuousTokenizer, DiscreteTokenizer, RAETokenizer, TiTokTokenizer]:
    """Load tokenizer from local path or HuggingFace Hub.

    Attempts to load as ContinuousTokenizer, DiscreteTokenizer, TiTokTokenizer,
    and RAETokenizer (in that order). Raises detailed error if all fail.

    Args:
        model_name_or_path: Local path or HuggingFace Hub repo ID
        device: Device to load model to (default: auto-detect cuda/cpu)
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded tokenizer model in inference mode

    Raises:
        ValueError: If model cannot be loaded as either tokenizer type.
            Error message includes details from both loading attempts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    errors: list[tuple[str, Exception]] = []

    for cls in [ContinuousTokenizer, DiscreteTokenizer, TiTokTokenizer, RAETokenizer]:
        try:
            model = cls.from_pretrained(
                model_name_or_path, map_location=device, **kwargs
            )
            model.train(False)
            return model
        except (KeyError, TypeError) as e:
            # Config mismatch - this tokenizer type doesn't match, try next
            errors.append((cls.__name__, e))
        except FileNotFoundError as e:
            # Model path doesn't exist - fail fast, don't try other types
            raise ValueError(f"Model not found at {model_name_or_path}: {e}") from e
        except (ValueError, RuntimeError, OSError) as e:
            # Common loading errors - record but continue to try other type
            errors.append((cls.__name__, e))

    # Both failed - provide detailed error message
    error_details = "; ".join(f"{name}: {err}" for name, err in errors)
    raise ValueError(
        f"Could not load model from {model_name_or_path}. "
        "Tried ContinuousTokenizer, DiscreteTokenizer, TiTokTokenizer, "
        f"and RAETokenizer. Errors: {error_details}"
    )


def save_indices(
    indices: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    dtype: Literal["int16", "int32"] = "int16",
) -> None:
    """Save discrete tokenizer indices to disk with efficient storage.

    Uses int16 by default which supports vocabulary sizes up to 32,767.
    For larger vocabularies (unlikely), use int32.

    Args:
        indices: Discrete indices from tokenizer.encode() or tokenizer.tokenize()
        save_path: Path to save file (will add .npz suffix)
        dtype: Storage dtype - "int16" (default, 2 bytes) or "int32" (4 bytes)

    Raises:
        TypeError: If indices are not integer tensors/arrays
        ValueError: If indices are negative or exceed dtype range

    Example:
        >>> indices, _, _ = tokenizer.encode(images)
        >>> save_indices(indices, "tokens/batch_001")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype not in ("int16", "int32"):
        raise ValueError(f"dtype must be 'int16' or 'int32', got {dtype}.")

    if isinstance(indices, torch.Tensor):
        if indices.is_floating_point() or indices.dtype == torch.bool:
            raise TypeError(
                f"save_indices() expects integer tensor indices, got {indices.dtype}."
            )
        numel = indices.numel()
        if numel > 0:
            min_val = int(indices.min().item())
            max_val = int(indices.max().item())
    elif isinstance(indices, np.ndarray):
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                f"save_indices() expects integer numpy arrays, got {indices.dtype}."
            )
        numel = indices.size
        if numel > 0:
            min_val = int(indices.min())
            max_val = int(indices.max())
    else:
        raise TypeError(
            "save_indices() expects torch.Tensor or numpy.ndarray, "
            f"got {type(indices).__name__}."
        )

    if numel > 0:
        if min_val < 0:
            raise ValueError(
                f"indices must be non-negative, found minimum value {min_val}."
            )

        max_allowed = (
            np.iinfo(np.int16).max if dtype == "int16" else np.iinfo(np.int32).max
        )
        if max_val > max_allowed:
            raise ValueError(
                f"max index {max_val} exceeds {dtype} range ({max_allowed}). "
                f"Use dtype='int32' or reduce codebook size."
            )

    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()

    np_dtype = INDICES_DTYPE if dtype == "int16" else np.int32
    indices = indices.astype(np_dtype)

    np.savez_compressed(save_path.with_suffix(".npz"), indices=indices)


def load_indices(
    load_path: Union[str, Path],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Load discrete tokenizer indices from disk.

    Args:
        load_path: Path to .npz file containing indices
        device: Device to load tensor to (default: CPU)
        dtype: Output dtype for PyTorch tensor (default: int64 for compatibility)

    Returns:
        Indices tensor ready for tokenizer.decode()

    Example:
        >>> indices = load_indices("tokens/batch_001.npz", device="cuda")
        >>> images = tokenizer.decode(indices)
    """
    indices = np.load(load_path)["indices"]
    indices = torch.from_numpy(indices.astype(np.int64)).to(dtype)

    if device is not None:
        indices = indices.to(device)

    return indices


def save_latents(
    latents: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    dtype: Literal["float16", "float32"] = "float16",
) -> None:
    """Save continuous tokenizer latents to disk with efficient storage.

    Uses float16 by default which is sufficient for most latent diffusion
    applications. Use float32 if full precision is required.

    Args:
        latents: Continuous latents from tokenizer.encode() or tokenizer.tokenize()
        save_path: Path to save file (will add .npz suffix)
        dtype: Storage dtype - "float16" (default, 2 bytes) or "float32" (4 bytes)

    Example:
        >>> latents, _ = tokenizer.encode(images)
        >>> save_latents(latents, "latents/batch_001")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()

    np_dtype = LATENTS_DTYPE if dtype == "float16" else np.float32
    latents = latents.astype(np_dtype)

    np.savez_compressed(save_path.with_suffix(".npz"), latents=latents)


def load_latents(
    load_path: Union[str, Path],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load continuous tokenizer latents from disk.

    Args:
        load_path: Path to .npz file containing latents
        device: Device to load tensor to (default: CPU)
        dtype: Output dtype for PyTorch tensor (default: float32)

    Returns:
        Latents tensor ready for tokenizer.decode()

    Example:
        >>> latents = load_latents("latents/batch_001.npz", device="cuda")
        >>> images = tokenizer.decode(latents)
    """
    latents = np.load(load_path)["latents"]
    latents = torch.from_numpy(latents.astype(np.float32)).to(dtype)

    if device is not None:
        latents = latents.to(device)

    return latents


__all__ = [
    "load_tokenizer",
    "save_indices",
    "load_indices",
    "save_latents",
    "load_latents",
    "INDICES_DTYPE",
    "LATENTS_DTYPE",
]
