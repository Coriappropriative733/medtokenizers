"""Data loading utilities for medical imaging datasets.

Provides functions for loading neuroimaging data from NIfTI files
using medrs for high-performance I/O and preprocessing.

Supports MAISI-style patch-wise training with random crops for efficient
training on large volumes.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import medrs
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, get_worker_info

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Seed NumPy RNG per worker for reproducible augmentation after fork."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        np.random.seed(worker_info.seed % (2**32))


def _percentile_normalize(
    data: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Normalize data using percentile scaling.

    Args:
        data: Input numpy array
        lower_percentile: Lower percentile for clipping (default: 0.5)
        upper_percentile: Upper percentile for clipping (default: 99.5)
        target_min: Target minimum value (default: 0.0)
        target_max: Target maximum value (default: 1.0)

    Returns:
        Normalized numpy array with NaNs replaced by zeros
    """
    # Replace NaNs and Infs with zeros before processing
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Use nanpercentile to be robust to any remaining edge cases
    p_low = np.nanpercentile(data, lower_percentile)
    p_high = np.nanpercentile(data, upper_percentile)

    # Avoid division by zero
    denom = p_high - p_low
    if denom < 1e-8:
        # Constant image - return zeros
        return np.zeros_like(data, dtype=np.float32)

    normalized = (data - p_low) / (denom + 1e-8)
    normalized = normalized * (target_max - target_min) + target_min

    # Final cleanup: replace any NaNs from edge cases and clip
    normalized = np.nan_to_num(
        normalized, nan=0.0, posinf=target_max, neginf=target_min
    )
    return np.clip(normalized, target_min, target_max).astype(np.float32)


VALID_CROP_SIZES = [16, 32, 48, 64, 80, 96, 112, 128]

# Gaussian-like weights centered on 64-96 (indices 3-5)
# [16,  32,   48,   64,   80,   96,  112,  128]
DEFAULT_CROP_SIZE_WEIGHTS = [0.02, 0.05, 0.13, 0.20, 0.20, 0.20, 0.13, 0.07]


class PatchDataset(torch.utils.data.Dataset):
    """Dataset that extracts random crops from volumes for MAISI-style training.

    Supports random anisotropic crop sizes and random target spacing for
    maximum generalization across different acquisition protocols.

    Sampling strategies:
    - "uniform": Uniform random sampling (fast but may sample background)
    - "center": Center-biased Gaussian sampling (good for brain MRI)
    - "foreground": Sample from non-zero regions (best but slower)
    """

    def __init__(
        self,
        file_list,
        crop_size,
        crops_per_volume,
        augment=True,
        use_cache: bool = True,
        max_cache_per_worker: int = 4,
        lowres: bool = False,
        resize_threshold: int = 256,
        reslice_prob: float = 0.5,
        sampling_strategy: str = "center",
        center_bias: float = 0.3,
        foreground_threshold: float = 0.01,
        random_crop_size: bool = False,
        crop_size_choices: Optional[list[int]] = None,
        crop_size_weights: Optional[list[float]] = None,
        anisotropic_crops: bool = False,
        spacing_range: Optional[tuple[float, float]] = None,
    ):
        """Initialize PatchDataset.

        Args:
            file_list: List of NIfTI file paths
            crop_size: Base crop size (used when random_crop_size=False)
            crops_per_volume: Number of crops per volume per epoch
            augment: Whether to apply data augmentation
            use_cache: Whether to cache volume metadata
            max_cache_per_worker: Maximum cached items per worker
            lowres: Whether to use lower resolution
            resize_threshold: Maximum dimension size for resizing
            reslice_prob: Probability of reslicing (0=native, 1=always reslice)
            sampling_strategy: "uniform", "center", or "foreground"
            center_bias: For "center" strategy, controls spread (0.1=tight, 0.5=loose)
            foreground_threshold: For "foreground" strategy, threshold for detection
            random_crop_size: If True, randomly sample crop size from crop_size_choices
            crop_size_choices: List of valid crop sizes (default: [16,32,48,64,80,96,112,128])
            crop_size_weights: Sampling weights for crop_size_choices (default: favor 64-96)
            anisotropic_crops: If True, sample independently per axis (e.g., 16x80x128)
            spacing_range: Tuple (min_mm, max_mm) for random target spacing when reslicing.
                          If None, uses fixed isotropic resampling.
        """
        self.file_list = file_list
        self.base_crop_size = (
            crop_size if isinstance(crop_size, (list, tuple)) else [crop_size] * 3
        )
        self.crops_per_volume = crops_per_volume
        self.augment = augment
        self.use_cache = use_cache
        self.max_cache_per_worker = max_cache_per_worker
        self.lowres = lowres
        self.resize_threshold = resize_threshold
        self.reslice_prob = reslice_prob
        self.sampling_strategy = sampling_strategy
        self.center_bias = center_bias
        self.foreground_threshold = foreground_threshold

        self.random_crop_size = random_crop_size
        self.crop_size_choices = crop_size_choices or VALID_CROP_SIZES
        self.anisotropic_crops = anisotropic_crops
        self.spacing_range = spacing_range

        if crop_size_weights is not None:
            weights = np.array(crop_size_weights, dtype=np.float64)
        else:
            weights = np.array(
                DEFAULT_CROP_SIZE_WEIGHTS[: len(self.crop_size_choices)],
                dtype=np.float64,
            )
        self.crop_size_weights = weights / weights.sum()

        self._metadata_cache = defaultdict(dict)
        self._foreground_cache = defaultdict(dict)
        self._volume_cache = {}
        self._volume_cache_order = []
        self._max_volume_cache = 8

    def _sample_crop_size(self) -> list[int]:
        """Sample a random crop size, optionally anisotropic."""
        if not self.random_crop_size:
            return list(self.base_crop_size)

        if self.anisotropic_crops:
            return [
                int(np.random.choice(self.crop_size_choices, p=self.crop_size_weights))
                for _ in range(3)
            ]
        else:
            size = int(
                np.random.choice(self.crop_size_choices, p=self.crop_size_weights)
            )
            return [size, size, size]

    def _sample_target_spacing(self) -> Optional[float]:
        """Sample random target spacing, or None for native resolution."""
        if self.spacing_range is None:
            return None
        min_sp, max_sp = self.spacing_range
        return float(np.random.uniform(min_sp, max_sp))

    @property
    def crop_size(self) -> list[int]:
        """For compatibility - returns base crop size."""
        return list(self.base_crop_size)

    def __len__(self):
        return len(self.file_list) * self.crops_per_volume

    def _get_volume_metadata(self, file_path: str) -> dict:
        """Get volume metadata (shape, spacing) with caching."""
        worker_key = self._worker_key()
        cache = self._metadata_cache[worker_key]

        if file_path not in cache:
            # Load metadata only (medrs loads lazily)
            img = medrs.load(file_path)
            shape = tuple(img.shape)
            # Handle 4D data - use first 3 dimensions for cropping
            if len(shape) == 4:
                shape = shape[:3]
            cache[file_path] = {
                "shape": shape,
                "spacing": tuple(img.spacing)
                if hasattr(img, "spacing")
                else (1.0, 1.0, 1.0),
            }
        return cache[file_path]

    def _worker_key(self) -> str:
        info = get_worker_info()
        return f"worker_{info.id}" if info is not None else "main"

    def _compute_random_crop_region(
        self, volume_shape: tuple, file_path: str = None, volume_data: np.ndarray = None
    ) -> tuple:
        """Compute crop start position based on sampling strategy.

        Args:
            volume_shape: Shape of the volume (H, W, D)
            file_path: Path to volume file (needed for foreground sampling)
            volume_data: Pre-loaded volume data (avoids double-load for foreground sampling)

        Returns:
            start: Starting position (h, w, d)
            size: Crop size (will be padded if volume is smaller)
        """
        if self.sampling_strategy == "center":
            return self._compute_center_biased_crop_region(volume_shape)
        elif self.sampling_strategy == "foreground" and file_path is not None:
            return self._compute_foreground_crop_region(
                volume_shape, file_path, volume_data=volume_data
            )
        else:
            return self._compute_uniform_crop_region(volume_shape)

    def _compute_uniform_crop_region(self, volume_shape: tuple) -> tuple:
        """Compute uniformly random crop start position.

        Args:
            volume_shape: Shape of the volume (H, W, D)

        Returns:
            start: Starting position (h, w, d)
            size: Crop size (will be padded if volume is smaller)
        """
        crop_size = self.crop_size
        start = []
        actual_size = []

        for dim_size, crop_dim in zip(volume_shape, crop_size):
            if dim_size >= crop_dim:
                # Random start position
                max_start = dim_size - crop_dim
                s = np.random.randint(0, max_start + 1)
                start.append(s)
                actual_size.append(crop_dim)
            else:
                # Volume smaller than crop - start at 0, will need padding
                start.append(0)
                actual_size.append(dim_size)

        return tuple(start), tuple(actual_size)

    def _compute_center_biased_crop_region(self, volume_shape: tuple) -> tuple:
        """Compute center-biased crop start position using truncated Gaussian.

        Samples are drawn from a Gaussian centered on the volume center,
        with spread controlled by center_bias. This is ideal for brain MRI
        where the anatomy is typically centered.

        Args:
            volume_shape: Shape of the volume (H, W, D)

        Returns:
            start: Starting position (h, w, d)
            size: Crop size
        """
        crop_size = self.crop_size
        start = []
        actual_size = []

        for dim_size, crop_dim in zip(volume_shape, crop_size):
            if dim_size >= crop_dim:
                max_start = dim_size - crop_dim
                # Center of valid start positions
                center = max_start / 2.0
                # Standard deviation as fraction of range
                std = max_start * self.center_bias

                if std > 0:
                    # Sample from Gaussian and clip to valid range
                    s = np.random.normal(center, std)
                    s = int(np.clip(s, 0, max_start))
                else:
                    s = int(center)

                start.append(s)
                actual_size.append(crop_dim)
            else:
                start.append(0)
                actual_size.append(dim_size)

        return tuple(start), tuple(actual_size)

    def _get_foreground_coords(
        self, file_path: str, volume_shape: tuple, volume_data: np.ndarray = None
    ) -> np.ndarray:
        """Get cached foreground coordinates for a volume.

        Computes and caches a downsampled foreground mask to find valid
        crop positions efficiently.

        Args:
            file_path: Path to volume file
            volume_shape: Shape of the volume
            volume_data: Pre-loaded volume data (avoids redundant I/O)

        Returns:
            Array of valid crop start positions (N, 3)
        """
        worker_key = self._worker_key()
        cache = self._foreground_cache[worker_key]

        if file_path not in cache:
            # Use pre-loaded data if available, otherwise load from disk
            if volume_data is not None:
                data = volume_data
            else:
                img = medrs.load(file_path)
                data = img.to_numpy()
                if data.ndim == 4:
                    data = data[..., 0]

            # Compute foreground mask (values above threshold percentile)
            threshold = np.percentile(data, 50) * self.foreground_threshold
            if threshold < 1e-6:
                threshold = np.percentile(data[data > 0], 10) if np.any(data > 0) else 0

            # Downsample for efficiency (stride = crop_size // 4)
            stride = max(1, min(self.crop_size) // 4)
            downsampled = data[::stride, ::stride, ::stride]
            fg_mask = downsampled > threshold

            struct = ndimage.generate_binary_structure(3, 1)
            erode_iters = max(1, min(self.crop_size) // (2 * stride))
            fg_mask = ndimage.binary_erosion(fg_mask, struct, iterations=erode_iters)

            coords = np.array(np.where(fg_mask)).T * stride

            valid_coords = []
            for coord in coords:
                valid = True
                for c, vs, cs in zip(coord, volume_shape, self.crop_size):
                    max_start = vs - cs
                    if c > max_start or c < 0:
                        valid = False
                        break
                if valid:
                    valid_coords.append(coord)

            cache[file_path] = np.array(valid_coords) if valid_coords else None

            if len(cache) > self.max_cache_per_worker:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

        return cache[file_path]

    def _compute_foreground_crop_region(
        self, volume_shape: tuple, file_path: str, volume_data: np.ndarray = None
    ) -> tuple:
        """Compute crop start position biased toward foreground regions.

        Samples from positions where the crop will contain actual tissue/signal
        rather than empty background.

        Args:
            volume_shape: Shape of the volume (H, W, D)
            file_path: Path to volume file
            volume_data: Pre-loaded volume data (avoids redundant I/O)

        Returns:
            start: Starting position (h, w, d)
            size: Crop size
        """
        crop_size = self.crop_size
        coords = self._get_foreground_coords(
            file_path, volume_shape, volume_data=volume_data
        )

        if coords is not None and len(coords) > 0:
            # Sample from foreground positions
            idx = np.random.randint(len(coords))
            start = tuple(int(c) for c in coords[idx])
            actual_size = tuple(
                min(cs, vs - s) for cs, vs, s in zip(crop_size, volume_shape, start)
            )
            return start, actual_size
        else:
            # Fallback to center-biased if no foreground found
            return self._compute_center_biased_crop_region(volume_shape)

    def _compute_center_crop_region(self, volume_shape: tuple) -> tuple:
        """Compute center crop start position.

        Args:
            volume_shape: Shape of the volume (H, W, D)

        Returns:
            start: Starting position (h, w, d)
            size: Crop size
        """
        crop_size = self.crop_size
        start = []
        actual_size = []

        for dim_size, crop_dim in zip(volume_shape, crop_size):
            if dim_size >= crop_dim:
                s = (dim_size - crop_dim) // 2
                start.append(s)
                actual_size.append(crop_dim)
            else:
                start.append(0)
                actual_size.append(dim_size)

        return tuple(start), tuple(actual_size)

    def _pad_to_crop_size(
        self, tensor: torch.Tensor, target_shape: list[int]
    ) -> torch.Tensor:
        """Pad tensor to match target_shape if smaller."""
        current_shape = tensor.shape[1:]

        padding = []
        for current, target in zip(reversed(current_shape), reversed(target_shape)):
            pad_needed = max(0, target - current)
            pad_front = pad_needed // 2
            pad_back = pad_needed - pad_front
            padding.extend([pad_front, pad_back])

        if any(p > 0 for p in padding):
            tensor = torch.nn.functional.pad(tensor, padding, mode="constant", value=0)

        return tensor

    def _apply_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a tensor."""
        if not self.augment:
            return tensor

        # Random flips with 10% probability
        if np.random.rand() < 0.1:
            for axis in range(3):
                if np.random.rand() < 0.5:
                    tensor = torch.flip(tensor, dims=[axis + 1])

        # Random intensity adjustments
        if np.random.rand() < 0.5:
            scale = 1.0 + (np.random.rand() - 0.5) * 0.2
            tensor = tensor * scale

        if np.random.rand() < 0.5:
            gamma = 0.8 + np.random.rand() * 0.4
            tensor = torch.pow(tensor.clamp(0, 1), gamma)

        # Random rotation (90 degree increments)
        if np.random.rand() < 0.1:
            k = np.random.randint(1, 4)
            axes = np.random.choice([1, 2, 3], size=2, replace=False)
            tensor = torch.rot90(tensor, k=k, dims=axes.tolist())

        return tensor

    def _get_uncompressed_path(self, file_path: str) -> str:
        """Return .nii path if it exists, otherwise original .nii.gz path."""
        if file_path.endswith(".nii.gz"):
            nii_path = file_path[:-3]
            if os.path.exists(nii_path):
                return nii_path
        return file_path

    def _load_volume_cached(self, file_path: str) -> tuple[np.ndarray, float, float]:
        """Load volume with per-worker LRU cache.

        Uses medrs for reliable loading of both .nii and .nii.gz files.
        Caches volume-level percentile bounds for fast normalization.

        Returns:
            Tuple of (data, p_low, p_high) where p_low/p_high are the 0.5/99.5
            percentile values computed once per volume.
        """
        worker_key = self._worker_key()

        if not hasattr(self, "_worker_volume_cache"):
            self._worker_volume_cache = {}

        cache = self._worker_volume_cache.setdefault(worker_key, {})

        if file_path in cache:
            return cache[file_path]

        actual_path = self._get_uncompressed_path(file_path)

        # Use medrs for reliable loading (handles both .nii and .nii.gz)
        data = medrs.load(actual_path).to_numpy()

        if data.ndim == 4:
            data = data[..., 0]

        # Convert to float32 for consistency
        data = data.astype(np.float32)

        # Compute and cache volume-level percentile bounds
        clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        p_low = float(np.nanpercentile(clean_data, 0.5))
        p_high = float(np.nanpercentile(clean_data, 99.5))

        if len(cache) >= self._max_volume_cache:
            oldest = next(iter(cache))
            del cache[oldest]

        cache[file_path] = (data, p_low, p_high)
        return data, p_low, p_high

    def __getitem__(self, idx):
        max_retries = 10

        current_crop_size = self._sample_crop_size()

        for attempt in range(max_retries):
            volume_idx = (idx + attempt) % len(self.file_list)
            file_path = self.file_list[volume_idx]
            subject_id = _extract_subject_id(file_path)

            try:
                data, p_low, p_high = self._load_volume_cached(file_path)
                volume_shape = data.shape[:3]

                old_crop_size = self.base_crop_size
                self.base_crop_size = current_crop_size

                if self.augment:
                    start, size = self._compute_random_crop_region(
                        volume_shape, file_path, volume_data=data
                    )
                else:
                    start, size = self._compute_center_crop_region(volume_shape)

                self.base_crop_size = old_crop_size

                clamped_start = tuple(
                    min(s, max(0, vs - sz))
                    for s, vs, sz in zip(start, volume_shape, current_crop_size)
                )
                actual_size = tuple(
                    min(sz, vs - cs)
                    for sz, vs, cs in zip(
                        current_crop_size, volume_shape, clamped_start
                    )
                )

                end = tuple(s + sz for s, sz in zip(clamped_start, actual_size))
                slices = tuple(slice(s, e) for s, e in zip(clamped_start, end))
                cropped_data = data[slices].copy()

                # Fast normalization using cached volume-level percentiles
                denom = p_high - p_low
                if denom < 1e-8:
                    cropped_data = np.zeros_like(cropped_data, dtype=np.float32)
                else:
                    cropped_data = np.nan_to_num(
                        cropped_data, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    cropped_data = (
                        ((cropped_data - p_low) / (denom + 1e-8))
                        .clip(0.0, 1.0)
                        .astype(np.float32)
                    )

                crop = torch.from_numpy(cropped_data)
                crop = torch.nan_to_num(crop, nan=0.0, posinf=1.0, neginf=0.0)

                if crop.dim() == 3:
                    crop = crop.unsqueeze(0)

                crop = self._pad_to_crop_size(crop, current_crop_size)
                crop = self._apply_augmentations(crop)

                return {"image": crop, "subject_id": subject_id}

            except Exception as e:
                logger.warning(f"Failed to load {file_path}, trying next volume: {e}")
                continue

        raise RuntimeError(
            f"Failed to load any volume after {max_retries} attempts starting from idx {idx}"
        )


def get_loaders(
    batch_size: int = 1,
    lowres: bool = False,
    augment: bool = True,
    cache: bool = True,
    resize_threshold: int = 256,
    data_dir: Optional[str] = None,
    crop_size: Optional[int] = None,
    crops_per_volume: int = 4,
    reslice_prob: float = 0.5,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    max_cache_per_worker: int = 4,
    sampling_strategy: str = "center",
    center_bias: float = 0.3,
    foreground_threshold: float = 0.01,
    random_crop_size: bool = False,
    crop_size_choices: Optional[list[int]] = None,
    crop_size_weights: Optional[list[float]] = None,
    anisotropic_crops: bool = False,
    spacing_range: Optional[tuple[float, float]] = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Load neuroimaging dataset with optional random crop sizes and spacing."""
    if data_dir is None:
        raise ValueError("data_dir is required for loading training data.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_dir = Path(data_dir)
    logger.info(f"Loading data from: {data_dir}")

    # Find all NIfTI files
    patterns = ["**/*.nii.gz", "**/*.nii"]
    all_files = []
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        all_files.extend([str(f) for f in files])

    # Remove duplicates
    all_files = sorted(set(all_files))

    # Filter out mask files and other non-brain-image files
    # Exclude: fb_mask, deface_mask, any file with "mask" in the name
    # Include: T1w, T2w, FLAIR, and other common brain MRI modalities
    original_count = len(all_files)
    mask_patterns = ["mask", "_seg", "_label", "_dseg", "_probseg"]

    def is_brain_image(filepath: str) -> bool:
        """Check if file is a brain image (not a mask or segmentation)."""
        filename = Path(filepath).name.lower()
        # Exclude files containing mask-related patterns
        for pattern in mask_patterns:
            if pattern in filename:
                return False
        return True

    all_files = [f for f in all_files if is_brain_image(f)]
    filtered_count = original_count - len(all_files)
    if filtered_count > 0:
        logger.info(
            f"Filtered out {filtered_count} mask/segmentation files, keeping {len(all_files)} brain images"
        )

    if len(all_files) == 0:
        raise FileNotFoundError(f"No neuroimaging files found in {data_dir}")

    logger.info(f"Found {len(all_files)} NIfTI files in {data_dir}")

    # Split into train/val (90/10) with reproducible shuffle
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val files")

    # If crop_size is specified, use MAISI-style patch training
    if crop_size is not None:
        logger.info(
            f"Using MAISI-style patch training with crop_size={crop_size}, crops_per_volume={crops_per_volume}"
        )

        train_dataset = PatchDataset(
            train_files,
            crop_size,
            crops_per_volume,
            augment=augment,
            use_cache=cache,
            max_cache_per_worker=max_cache_per_worker,
            lowres=lowres,
            resize_threshold=resize_threshold,
            reslice_prob=reslice_prob,
            sampling_strategy=sampling_strategy,
            center_bias=center_bias,
            foreground_threshold=foreground_threshold,
            random_crop_size=random_crop_size,
            crop_size_choices=crop_size_choices,
            crop_size_weights=crop_size_weights,
            anisotropic_crops=anisotropic_crops,
            spacing_range=spacing_range,
        )
        val_dataset = PatchDataset(
            val_files,
            crop_size,
            1,
            augment=False,
            use_cache=cache,
            max_cache_per_worker=max_cache_per_worker,
            lowres=lowres,
            resize_threshold=resize_threshold,
            reslice_prob=0.0,
            sampling_strategy="uniform",
            center_bias=center_bias,
            foreground_threshold=foreground_threshold,
            random_crop_size=False,
            crop_size_choices=None,
            crop_size_weights=None,
            anisotropic_crops=False,
            spacing_range=None,
        )

        logger.info(
            f"Created patch datasets: {len(train_dataset)} train crops, {len(val_dataset)} val crops"
        )
    else:
        # Whole-volume approach using medrs
        train_dataset = MedrsVolumeDataset(
            train_files,
            augment=augment,
            lowres=lowres,
            resize_threshold=resize_threshold,
            reslice_prob=reslice_prob,
        )
        val_dataset = MedrsVolumeDataset(
            val_files,
            augment=False,
            lowres=lowres,
            resize_threshold=resize_threshold,
            reslice_prob=reslice_prob,
        )
        logger.info(
            f"Created medrs data loaders with {len(train_files)} train and {len(val_files)} val files"
        )

    # Use more workers for faster data loading (adjust based on CPU cores)
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if prefetch_factor is None:
        prefetch_factor = 8 if num_workers > 0 else None

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = prefetch_factor
        loader_common["persistent_workers"] = persistent_workers
    else:
        # prefetch_factor/persistent_workers are invalid when num_workers == 0
        loader_common.pop("prefetch_factor", None)
        loader_common.pop("persistent_workers", None)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        **loader_common,
    )

    return train_loader, val_loader


class MedrsVolumeDataset(torch.utils.data.Dataset):
    """Dataset for loading whole volumes using medrs."""

    def __init__(
        self,
        file_list,
        augment: bool = True,
        lowres: bool = False,
        resize_threshold: int = 256,
        reslice_prob: float = 1.0,
    ):
        self.file_list = file_list
        self.augment = augment
        self.lowres = lowres
        self.resize_threshold = resize_threshold
        self.reslice_prob = reslice_prob

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        subject_id = _extract_subject_id(file_path)

        try:
            # Load with medrs
            img = medrs.load(file_path)

            # Optionally reslice to isotropic (before normalization for efficiency)
            should_reslice = np.random.rand() < self.reslice_prob
            if should_reslice:
                if self.lowres:
                    img = img.resample_to_shape([64, 64, 64])
                elif self.resize_threshold > 0:
                    target_size = min(self.resize_threshold, 192)
                    img = img.resample_to_shape([target_size, target_size, target_size])

            # Convert to numpy for percentile normalization
            data = img.to_numpy()

            # Normalize using percentile scaling (0.5 to 99.5 percentile -> [0, 1])
            data = _percentile_normalize(data, 0.5, 99.5, 0.0, 1.0)

            # Convert to tensor
            tensor = torch.from_numpy(data)

            # Ensure channel first
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            # Apply augmentations
            if self.augment:
                tensor = self._apply_augmentations(tensor)

            # Pad to divisible by 16
            tensor = self._pad_divisible(tensor, k=16)

            return {"image": tensor, "subject_id": subject_id}

        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}") from e

    def _apply_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random flips
        if np.random.rand() < 0.1:
            for axis in range(3):
                if np.random.rand() < 0.5:
                    tensor = torch.flip(tensor, dims=[axis + 1])

        # Random intensity adjustments (histogram shift approximation)
        if np.random.rand() < 0.5:
            scale = 1.0 + (np.random.rand() - 0.5) * 0.2
            tensor = tensor * scale

        # Random contrast adjustment (gamma)
        if np.random.rand() < 0.5:
            gamma = 0.8 + np.random.rand() * 0.4
            tensor = torch.pow(tensor.clamp(0, 1), gamma)

        # Random rotation (90 degree increments)
        if np.random.rand() < 0.1:
            k = np.random.randint(1, 4)
            tensor = torch.rot90(tensor, k=k, dims=[1, 2])

        return tensor

    def _pad_divisible(self, tensor: torch.Tensor, k: int = 16) -> torch.Tensor:
        """Pad tensor to be divisible by k in all spatial dimensions."""
        spatial_dims = tensor.shape[1:]
        padding = []
        for dim in reversed(spatial_dims):
            pad_amount = (k - dim % k) % k
            padding.extend([0, pad_amount])

        if any(p > 0 for p in padding):
            tensor = torch.nn.functional.pad(tensor, padding, mode="constant", value=0)

        return tensor


def get_2d_loaders(
    batch_size: int = 1,
    lowres: bool = False,
    augment: bool = True,
    cache: bool = True,
    data_dir: Optional[str] = None,
    cache_size: Optional[int] = None,
    test_run: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Load 2D slices from neuroimaging dataset.

    Uses medrs for NIfTI loading.

    Args:
        batch_size: Batch size for data loaders
        lowres: Use lower resolution data
        augment: Apply data augmentation
        cache: Cache data in memory
        data_dir: Path to dataset directory (required)
        cache_size: Maximum number of slices to cache (None = cache all)
        test_run: If True, use smaller dataset for quick testing

    Returns:
        train_loader, val_loader: PyTorch data loaders
    """
    if data_dir is None:
        raise ValueError("data_dir is required for loading 2D data.")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_dir = Path(data_dir)
    logger.info(f"Loading 2D slices from: {data_dir}")

    # Find all NIfTI files
    all_files = sorted(
        list(data_dir.glob("**/*.nii.gz")) + list(data_dir.glob("**/*.nii"))
    )

    # Filter out mask files (same logic as get_loaders)
    mask_patterns = ["mask", "_seg", "_label", "_dseg", "_probseg"]
    original_count = len(all_files)
    all_files = [
        f for f in all_files if not any(p in f.name.lower() for p in mask_patterns)
    ]
    filtered_count = original_count - len(all_files)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} mask/segmentation files")

    if len(all_files) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {data_dir}")

    # Limit files for test run
    if test_run:
        all_files = all_files[:10]
        logger.info(f"Test run: Using {len(all_files)} files")

    # Extract 2D slices from volumes using medrs
    logger.info(f"Extracting 2D slices from {len(all_files)} volumes...")
    slices = []
    subject_ids = []

    for file_path in all_files:
        try:
            # Load with medrs
            img = medrs.load(str(file_path))
            data = img.to_numpy()

            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            tensor = torch.from_numpy(data.astype(np.float32))

            # Extract middle slices from each axis
            h, w, d = tensor.shape
            slices.append(tensor[:, :, d // 2].unsqueeze(0))  # Z-axis
            slices.append(tensor[:, h // 2, :].unsqueeze(0))  # Y-axis
            slices.append(tensor[w // 2, :, :].unsqueeze(0))  # X-axis

            # Extract subject ID
            subject_id = _extract_subject_id(str(file_path))
            subject_ids.extend(
                [f"{subject_id}_z", f"{subject_id}_y", f"{subject_id}_x"]
            )

            # Limit cache size
            if cache_size and len(slices) >= cache_size * 3:
                break

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue

    if len(slices) == 0:
        raise ValueError(f"No slices extracted from files in {data_dir}")

    logger.info(f"Extracted {len(slices)} 2D slices")

    # Resize slices if needed
    if lowres:
        slices = [
            torch.nn.functional.interpolate(
                s.unsqueeze(0), size=(128, 128), mode="bilinear"
            ).squeeze(0)
            for s in slices
        ]

    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(slices))
    split_idx = int(0.9 * len(slices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    class SliceDataset(torch.utils.data.Dataset):
        def __init__(self, slices, subject_ids, indices, augment=False):
            self.slices = [slices[i] for i in indices]
            self.subject_ids = [subject_ids[i] for i in indices]
            self.augment = augment

        def __len__(self):
            return len(self.slices)

        def __getitem__(self, idx):
            slice_tensor = self.slices[idx]

            if self.augment:
                # Simple augmentation: random flip
                if np.random.rand() > 0.5:
                    slice_tensor = torch.flip(slice_tensor, [1])
                if np.random.rand() > 0.5:
                    slice_tensor = torch.flip(slice_tensor, [2])

            return {"image": slice_tensor, "subject_id": self.subject_ids[idx]}

    train_dataset = SliceDataset(slices, subject_ids, train_indices, augment=augment)
    val_dataset = SliceDataset(slices, subject_ids, val_indices, augment=False)

    # Use more workers for faster data loading
    num_workers = kwargs.get("num_workers")
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if prefetch_factor is None:
        prefetch_factor = 8 if num_workers > 0 else None

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = prefetch_factor
        loader_common["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset, shuffle=True, worker_init_fn=_worker_init_fn, **loader_common
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, worker_init_fn=_worker_init_fn, **loader_common
    )

    logger.info(
        f"Created 2D loaders: {len(train_dataset)} train, {len(val_dataset)} val slices"
    )
    return train_loader, val_loader


def _extract_subject_id(file_path: str) -> str:
    """Extract subject ID from file path.

    Args:
        file_path: Path to NIfTI file

    Returns:
        Subject ID string
    """
    parts = Path(file_path).parts
    for i, part in enumerate(parts):
        if part.startswith("sub-"):
            dataset_id = parts[i - 1] if i > 0 else "unknown"
            return f"{dataset_id}_{part}"
    return "unknown"
