"""Evaluator class for tokenizer evaluation.

Provides a high-level interface for evaluating trained tokenizers on test
datasets, computing reconstruction metrics (PSNR, SSIM, LPIPS), compression
ratio, and (for discrete tokenizers) codebook utilization metrics.

Example Usage
-------------
>>> from medtokenizers.evaluation import TokenizerEvaluator
>>> from medtokenizers.networks import ContinuousTokenizer
>>>
>>> # Load model and create evaluator
>>> model = ContinuousTokenizer.from_pretrained("./my-vae")
>>> evaluator = TokenizerEvaluator(model, device='cuda')
>>>
>>> # Evaluate on test data
>>> results = evaluator.evaluate(test_loader)
>>> evaluator.print_results(results)
>>>
>>> # Save for later analysis
>>> TokenizerEvaluator.save_results(results, "./eval_results.json")
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from medtokenizers.evaluation.metrics import (
    EvaluationMetrics,
    compute_all_metrics,
    compute_compression_ratio,
)
from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer

if TYPE_CHECKING:
    from collections.abc import Generator


# Type alias for tokenizers
TokenizerType = Union[ContinuousTokenizer, DiscreteTokenizer]


class TokenizerEvaluator:
    """Comprehensive evaluator for medical image tokenizers.

    This class provides a high-level interface for evaluating tokenizers on
    test datasets with detailed metrics and reporting. It handles both
    continuous (VAE) and discrete (VQ-VAE, FSQ) tokenizers.

    The Evaluation Pipeline
    -----------------------
    For each batch:
    1. Forward pass through tokenizer (encode -> [quantize] -> decode)
    2. Compute reconstruction metrics (PSNR, SSIM, optional LPIPS)
    3. For discrete: compute codebook metrics (perplexity, usage)
    4. Aggregate statistics across all batches

    Thread Safety
    -------------
    The evaluator maintains state (model, device) and should not be used
    concurrently from multiple threads. Create separate evaluators for
    parallel evaluation.

    Args:
        model: Trained tokenizer model (ContinuousTokenizer or DiscreteTokenizer)
        device: Device for evaluation ('cuda', 'cpu', or specific GPU like 'cuda:0')
                Auto-detects if None.
        data_range: Maximum pixel value in images (default: 1.0 for normalized data)
        compute_lpips: Whether to compute LPIPS metric. Requires lpips package
                      and adds ~10% overhead.
        use_amp: Whether to use automatic mixed precision for faster evaluation.

    Example:
        >>> model = DiscreteTokenizer.from_pretrained("./my-vqvae")
        >>> evaluator = TokenizerEvaluator(model, device='cuda', compute_lpips=True)
        >>>
        >>> # Full dataset evaluation
        >>> results = evaluator.evaluate(test_loader)
        >>> print(f"PSNR: {results['avg_metrics'].psnr:.2f} dB")
        >>> print(f"SSIM: {results['avg_metrics'].ssim:.4f}")
        >>>
        >>> # Quick sanity check on subset
        >>> quick_results = evaluator.evaluate(test_loader, num_samples=100)
    """

    def __init__(
        self,
        model: TokenizerType,
        device: str | None = None,
        data_range: float = 1.0,
        compute_lpips: bool = False,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.data_range = data_range
        self.compute_lpips = compute_lpips
        self.use_amp = use_amp

        # Determine tokenizer type
        self.is_discrete = isinstance(model, DiscreteTokenizer)

        # Get codebook size for discrete tokenizers
        self.codebook_size: int | None = None
        if self.is_discrete:
            self.codebook_size = self._get_codebook_size()

    def _get_codebook_size(self) -> int | None:
        """Extract codebook size from discrete tokenizer."""
        quantizer = self.model.quantizer

        num_embeddings = getattr(quantizer, "num_embeddings", None)
        if num_embeddings is not None:
            return int(num_embeddings)

        codebook_size = getattr(quantizer, "codebook_size", None)
        if codebook_size is not None:
            return int(codebook_size)

        get_codebook_size = getattr(quantizer, "get_codebook_size", None)
        if callable(get_codebook_size):
            return int(get_codebook_size())

        levels = getattr(quantizer, "levels", None)
        if levels is not None:
            return int(np.prod(levels))

        return None

    @contextmanager
    def _evaluation_context(self) -> Generator[None, None, None]:
        """Context manager for evaluation mode with optional AMP.

        Configures:
        - Model in eval mode
        - torch.inference_mode() for maximum speed
        - Optional autocast for mixed precision

        Yields:
            None
        """
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                if self.use_amp and self.device != "cpu":
                    with torch.autocast(device_type="cuda"):
                        yield
                else:
                    yield
        finally:
            if was_training:
                self.model.train()

    def evaluate_batch(
        self, images: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Evaluate a single batch of images.

        Processes one batch through the tokenizer and computes all metrics.
        Uses inference mode and optional AMP for efficiency.

        Args:
            images: Input images of shape (B, C, H, W) or (B, C, H, W, D)
            mask: Optional binary mask for masked metrics. Same spatial shape.

        Returns:
            Dictionary containing:
            - 'metrics': EvaluationMetrics object with all computed metrics
            - 'compression_ratio': Input/latent size ratio
            - 'reconstructions': Reconstructed images (on CPU)
            - 'indices': Quantization indices (discrete only, on CPU)

        Example:
            >>> batch = next(iter(test_loader))
            >>> results = evaluator.evaluate_batch(batch)
            >>> print(f"Batch PSNR: {results['metrics'].psnr:.2f}")
        """
        with self._evaluation_context():
            # Transfer to device (non-blocking for async)
            images = images.to(self.device, non_blocking=True)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)

            # Forward pass through tokenizer
            indices: torch.Tensor | None = None
            latent_shape_tensor: torch.Tensor

            if self.is_discrete:
                indices, quant_codes, _ = self.model.encode(images)  # type: ignore[misc]
                reconstructions = self.model.decode(quant_codes)  # type: ignore[arg-type]
                latent_shape_tensor = indices
            else:
                latents, _ = self.model.encode(images)  # type: ignore[misc]
                reconstructions = self.model.decode(latents)
                latent_shape_tensor = latents

            # Transfer results to CPU for metric computation
            # (metrics library may not support CUDA)
            images_cpu = images.cpu()
            reconstructions_cpu = reconstructions.cpu()
            indices_cpu = indices.cpu() if indices is not None else None
            mask_cpu = mask.cpu() if mask is not None else None

        # Compute metrics (outside inference_mode for potential autograd in LPIPS)
        metrics = compute_all_metrics(
            reconstruction=reconstructions_cpu,
            target=images_cpu,
            data_range=self.data_range,
            indices=indices_cpu,
            codebook_size=self.codebook_size,
            compute_lpips_metric=self.compute_lpips,
            mask=mask_cpu,
        )

        # Compute compression ratio
        input_shape = tuple(images.shape)
        latent_shape = tuple(latent_shape_tensor.shape)

        compression_ratio = compute_compression_ratio(input_shape, latent_shape)

        return {
            "metrics": metrics,
            "compression_ratio": compression_ratio,
            "reconstructions": reconstructions_cpu,
            "indices": indices_cpu,
        }

    def evaluate(
        self,
        data_loader: DataLoader,
        num_samples: int | None = None,
        save_reconstructions: bool = False,
    ) -> dict[str, Any]:
        """Evaluate model on a complete dataset.

        Iterates through the data loader, computing metrics for each batch
        and aggregating into summary statistics.

        Args:
            data_loader: DataLoader yielding test batches. Can yield:
                        - Tensor: images only
                        - Tuple: (images, masks)
                        - Dict: {'image': images, 'mask': masks}
            num_samples: Maximum samples to evaluate (None = all)
            save_reconstructions: Whether to keep first 10 reconstruction samples

        Returns:
            Dictionary containing:
            - 'avg_metrics': Aggregated EvaluationMetrics
            - 'num_samples': Total samples evaluated
            - 'model_config': Model configuration dict
            - 'is_discrete': Whether discrete tokenizer
            - 'reconstruction_samples': List of sample dicts (if save_reconstructions)

        Example:
            >>> # Full evaluation
            >>> results = evaluator.evaluate(test_loader)
            >>>
            >>> # Quick subset evaluation
            >>> results = evaluator.evaluate(test_loader, num_samples=100)
            >>>
            >>> # With reconstruction samples
            >>> results = evaluator.evaluate(test_loader, save_reconstructions=True)
            >>> for i, sample in enumerate(results['reconstruction_samples']):
            ...     visualize(sample['original'], sample['reconstruction'])
        """
        all_metrics: list[EvaluationMetrics] = []
        compression_ratios: list[float] = []
        reconstruction_samples: list[dict[str, torch.Tensor | None]] = []

        num_processed = 0
        pbar = tqdm(data_loader, desc="Evaluating")

        for batch in pbar:
            # Handle different batch formats
            images, mask = self._unpack_batch(batch)

            # Evaluate batch
            batch_results = self.evaluate_batch(images, mask)

            # Collect metrics
            all_metrics.append(batch_results["metrics"])
            compression_ratios.append(batch_results["compression_ratio"])

            # Optionally save reconstruction samples
            if save_reconstructions and len(reconstruction_samples) < 10:
                reconstruction_samples.append(
                    {
                        "original": images.cpu(),
                        "reconstruction": batch_results["reconstructions"],
                        "indices": batch_results["indices"],
                    }
                )

            # Update progress bar
            num_processed += len(images)
            pbar.set_postfix(
                {
                    "PSNR": f"{batch_results['metrics'].psnr:.2f}",
                    "SSIM": f"{batch_results['metrics'].ssim:.4f}",
                }
            )

            # Check sample limit
            if num_samples is not None and num_processed >= num_samples:
                break

        # Aggregate metrics across all batches
        avg_metrics = self._aggregate_metrics(all_metrics)
        avg_metrics.compression_ratio = float(np.mean(compression_ratios))

        results: dict[str, Any] = {
            "avg_metrics": avg_metrics,
            "num_samples": num_processed,
            "model_config": self.model.config,
            "is_discrete": self.is_discrete,
        }

        if save_reconstructions:
            results["reconstruction_samples"] = reconstruction_samples

        return results

    def _unpack_batch(
        self, batch: torch.Tensor | tuple | dict
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract images and optional mask from batch.

        Args:
            batch: Batch in various formats

        Returns:
            Tuple of (images, mask or None)
        """
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            images = batch.get("image") or batch.get("images")
            if images is None:
                raise ValueError("Dict batch must contain 'image' or 'images' key")
            mask = batch.get("mask")
        else:
            images = batch
            mask = None
        return images, mask

    def _aggregate_metrics(
        self, metrics_list: list[EvaluationMetrics]
    ) -> EvaluationMetrics:
        """Aggregate metrics from multiple batches.

        Computes mean of each metric across all batches, handling
        None values appropriately.

        Args:
            metrics_list: List of per-batch EvaluationMetrics

        Returns:
            Single EvaluationMetrics with aggregated values
        """
        metric_dict: dict[str, float | None] = {}

        for metric_name in metrics_list[0].__dict__.keys():
            values = [
                getattr(m, metric_name)
                for m in metrics_list
                if getattr(m, metric_name) is not None
            ]
            if values:
                metric_dict[metric_name] = float(np.mean(values))
            else:
                metric_dict[metric_name] = None

        return EvaluationMetrics(**metric_dict)

    @staticmethod
    def save_results(
        results: dict[str, Any], save_path: str | Path, save_samples: bool = False
    ) -> None:
        """Save evaluation results to disk.

        Saves metrics as JSON for easy parsing. Optionally saves
        reconstruction samples as compressed numpy archive.

        Args:
            results: Results dictionary from evaluate()
            save_path: Path for JSON results
            save_samples: Whether to also save reconstruction samples as .npz

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> TokenizerEvaluator.save_results(results, "./results.json")
            >>> # Creates: ./results.json and optionally ./results.npz
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare JSON-serializable results
        json_results = {
            "avg_metrics": results["avg_metrics"].to_dict(),
            "num_samples": results["num_samples"],
            "model_config": results["model_config"],
            "is_discrete": results["is_discrete"],
        }

        # Save JSON
        with open(save_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {save_path}")

        # Save reconstruction samples if requested
        if save_samples and "reconstruction_samples" in results:
            samples_path = save_path.with_suffix(".npz")
            samples_data: dict[str, np.ndarray] = {}

            for i, sample in enumerate(results["reconstruction_samples"]):
                samples_data[f"original_{i}"] = sample["original"].numpy()
                samples_data[f"reconstruction_{i}"] = sample["reconstruction"].numpy()
                if sample["indices"] is not None:
                    samples_data[f"indices_{i}"] = sample["indices"].numpy()

            np.savez_compressed(samples_path, **samples_data)
            print(f"Reconstruction samples saved to {samples_path}")

    @staticmethod
    def load_results(load_path: str | Path) -> dict[str, Any]:
        """Load evaluation results from disk.

        Args:
            load_path: Path to results JSON file

        Returns:
            Results dictionary with EvaluationMetrics object

        Example:
            >>> results = TokenizerEvaluator.load_results("./results.json")
            >>> print(f"Loaded results: PSNR={results['avg_metrics'].psnr:.2f}")
        """
        load_path = Path(load_path)

        with open(load_path) as f:
            json_results = json.load(f)

        # Convert metrics dict back to EvaluationMetrics object
        json_results["avg_metrics"] = EvaluationMetrics(**json_results["avg_metrics"])

        return json_results

    def print_results(self, results: dict[str, Any]) -> None:
        """Pretty print evaluation results to console.

        Displays a formatted summary of evaluation metrics and model info.

        Args:
            results: Results dictionary from evaluate()

        Example:
            >>> results = evaluator.evaluate(test_loader)
            >>> evaluator.print_results(results)
            # Prints formatted table of metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {results['model_config'].get('name', 'Unknown')}")
        model_type = "Discrete" if results["is_discrete"] else "Continuous"
        print(f"Type: {model_type} Tokenizer")
        print(f"Samples evaluated: {results['num_samples']}")
        print("=" * 60)
        print(results["avg_metrics"])
        print("=" * 60)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for evaluation from tensors/arrays.

    Convenient wrapper when you have images already loaded in memory
    and want to evaluate with TokenizerEvaluator.

    Args:
        data: Images as tensor (N, C, *spatial) or numpy array
        masks: Optional masks with same batch dimension

    Example:
        >>> # From numpy arrays
        >>> images = np.random.randn(100, 1, 128, 128, 128).astype(np.float32)
        >>> dataset = SimpleDataset(images)
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> results = evaluator.evaluate(loader)
        >>>
        >>> # From torch tensors
        >>> images = torch.randn(100, 1, 64, 64, 64)
        >>> dataset = SimpleDataset(images)
    """

    def __init__(
        self,
        data: torch.Tensor | np.ndarray,
        masks: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        # Convert to torch if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        self.data = data

        if masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()
            self.masks = masks
        else:
            self.masks = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.masks is not None:
            return self.data[idx], self.masks[idx]
        return self.data[idx]
