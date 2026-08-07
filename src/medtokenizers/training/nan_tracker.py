"""NaN tracking and recovery for robust training."""

from __future__ import annotations

import torch


class NaNTracker:
    """Track NaN occurrences in outputs and gradients during training.

    Supports:
    - Tracking NaN count and rate
    - Skipping batches with NaN
    - Aborting training if NaN rate exceeds threshold
    """

    def __init__(self, threshold: float = 0.1):
        """Initialize NaN tracker.

        Args:
            threshold: Maximum allowed NaN rate (default: 0.1 = 10%)
        """
        self.threshold = threshold
        self.recon_nan_count = 0
        self.quant_nan_count = 0
        self.total_batches = 0
        self.skipped_batches = 0

    def check_outputs(
        self, reconstruction: torch.Tensor, quant_loss: torch.Tensor
    ) -> bool:
        """Check model outputs for NaN and update tracking.

        Args:
            reconstruction: Model reconstruction output
            quant_loss: Quantization loss

        Returns:
            True if batch should be skipped (NaN detected), False otherwise

        Raises:
            RuntimeError: If NaN rate exceeds threshold
        """
        self.total_batches += 1
        has_nan = False

        if torch.isnan(reconstruction).any():
            self.recon_nan_count += 1
            has_nan = True

        if torch.isnan(quant_loss):
            self.quant_nan_count += 1
            has_nan = True

        if has_nan:
            nan_rate = (
                max(self.recon_nan_count, self.quant_nan_count) / self.total_batches
                if self.total_batches > 0
                else 0.0
            )

            if nan_rate > self.threshold:
                raise RuntimeError(
                    f"NaN rate {nan_rate:.2%} exceeded {self.threshold:.0%} threshold. "
                    f"Training diverged. Check learning rate, data normalization, and model architecture."
                )

            self.skipped_batches += 1
            return True

        return False

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check total loss for NaN.

        Args:
            loss: Total loss tensor

        Returns:
            True if batch should be skipped (NaN detected), False otherwise
        """
        if torch.isnan(loss):
            self.skipped_batches += 1
            return True
        return False

    def check_gradients(self, model: torch.nn.Module) -> bool:
        """Check model gradients for NaN.

        Args:
            model: PyTorch model

        Returns:
            True if optimization step should be skipped (NaN detected), False otherwise
        """
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                self.skipped_batches += 1
                return True
        return False

    def get_statistics(self) -> dict:
        """Get current NaN tracking statistics.

        Returns:
            Dictionary with nan counts, total batches, skipped batches, and rate
        """
        nan_rate = (
            max(self.recon_nan_count, self.quant_nan_count) / self.total_batches
            if self.total_batches > 0
            else 0.0
        )
        return {
            "recon_nan_count": self.recon_nan_count,
            "quant_nan_count": self.quant_nan_count,
            "total_batches": self.total_batches,
            "skipped_batches": self.skipped_batches,
            "nan_rate": nan_rate,
        }

    def reset(self):
        """Reset all counters."""
        self.recon_nan_count = 0
        self.quant_nan_count = 0
        self.total_batches = 0
        self.skipped_batches = 0
