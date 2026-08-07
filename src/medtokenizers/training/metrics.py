"""Centralized metrics logging for Accelerate training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from accelerate import Accelerator

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Centralized metrics logger with Accelerate integration.

    Provides prefixing and step logging.
    Only logs from main process when using Accelerate.
    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        prefix: str = "",
    ):
        """Initialize metrics logger.

        Args:
            accelerator: Optional Accelerate instance (for distributed training)
            prefix: Prefix to add to all metric names (e.g., "train/" or "val/")
        """
        self.accelerator = accelerator
        self.prefix = prefix
        self._step = 0

    def should_log(self) -> bool:
        """Check if current process should log.

        Returns:
            True if main process or no Accelerate, False otherwise
        """
        if self.accelerator is None:
            return True
        return (
            self.accelerator.is_main_process or self.accelerator.is_local_main_process
        )

    def log_step(self, metrics: dict, step: Optional[int] = None):
        """Log step-level metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (uses internal counter if None)
        """
        if not self.should_log():
            return

        if step is not None:
            self._step = step

        prefixed_metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        prefixed_metrics["step"] = self._step

        if self.accelerator is not None:
            self.accelerator.log(prefixed_metrics, step=self._step)
        else:
            import torch

            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    logger.info(f"Step {self._step}: {prefixed_metrics}")
            else:
                logger.info(f"Step {self._step}: {prefixed_metrics}")
