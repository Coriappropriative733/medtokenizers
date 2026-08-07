"""Callback system for extensible training."""

from __future__ import annotations

import logging
import math
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


def _metric_to_float(value: Any) -> float:
    """Convert scalar metric values to Python floats for serialization/comparison."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().item())
    return float(value)


def _emit_message(trainer: Any, message: str) -> None:
    """Emit callback output through accelerator.print when available, else stdout."""
    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "print"):
        trainer.accelerator.print(message)
    elif hasattr(trainer, "_log"):
        trainer._log(message)
    else:
        print(message)


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks allow you to inject custom behavior at different points
    during training without modifying the trainer code.
    """

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: Trainer, batch: Any, batch_idx: int) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(
        self, trainer: Trainer, batch: Any, batch_idx: int, loss: float
    ) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, trainer: Trainer) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer: Trainer, metrics: dict, **kwargs) -> None:
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to stop training when validation loss stops improving."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, monitor: str = "val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value
        elif current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                logger.info(f"Early stopping triggered at epoch {epoch}")


class Checkpoint(Callback):
    """Save model checkpoints during training."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        verbose: bool = True,
        save_every_n_epochs: Optional[int] = None,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_value: Optional[float] = None
        self.save_every_n_epochs = save_every_n_epochs

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        current_raw = metrics.get(self.monitor)
        if current_raw is None:
            return
        current_value = _metric_to_float(current_raw)
        if math.isnan(current_value):
            return

        should_save = False
        if not self.save_best_only:
            should_save = True
        else:
            if self.best_value is None:
                should_save = True
                self.best_value = current_value
            else:
                if self.mode == "min" and current_value < self.best_value:
                    should_save = True
                    self.best_value = current_value
                elif self.mode == "max" and current_value > self.best_value:
                    should_save = True
                    self.best_value = current_value

        if should_save:
            self._save(trainer, epoch, metrics, tag="best")

        # Periodic checkpoint saving (unconditional, at fixed intervals)
        if (
            self.save_every_n_epochs is not None
            and (epoch + 1) % self.save_every_n_epochs == 0
            and not should_save  # avoid duplicate save on same epoch
        ):
            self._save(trainer, epoch, metrics, tag="periodic")

    def _save(self, trainer: Trainer, epoch: int, metrics: dict, tag: str = "") -> None:
        """Save a checkpoint to disk.

        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Epoch metrics dict
            tag: Label for verbose output ('best', 'periodic', etc.)
        """
        metrics_payload = {k: _metric_to_float(v) for k, v in metrics.items()}
        checkpoint_stem = (
            self.filepath.stem if self.filepath.suffix else self.filepath.name
        )

        if hasattr(trainer, "save_checkpoint"):
            checkpoint_path = self.filepath.parent / f"{checkpoint_stem}_epoch{epoch}"
            trainer.save_checkpoint(
                str(checkpoint_path),
                metadata={"metrics": metrics_payload},
            )
        else:
            checkpoint_path = (
                self.filepath.parent / f"{checkpoint_stem}_epoch{epoch}.pt"
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics_payload,
            }
            torch.save(checkpoint, checkpoint_path)
        if self.verbose:
            label = f" ({tag})" if tag else ""
            _emit_message(trainer, f"Saved checkpoint{label} to {checkpoint_path}")


class LRScheduler(Callback):
    """Learning rate scheduling."""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val_loss = metrics.get("val_loss")
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


class Logger(Callback):
    """Log training progress."""

    def __init__(self, print_every: int = 10):
        self.print_every = print_every
        self.batch_losses = []

    def on_batch_end(
        self, trainer: Trainer, batch: Any, batch_idx: int, loss: float
    ) -> None:
        self.batch_losses.append(loss)
        if (batch_idx + 1) % self.print_every == 0:
            avg_loss = sum(self.batch_losses[-self.print_every :]) / len(
                self.batch_losses[-self.print_every :]
            )
            _emit_message(trainer, f"Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict) -> None:
        if metrics:
            metrics_str = " | ".join(
                [f"{k}: {_metric_to_float(v):.4f}" for k, v in metrics.items()]
            )
            _emit_message(trainer, f"Epoch {epoch}: {metrics_str}")
        else:
            _emit_message(trainer, f"Epoch {epoch}")
        self.batch_losses = []


class ReconstructionLogger(Callback):
    """Log reconstruction samples to wandb."""

    def __init__(self, num_samples: int = 4, every_n_epochs: int = 1):
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_end(
        self, trainer: Trainer, metrics: dict, batch: Any = None, **kwargs
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if batch is None:
            return
        if not trainer.accelerator.is_main_process:
            return

        try:
            import numpy as np
            import wandb
        except ImportError:
            return

        images = batch
        if isinstance(images, dict):
            images = images["image"]
        elif isinstance(images, (list, tuple)):
            images = images[0]
        images = images[: self.num_samples].to(trainer.accelerator.device)

        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(images)
            if isinstance(output, dict):
                recons = output["reconstructions"]
            else:
                recons = output.reconstructions

        images_np = images.cpu().float().numpy()
        recons_np = recons.cpu().float().numpy()

        def _normalize_slice(x: Any) -> Any:
            x_min = x.min()
            x_max = x.max()
            denom = x_max - x_min
            if denom < 1e-8:
                return np.zeros_like(x)
            return (x - x_min) / (denom + 1e-8)

        log_images = []
        for i in range(min(self.num_samples, images_np.shape[0])):
            img = images_np[i, 0]
            rec = recons_np[i, 0]

            # 3D volumes: log the middle slice. 2D images: log full frame.
            if img.ndim >= 3:
                mid_slice = img.shape[0] // 2
                img_slice = img[mid_slice]
                rec_slice = rec[mid_slice]
            else:
                img_slice = img
                rec_slice = rec

            # Defensive handling for unexpected 1D shapes
            img_slice = np.atleast_2d(img_slice)
            rec_slice = np.atleast_2d(rec_slice)
            diff_slice = np.abs(img_slice - rec_slice)

            img_slice = _normalize_slice(img_slice)
            rec_slice = _normalize_slice(rec_slice)
            diff_slice = _normalize_slice(diff_slice)

            combined = np.concatenate([img_slice, rec_slice, diff_slice], axis=1)
            log_images.append(
                wandb.Image(combined, caption=f"Sample {i}: input | recon | diff")
            )

        trainer.accelerator.log(
            {"reconstructions": log_images}, step=trainer.global_step
        )
