"""Accelerate-first trainer for medical tokenizers."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..modules.base import BaseTokenizer
from .callbacks import Callback, Logger
from .configs import LossConfig
from .losses import Combined
from .metrics import MetricsLogger
from .nan_tracker import NaNTracker


def _infinite_loader(loader):
    """Yield batches indefinitely, creating fresh iterators each pass."""
    while True:
        yield from loader


# Enable faster kernel selection and matmul where supported
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    # Use "high" to enable TF32 for Ampere+ GPUs (faster matmul with minimal precision loss)
    torch.set_float32_matmul_precision("high")


class Trainer:
    """Accelerate-first trainer for medical tokenizers.

    Supports:
    - Flexible callback system
    - Mixed precision training (via Accelerate)
    - Gradient accumulation
    - Validation during training
    - Checkpoint saving and loading (Accelerate directory format)
    - Multi-loss training with stage scheduling
    - NaN detection and recovery
    - Compound losses (VQGANLoss, VAEGANLoss) with separate discriminator optimizer
    - KL annealing for VAE training
    """

    def __init__(
        self,
        model: BaseTokenizer,
        optimizer: torch.optim.Optimizer,
        accelerator,
        loss_fn: Optional[nn.Module] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        callbacks: Optional[list[Callback]] = None,
        channels_last: bool = False,
        loss_config: Optional[LossConfig] = None,
        scheduler=None,
        warmup_scheduler=None,
        warmup_steps: int = 0,
        log_every_steps: int = 1,
        nan_threshold: float = 0.1,
        enable_compile_train_step: bool = False,
        enable_gradient_checkpointing: bool = False,
        disc_optimizer: Optional[torch.optim.Optimizer] = None,
        model_ema_decay: Optional[float] = None,
    ):
        """Initialize trainer.

        Args:
            model: Tokenizer model to train
            optimizer: Optimizer for training
            accelerator: Accelerate instance (required)
            loss_fn: Loss function (defaults to Combined)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            callbacks: List of callbacks for extensibility
            channels_last: If True, convert inputs to channels_last or channels_last_3d
                based on model dimensionality for better convolution throughput.
            loss_config: Optional LossConfig for multi-loss training
            scheduler: Optional main learning rate scheduler
            warmup_scheduler: Optional warmup learning rate scheduler
            warmup_steps: Number of warmup steps (must match scheduler if provided)
            log_every_steps: Log metrics every N steps
            nan_threshold: Maximum allowed NaN rate (0.1 = 10%)
            enable_compile_train_step: Whether to compile the model with torch.compile
            enable_gradient_checkpointing: Whether to enable gradient checkpointing on
                encoder/decoder blocks (trades compute for memory)
            disc_optimizer: Optional separate optimizer for discriminator parameters
                (used with VQGANLoss/VAEGANLoss when GAN training is enabled)
            model_ema_decay: Optional EMA decay rate for model weights (e.g., 0.9999).
                When set, maintains an exponential moving average of model parameters.
                EMA weights are used for validation and saved alongside checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.loss_fn = loss_fn or Combined()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.callbacks = callbacks or [Logger()]
        self.channels_last = channels_last
        self.loss_config = loss_config or LossConfig()
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.warmup_steps = warmup_steps
        self.log_every_steps = log_every_steps
        self.nan_threshold = nan_threshold
        self.enable_compile_train_step = enable_compile_train_step
        self.disc_optimizer = disc_optimizer
        self.model_ema_decay = model_ema_decay
        self._ema_state: Optional[dict[str, torch.Tensor]] = None

        self.current_epoch = 0
        self.stop_training = False
        self.nan_tracker = NaNTracker(threshold=nan_threshold)
        self.train_logger = MetricsLogger(accelerator=accelerator, prefix="train/")
        self.val_logger = MetricsLogger(accelerator=accelerator, prefix="val/")
        self.global_step = 0

        # Stored tensors for discriminator step (set during train_step)
        self._last_images = None
        self._last_reconstruction = None

        # Register schedulers for checkpointing if provided
        if scheduler is not None:
            accelerator.register_for_checkpointing(scheduler)
        if warmup_scheduler is not None:
            accelerator.register_for_checkpointing(warmup_scheduler)

        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        if enable_compile_train_step and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="default", dynamic=True)

    def _is_compound_loss(self) -> bool:
        """Check if loss_fn is a compound loss (VQGANLoss or VAEGANLoss)."""
        from .losses.compound import VAEGANLoss, VQGANLoss

        return isinstance(self.loss_fn, (VQGANLoss, VAEGANLoss))

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        for module in model.modules():
            if hasattr(module, "use_checkpointing"):
                module.use_checkpointing = True

    def init_trackers(self, project_name: str, config: dict = None, **kwargs):
        """Initialize Accelerate trackers (e.g., wandb).

        Args:
            project_name: Name of project
            config: Optional configuration to log
            **kwargs: Additional arguments passed to accelerator.init_trackers
        """
        if config is None:
            config = {}
        self.accelerator.init_trackers(project_name, config=config, **kwargs)

    def _log(self, message: str):
        """Log a message via accelerator.print."""
        self.accelerator.print(message)

    def end_training(self):
        """End training and wait for all processes."""
        self.accelerator.end_training()

    def _run_callbacks(self, event: str, **kwargs):
        """Run all callbacks for a given event."""
        for callback in self.callbacks:
            getattr(callback, event)(self, **kwargs)

    def _compute_loss(
        self, images: torch.Tensor, output: dict
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute loss from model output, dispatching to the appropriate loss function.

        Handles Combined, CombinedPerceptual, VQGANLoss, and VAEGANLoss.

        Args:
            images: Original input images
            output: Model output (dict or namedtuple)

        Returns:
            Tuple of (loss_tensor, loss_dict, reconstruction_tensor)
        """
        if isinstance(output, dict):
            reconstruction = output["reconstructions"]
            quant_loss = output.get("quant_loss")
            kl_loss = output.get("kl_loss")
            posteriors = output.get("posteriors")
        else:
            reconstruction = output.reconstructions
            quant_loss = getattr(output, "quant_loss", None)
            kl_loss = getattr(output, "kl_loss", None)
            posteriors = getattr(output, "posteriors", None)

        from .losses.basic import CombinedPerceptual
        from .losses.compound import VAEGANLoss, VQGANLoss

        if isinstance(self.loss_fn, VAEGANLoss):
            loss, loss_dict = self.loss_fn.generator_step(
                images, reconstruction, kl_loss=kl_loss, posteriors=posteriors
            )
        elif isinstance(self.loss_fn, VQGANLoss):
            loss, loss_dict = self.loss_fn.generator_step(
                images, reconstruction, quant_loss=quant_loss
            )
        elif isinstance(self.loss_fn, (Combined, CombinedPerceptual)):
            aux_loss = quant_loss if quant_loss is not None else kl_loss
            loss, loss_dict = self.loss_fn(reconstruction, images, aux_loss)
            if kl_loss is not None:
                loss_dict["kl"] = kl_loss.mean().item()
        else:
            loss = self.loss_fn(reconstruction, images)
            loss_dict = {"total": loss.item()}

        return loss, loss_dict, reconstruction

    def train_step(self, batch, sync_gradients: bool = True) -> tuple[float, dict]:
        """Single training step.

        Args:
            batch: Input batch (tensor or dict with "image" key)
            sync_gradients: If False and model supports no_sync(), skip gradient
                all-reduce (used for gradient accumulation intermediate steps).
                Default True syncs every step.

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics
        """
        # Handle dict batches from dataloaders
        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(self.accelerator.device, non_blocking=True)
        if self.channels_last:
            mem_format = (
                torch.channels_last_3d if images.dim() == 5 else torch.channels_last
            )
            images = images.contiguous(memory_format=mem_format)

        if not sync_gradients and hasattr(self.model, "no_sync"):
            ctx = self.model.no_sync()
        else:
            ctx = nullcontext()

        with ctx:
            output = self.model(images)
            loss, loss_dict, reconstruction = self._compute_loss(images, output)

            # NaN guarding: detect NaN in reconstruction/aux-loss and total loss.
            # If found, skip backward + optimizer step so NaN never enters weights.
            aux_loss = self._aux_loss_tensor(output, reconstruction)
            nan_detected = self.nan_tracker.check_outputs(
                reconstruction, aux_loss
            ) or self.nan_tracker.check_loss(loss)

            if nan_detected:
                self._log(
                    f"NaN detected at step {self.global_step}; skipping batch "
                    f"(skipped={self.nan_tracker.skipped_batches})."
                )
                self._last_images = None
                self._last_reconstruction = None
                skipped_dict = dict.fromkeys(loss_dict, 0.0)
                skipped_dict["skipped"] = 1.0
                return 0.0, skipped_dict

            self.accelerator.backward(loss / self.gradient_accumulation_steps)

        # Store detached tensors for discriminator step
        if self.disc_optimizer is not None:
            self._last_images = images.detach()
            self._last_reconstruction = reconstruction.detach()

        return loss.item(), loss_dict

    def _aux_loss_tensor(self, output, reconstruction: torch.Tensor) -> torch.Tensor:
        """Extract an auxiliary-loss tensor for NaN checking.

        Returns quant_loss for discrete tokenizers, kl_loss for VAEs, or a
        zero tensor (matching reconstruction's device/dtype) otherwise.
        """
        if isinstance(output, dict):
            aux = output.get("quant_loss")
            if aux is None:
                aux = output.get("kl_loss")
        else:
            aux = getattr(output, "quant_loss", None)
            if aux is None:
                aux = getattr(output, "kl_loss", None)

        if aux is None:
            return torch.zeros(
                (), device=reconstruction.device, dtype=reconstruction.dtype
            )
        # Reduce to a scalar so torch.isnan(...) inside NaNTracker.check_outputs
        # behaves correctly for multi-element quant/kl loss tensors.
        return aux.sum()

    def _disc_train_step(self) -> tuple[float, dict]:
        """Discriminator training step (called after generator optimizer step).

        Uses stored images/reconstruction from the most recent train_step.

        Returns:
            disc_loss: Scalar discriminator loss value
            disc_dict: Dictionary of discriminator metrics
        """
        if self._last_images is None or self._last_reconstruction is None:
            return 0.0, {}

        if not getattr(self.loss_fn, "has_discriminator", False):
            return 0.0, {}

        self.disc_optimizer.zero_grad(set_to_none=True)

        disc_loss, disc_dict = self.loss_fn.discriminator_step(
            self._last_images, self._last_reconstruction
        )

        if disc_loss.item() == 0.0:
            return 0.0, disc_dict

        self.accelerator.backward(disc_loss)

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.loss_fn.discriminator.parameters(), self.max_grad_norm
            )

        self.disc_optimizer.step()

        return disc_loss.item(), disc_dict

    def _update_weights(self):
        """Update model weights after gradient accumulation."""
        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Update EMA weights
        if self.model_ema_decay is not None and self._ema_state is not None:
            decay = self.model_ema_decay
            for name, param in self.model.named_parameters():
                if name in self._ema_state:
                    self._ema_state[name].lerp_(param.data, 1 - decay)

        if self.warmup_scheduler is not None and self.global_step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()

    def _swap_ema_weights(self) -> Optional[dict[str, torch.Tensor]]:
        """Swap model weights with EMA weights. Returns original weights for restoring."""
        if self._ema_state is None:
            return None
        original = {}
        for name, param in self.model.named_parameters():
            if name in self._ema_state:
                original[name] = param.data.clone()
                param.data.copy_(self._ema_state[name])
        return original

    def _restore_weights(self, original: dict[str, torch.Tensor]) -> None:
        """Restore model weights from saved original state."""
        for name, param in self.model.named_parameters():
            if name in original:
                param.data.copy_(original[name])

    @torch.no_grad()
    def validate(
        self, val_loader: DataLoader, max_batches: Optional[int] = None
    ) -> dict:
        """Run validation.

        Args:
            val_loader: Validation data loader
            max_batches: Optional maximum number of batches to validate on.
                If None, validates on the entire dataset.

        Returns:
            metrics: Dictionary of validation metrics
        """
        # Swap to EMA weights for validation
        original_weights = self._swap_ema_weights()

        self.model.eval()
        self._run_callbacks("on_validation_begin")

        total_loss = 0.0
        total_samples = 0
        metrics_sum = {}
        first_batch = None

        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if first_batch is None:
                first_batch = batch
            # Handle dict batches from dataloaders
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.accelerator.device, non_blocking=True)
            batch_size = images.shape[0]

            output = self.model(images)
            loss, loss_dict, _ = self._compute_loss(images, output)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for key, value in loss_dict.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value * batch_size

        metrics = {f"val_{k}": v / total_samples for k, v in metrics_sum.items()}

        self._run_callbacks("on_validation_end", metrics=metrics, batch=first_batch)
        self.model.train()

        # Restore original (non-EMA) weights for continued training
        if original_weights is not None:
            self._restore_weights(original_weights)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        steps_per_epoch: Optional[int] = None,
        val_interval: int = 1,
        resume: Optional[str] = None,
        max_val_batches: Optional[int] = None,
    ):
        """Train the model.

        Args:
            train_loader: Training data loader
            epochs: Number of epochs to train
            val_loader: Optional validation data loader
            steps_per_epoch: Optional number of steps per epoch (for infinite dataloaders)
            val_interval: Run validation every N epochs
            resume: Optional checkpoint directory path to resume from
            max_val_batches: Optional maximum batches for validation (None = full dataset)
        """
        if resume is not None:
            self.load_checkpoint(resume)

        # Prepare model, optimizer, dataloaders, schedulers
        # This is the ONLY place accelerator.prepare() is called
        model, optimizer, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader
        )
        self.model = model
        self.optimizer = optimizer

        self.loss_fn = self.loss_fn.to(self.accelerator.device)

        # Prepare discriminator optimizer if present
        if self.disc_optimizer is not None:
            self.disc_optimizer = self.accelerator.prepare(self.disc_optimizer)

        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        # Initialize EMA state (after accelerator.prepare so weights are on device)
        if self.model_ema_decay is not None and self._ema_state is None:
            self._ema_state = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }
            self._log(f"Model EMA initialized (decay={self.model_ema_decay})")

        self._run_callbacks("on_train_begin")

        # Use self.current_epoch from checkpoint if resuming, otherwise start from 0
        start_epoch = self.current_epoch + 1 if resume else 0

        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch
            self.model.train()
            self._run_callbacks("on_epoch_begin", epoch=epoch)

            # KL annealing for VAEGANLoss
            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch)
                from .losses.compound import VAEGANLoss

                if isinstance(self.loss_fn, VAEGANLoss) and (
                    epoch == start_epoch
                    or (
                        self.loss_fn.kl_warmup_epochs > 0
                        and epoch <= self.loss_fn.kl_warmup_epochs
                    )
                ):
                    self._log(
                        f"Epoch {epoch}: kl_weight={self.loss_fn.kl_weight:.2e} "
                        f"(target={self.loss_fn._kl_weight_target:.2e})"
                    )

            if self.loss_config is not None and hasattr(self.loss_fn, "set_weights"):
                vgg_w, gram_w, stage = self.loss_config.compute_stage_weights(epoch)
                self.loss_fn.set_weights(vgg_w, gram_w)
                if epoch == start_epoch or epoch == self.loss_config.stage1_epochs:
                    self._log(
                        f"Loss stage {stage}: vgg_weight={vgg_w:.4f}, gram_weight={gram_w:.4f}"
                    )

            epoch_metrics = {"epoch": epoch}
            total_loss = 0.0
            num_batches = 0

            if steps_per_epoch is not None:
                train_iter = _infinite_loader(train_loader)
            else:
                train_iter = train_loader

            pbar = tqdm(
                train_iter,
                total=steps_per_epoch or len(train_loader),
                desc=f"Epoch {epoch}/{epochs}",
            )
            for batch_idx, batch in enumerate(pbar):
                if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                    break

                self._run_callbacks("on_batch_begin", batch=batch, batch_idx=batch_idx)

                is_sync_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                loss, loss_dict = self.train_step(batch, sync_gradients=is_sync_step)

                total_loss += loss
                num_batches += 1

                # Only step optimizer when gradients are synced
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self._update_weights()

                    # Discriminator step (after model optimizer step)
                    if self.disc_optimizer is not None:
                        disc_loss, disc_dict = self._disc_train_step()
                        loss_dict.update(disc_dict)

                    # global_step only increments on optimizer steps (sync_gradients=True)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1

                if self.global_step % self.log_every_steps == 0:
                    self.train_logger.log_step(loss_dict, step=self.global_step)

                pbar.set_postfix({"loss": f"{loss:.4f}", "step": self.global_step})

                self._run_callbacks(
                    "on_batch_end", batch=batch, batch_idx=batch_idx, loss=loss
                )

            epoch_metrics["train_loss"] = total_loss / num_batches

            if val_loader is not None and epoch % val_interval == 0:
                val_metrics = self.validate(val_loader, max_batches=max_val_batches)
                epoch_metrics.update(val_metrics)
                self.val_logger.log_step(val_metrics, step=self.global_step)

            self._run_callbacks("on_epoch_end", epoch=epoch, metrics=epoch_metrics)

        self._run_callbacks("on_train_end")

    def save_checkpoint(self, output_dir: str, metadata: dict = None):
        """Save training checkpoint.

        Saves Accelerate state (model, optimizer, schedulers) plus metadata.json
        containing epoch, global_step, warmup_steps, and scheduler state info.

        Args:
            output_dir: Directory path to save checkpoint (Accelerate format)
            metadata: Optional metadata to save (hparams, wandb_id, etc.)

        Raises:
            ValueError: If output_dir is not a directory path
        """
        checkpoint_dir = Path(output_dir)
        if not str(checkpoint_dir).endswith("/") and checkpoint_dir.suffix:
            raise ValueError(
                "Trainer now uses Accelerate directory format. "
                f"Expected a directory path, got: {output_dir}. "
                "Please specify a checkpoint directory (e.g., 'checkpoints/step_100/')"
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Accelerate saves model, optimizer, and registered schedulers
        self.accelerator.save_state(str(checkpoint_dir))

        # Determine current scheduler phase
        current_scheduler = "warmup" if self.global_step < self.warmup_steps else "main"

        # Build metadata with scheduler state
        metadata_dict = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "warmup_steps": self.warmup_steps,
            "current_scheduler": current_scheduler,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        if metadata is not None:
            metadata_dict.update(metadata)

        metadata_file = checkpoint_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Save EMA state alongside checkpoint
        if self._ema_state is not None:
            ema_path = checkpoint_dir / "ema_state.pt"
            torch.save(self._ema_state, str(ema_path))

    def load_checkpoint(self, input_dir: str):
        """Load training checkpoint.

        Loads Accelerate state (model, optimizer, schedulers) and metadata.json.
        Validates that warmup_steps matches the current config and warns on mismatch.

        Args:
            input_dir: Directory path to load checkpoint from (Accelerate format)

        Raises:
            ValueError: If input_dir is not a directory or is a legacy .pt file
        """
        checkpoint_path = Path(input_dir)

        # Check for legacy .pt file and raise clear error
        if checkpoint_path.suffix in (".pt", ".pth", ".ckpt"):
            raise ValueError(
                f"Legacy checkpoint file detected: {input_dir}\n"
                "Trainer now uses Accelerate directory checkpoints ONLY.\n"
                "Expected a checkpoint DIRECTORY containing:\n"
                "  - model weights (pytorch_model.bin or model.safetensors)\n"
                "  - optimizer state (optimizer.bin)\n"
                "  - scheduler states (scheduler.bin)\n"
                "  - metadata.json\n"
                "To convert old checkpoints, load the .pt file manually and save via Trainer.save_checkpoint()."
            )

        if not checkpoint_path.is_dir():
            raise ValueError(
                "Trainer now uses Accelerate directory format. "
                f"Expected a directory, got: {input_dir}. "
                "Please use a checkpoint directory with metadata.json"
            )

        # Load Accelerate state (model, optimizer, registered schedulers)
        self.accelerator.load_state(str(checkpoint_path))

        # Load metadata and restore training state
        metadata_file = checkpoint_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            self.current_epoch = metadata.get("epoch", 0)
            self.global_step = metadata.get("global_step", 0)

            # Validate warmup_steps config matches checkpoint
            saved_warmup_steps = metadata.get("warmup_steps")
            if (
                saved_warmup_steps is not None
                and saved_warmup_steps != self.warmup_steps
            ):
                self._log(
                    f"WARNING: warmup_steps mismatch! "
                    f"Checkpoint has warmup_steps={saved_warmup_steps}, "
                    f"but current config has warmup_steps={self.warmup_steps}. "
                    f"Scheduler behavior may be unexpected."
                )

            # Log resume info
            saved_scheduler = metadata.get("current_scheduler", "unknown")
            saved_lr = metadata.get("lr", "unknown")
            self._log(
                f"Resumed from checkpoint: epoch={self.current_epoch}, "
                f"global_step={self.global_step}, scheduler={saved_scheduler}, lr={saved_lr}"
            )
        else:
            self._log(
                f"WARNING: No metadata.json found in {input_dir}. "
                f"Training state (epoch, global_step) will start from 0."
            )

        # Load EMA state if present
        ema_path = checkpoint_path / "ema_state.pt"
        if ema_path.exists() and self.model_ema_decay is not None:
            self._ema_state = torch.load(
                str(ema_path),
                map_location=self.accelerator.device,
                weights_only=True,
            )
            self._log("Loaded EMA state from checkpoint")
