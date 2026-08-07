"""Tests for training callbacks."""

import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from medtokenizers.training.callbacks import (
    Callback,
    Checkpoint,
    EarlyStopping,
    Logger,
    LRScheduler,
    ReconstructionLogger,
)


class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self):
        self.model = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.stop_training = False


class MockAccelerator:
    """Minimal accelerator mock for ReconstructionLogger tests."""

    def __init__(self):
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.logged = []

    def log(self, payload, step=0):
        self.logged.append((payload, step))


class EchoReconstructionModel(nn.Module):
    """Model that returns identity reconstructions."""

    def forward(self, x):
        return {"reconstructions": x.clone()}


class TestCallback:
    """Tests for base Callback class."""

    def test_base_callback_methods(self):
        callback = Callback()
        trainer = MockTrainer()

        # All methods should be callable and do nothing by default
        callback.on_train_begin(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_begin(trainer, epoch=0)
        callback.on_epoch_end(trainer, epoch=0, metrics={})
        callback.on_batch_begin(trainer, batch=None, batch_idx=0)
        callback.on_batch_end(trainer, batch=None, batch_idx=0, loss=0.0)
        callback.on_validation_begin(trainer)
        callback.on_validation_end(trainer, metrics={})

        # No assertions needed - just checking methods exist and don't crash


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_stops_when_no_improvement(self):
        callback = EarlyStopping(patience=3)
        trainer = MockTrainer()

        # Loss doesn't improve
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        assert not trainer.stop_training

        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})
        assert not trainer.stop_training

        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 1.2})
        assert not trainer.stop_training

        callback.on_epoch_end(trainer, epoch=3, metrics={"val_loss": 1.3})
        assert trainer.stop_training
        assert callback.stopped_epoch == 3

    def test_resets_wait_on_improvement(self):
        callback = EarlyStopping(patience=3)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})
        callback.on_epoch_end(
            trainer, epoch=2, metrics={"val_loss": 0.9}
        )  # Improvement
        assert callback.wait == 0
        assert not trainer.stop_training

    def test_min_delta(self):
        callback = EarlyStopping(patience=2, min_delta=0.1)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        callback.on_epoch_end(
            trainer, epoch=1, metrics={"val_loss": 0.95}
        )  # < min_delta
        assert callback.wait == 1

        callback.on_epoch_end(
            trainer, epoch=2, metrics={"val_loss": 0.85}
        )  # >= min_delta
        assert callback.wait == 0

    def test_custom_monitor(self):
        # EarlyStopping always treats lower as better
        # Test with a custom metric name
        callback = EarlyStopping(patience=2, monitor="custom_metric")
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"custom_metric": 1.0})
        assert not trainer.stop_training

        callback.on_epoch_end(trainer, epoch=1, metrics={"custom_metric": 1.1})
        assert not trainer.stop_training

        callback.on_epoch_end(trainer, epoch=2, metrics={"custom_metric": 1.2})
        # After patience=2 epochs without improvement (metric increasing), should stop
        assert trainer.stop_training

    def test_missing_metric(self):
        callback = EarlyStopping(patience=2)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={})
        callback.on_epoch_end(trainer, epoch=1, metrics={})
        assert not trainer.stop_training  # Should not crash

    def test_first_epoch_sets_best(self):
        callback = EarlyStopping(patience=2)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        assert callback.best_value == 1.0
        assert callback.wait == 0


class TestCheckpoint:
    """Tests for Checkpoint callback."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_saves_checkpoint(self, temp_dir):
        filepath = temp_dir / "model.pt"
        callback = Checkpoint(str(filepath), save_best_only=False, verbose=False)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})

        checkpoint_path = temp_dir / "model_epoch0.pt"
        assert checkpoint_path.exists()

    def test_save_best_only_min(self, temp_dir):
        filepath = temp_dir / "model.pt"
        callback = Checkpoint(
            str(filepath), save_best_only=True, mode="min", verbose=False
        )
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        assert (temp_dir / "model_epoch0.pt").exists()

        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})
        assert not (temp_dir / "model_epoch1.pt").exists()  # Worse, not saved

        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 0.9})
        assert (temp_dir / "model_epoch2.pt").exists()  # Better, saved

    def test_save_best_only_max(self, temp_dir):
        filepath = temp_dir / "model.pt"
        callback = Checkpoint(
            str(filepath),
            save_best_only=True,
            mode="max",
            monitor="val_accuracy",
            verbose=False,
        )
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_accuracy": 0.8})
        assert (temp_dir / "model_epoch0.pt").exists()

        callback.on_epoch_end(trainer, epoch=1, metrics={"val_accuracy": 0.7})
        assert not (temp_dir / "model_epoch1.pt").exists()  # Worse, not saved

        callback.on_epoch_end(trainer, epoch=2, metrics={"val_accuracy": 0.9})
        assert (temp_dir / "model_epoch2.pt").exists()  # Better, saved

    def test_checkpoint_content(self, temp_dir):
        filepath = temp_dir / "model.pt"
        callback = Checkpoint(str(filepath), save_best_only=False, verbose=False)
        trainer = MockTrainer()

        metrics = {"val_loss": 1.0, "train_loss": 0.5}
        callback.on_epoch_end(trainer, epoch=5, metrics=metrics)

        checkpoint_path = temp_dir / "model_epoch5.pt"
        checkpoint = torch.load(checkpoint_path)

        assert checkpoint["epoch"] == 5
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["metrics"] == metrics

    def test_creates_parent_directory(self, temp_dir):
        filepath = temp_dir / "nested" / "dir" / "model.pt"
        callback = Checkpoint(str(filepath), verbose=False)

        assert filepath.parent.exists()

    def test_missing_metric(self, temp_dir):
        filepath = temp_dir / "model.pt"
        callback = Checkpoint(str(filepath), verbose=False)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={})

        # Should not crash, and should not save
        assert not (temp_dir / "model_epoch0.pt").exists()


class TestLRScheduler:
    """Tests for LRScheduler callback."""

    def test_step_lr_scheduler(self):
        trainer = MockTrainer()
        # StepLR reduces LR every step_size steps
        scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer, step_size=1, gamma=0.5
        )
        callback = LRScheduler(scheduler)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # First epoch - LR changes after first step
        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=0, metrics={})
        lr_after_step = trainer.optimizer.param_groups[0]["lr"]

        # LR should have been reduced
        assert lr_after_step < initial_lr
        assert lr_after_step == initial_lr * 0.5

    def test_reduce_on_plateau(self):
        trainer = MockTrainer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode="min", factor=0.5, patience=1
        )
        callback = LRScheduler(scheduler)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # No improvement for patience epochs
        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})

        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})

        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 1.2})

        # LR should be reduced after patience epochs
        assert trainer.optimizer.param_groups[0]["lr"] < initial_lr

    def test_reduce_on_plateau_missing_metric(self):
        trainer = MockTrainer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
        callback = LRScheduler(scheduler)

        # Should not crash when val_loss is missing
        callback.on_epoch_end(trainer, epoch=0, metrics={})

    def test_cosine_annealing(self):
        trainer = MockTrainer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        callback = LRScheduler(scheduler)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=0, metrics={})
        lr_after_1 = trainer.optimizer.param_groups[0]["lr"]

        trainer.optimizer.step()
        callback.on_epoch_end(trainer, epoch=1, metrics={})
        lr_after_2 = trainer.optimizer.param_groups[0]["lr"]

        # LR should be decreasing with cosine schedule
        assert lr_after_1 < initial_lr
        assert lr_after_2 < lr_after_1


class TestLogger:
    """Tests for Logger callback."""

    def test_batch_logging(self, capsys):
        callback = Logger(print_every=2)
        trainer = MockTrainer()

        callback.on_batch_end(trainer, batch=None, batch_idx=0, loss=1.0)
        captured = capsys.readouterr()
        assert captured.out == ""  # Not printed yet

        callback.on_batch_end(trainer, batch=None, batch_idx=1, loss=0.9)
        captured = capsys.readouterr()
        assert "Batch 2" in captured.out
        assert "0.9500" in captured.out

    def test_epoch_logging(self, capsys):
        callback = Logger(print_every=10)
        trainer = MockTrainer()

        metrics = {"train_loss": 0.5, "val_loss": 0.6}
        callback.on_epoch_end(trainer, epoch=1, metrics=metrics)

        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out
        assert "train_loss" in captured.out
        assert "val_loss" in captured.out

    def test_batch_losses_reset(self):
        callback = Logger(print_every=2)
        trainer = MockTrainer()

        callback.on_batch_end(trainer, batch=None, batch_idx=0, loss=1.0)
        callback.on_batch_end(trainer, batch=None, batch_idx=1, loss=0.9)

        assert len(callback.batch_losses) == 2

        callback.on_epoch_end(trainer, epoch=0, metrics={})

        assert len(callback.batch_losses) == 0

    def test_average_calculation(self, capsys):
        callback = Logger(print_every=3)
        trainer = MockTrainer()

        callback.on_batch_end(trainer, batch=None, batch_idx=0, loss=1.0)
        callback.on_batch_end(trainer, batch=None, batch_idx=1, loss=2.0)
        callback.on_batch_end(trainer, batch=None, batch_idx=2, loss=3.0)

        captured = capsys.readouterr()
        # Average of [1.0, 2.0, 3.0] = 2.0
        assert "2.0000" in captured.out


class TestCallbackEdgeCases:
    """Edge case tests for callbacks."""

    def test_early_stopping_negative_patience(self):
        # Should handle edge case gracefully
        callback = EarlyStopping(patience=0)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        # With patience=0, should stop immediately on first non-improvement
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.0})
        assert trainer.stop_training

    def test_checkpoint_with_nan_metric(self, tmp_path):
        filepath = tmp_path / "model.pt"
        callback = Checkpoint(str(filepath), verbose=False)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": float("nan")})

        # Should handle NaN without crashing
        # Behavior may vary - this just ensures no crash

    def test_logger_empty_metrics(self, capsys):
        callback = Logger(print_every=1)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={})

        captured = capsys.readouterr()
        assert "Epoch 0" in captured.out

    def test_multiple_callbacks_together(self):
        """Test using multiple callbacks simultaneously."""
        trainer = MockTrainer()
        early_stop = EarlyStopping(patience=2)
        logger = Logger(print_every=1)

        metrics = {"val_loss": 1.0}

        early_stop.on_epoch_end(trainer, epoch=0, metrics=metrics)
        logger.on_epoch_end(trainer, epoch=0, metrics=metrics)

        assert not trainer.stop_training

        metrics = {"val_loss": 1.1}
        early_stop.on_epoch_end(trainer, epoch=1, metrics=metrics)
        logger.on_epoch_end(trainer, epoch=1, metrics=metrics)

        assert not trainer.stop_training

        metrics = {"val_loss": 1.2}
        early_stop.on_epoch_end(trainer, epoch=2, metrics=metrics)
        logger.on_epoch_end(trainer, epoch=2, metrics=metrics)

        assert trainer.stop_training

    def test_early_stopping_improvement_on_boundary(self):
        callback = EarlyStopping(patience=2, min_delta=0.0)
        trainer = MockTrainer()

        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        callback.on_epoch_end(
            trainer, epoch=1, metrics={"val_loss": 1.0}
        )  # Equal, not better
        assert callback.wait == 1


class TestReconstructionLogger:
    """Tests for ReconstructionLogger callback."""

    @pytest.fixture
    def fake_wandb(self, monkeypatch):
        class _FakeImage:
            def __init__(self, data, caption=None):
                self.data = data
                self.caption = caption

        fake_module = types.SimpleNamespace(Image=_FakeImage)
        monkeypatch.setitem(sys.modules, "wandb", fake_module)
        return fake_module

    def _make_trainer(self):
        trainer = types.SimpleNamespace()
        trainer.current_epoch = 0
        trainer.global_step = 7
        trainer.model = EchoReconstructionModel()
        trainer.accelerator = MockAccelerator()
        return trainer

    def test_reconstruction_logger_handles_2d_batches(self, fake_wandb):
        callback = ReconstructionLogger(num_samples=2, every_n_epochs=1)
        trainer = self._make_trainer()

        batch = torch.rand(2, 1, 64, 64)
        callback.on_validation_end(trainer, metrics={}, batch=batch)

        assert len(trainer.accelerator.logged) == 1
        payload, step = trainer.accelerator.logged[0]
        assert "reconstructions" in payload
        assert step == trainer.global_step
        assert len(payload["reconstructions"]) == 2

    def test_reconstruction_logger_handles_3d_batches(self, fake_wandb):
        callback = ReconstructionLogger(num_samples=1, every_n_epochs=1)
        trainer = self._make_trainer()

        batch = torch.rand(1, 1, 16, 32, 32)
        callback.on_validation_end(trainer, metrics={}, batch=batch)

        assert len(trainer.accelerator.logged) == 1
        payload, _ = trainer.accelerator.logged[0]
        assert len(payload["reconstructions"]) == 1
