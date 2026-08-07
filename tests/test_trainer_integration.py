"""Integration tests for Accelerate-enabled Trainer."""

import tempfile
from pathlib import Path

import pytest
import torch

from medtokenizers.networks.discrete import DiscreteTokenizer
from medtokenizers.training.configs import LossConfig
from medtokenizers.training.metrics import MetricsLogger
from medtokenizers.training.nan_tracker import NaNTracker
from medtokenizers.training.trainer import Trainer

ACCELERATE_AVAILABLE = False
Accelerator = None

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def dummy_accelerator():
    """Create a dummy CPU Accelerator for testing."""
    if not ACCELERATE_AVAILABLE:
        pytest.skip("accelerate not installed")
    return Accelerator(device_placement=False, cpu=True)


@pytest.fixture
def simple_model():
    """Create a simple discrete tokenizer for testing."""
    return DiscreteTokenizer(
        dim=2,
        in_channels=1,
        out_channels=1,
        z_channels=8,
        embedding_dim=8,
        channels=32,
        channels_mult=[1, 1],
        num_res_blocks=1,
        resolution=32,
        quantizer_type="VQ",
    )


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer for testing."""
    return torch.optim.AdamW(simple_model.parameters(), lr=1e-4)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    dataset = torch.randn(10, 1, 32, 32)
    return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)


def test_loss_config():
    """Test LossConfig creation and stage weights."""
    config = LossConfig(
        l1_weight=1.0,
        quant_weight=0.5,
        vgg_weight=0.1,
        gram_weight=0.1,
        stage1_epochs=2,
        stage1_vgg_weight=0.0,
        stage1_gram_weight=0.0,
        stage2_vgg_weight=0.1,
        stage2_gram_weight=0.1,
        stage2_warmup_epochs=2,
    )

    vgg, gram, stage = config.compute_stage_weights(0)
    assert vgg == 0.0
    assert gram == 0.0
    assert stage == 1

    vgg, gram, stage = config.compute_stage_weights(2)
    assert vgg == 0.0
    assert gram == 0.0
    assert stage == 2

    vgg, gram, stage = config.compute_stage_weights(3)
    assert vgg == 0.05
    assert gram == 0.05
    assert stage == 2

    vgg, gram, stage = config.compute_stage_weights(4)
    assert vgg == 0.1
    assert gram == 0.1
    assert stage == 2


def test_nan_tracker():
    """Test NaNTracker."""
    tracker = NaNTracker(threshold=1.0)

    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert not tracker.check_outputs(tensor, torch.tensor(0.5))
    assert not tracker.check_loss(torch.tensor(1.0))

    assert tracker.check_outputs(torch.tensor([float("nan"), 2.0]), torch.tensor(0.5))
    assert tracker.recon_nan_count == 1
    assert tracker.total_batches == 2

    assert tracker.check_loss(torch.tensor(float("nan")))
    assert tracker.skipped_batches == 2

    stats = tracker.get_statistics()
    assert stats["recon_nan_count"] == 1
    assert stats["total_batches"] == 2
    assert stats["skipped_batches"] == 2

    tracker.reset()
    assert tracker.recon_nan_count == 0


def test_metrics_logger_no_accelerate():
    """Test MetricsLogger without Accelerate."""
    logger = MetricsLogger(prefix="train/")
    assert logger.should_log()

    logger.log_step({"loss": 0.5, "lr": 1e-4}, step=0)


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_metrics_logger_with_accelerate(dummy_accelerator):
    """Test MetricsLogger with Accelerate."""
    logger = MetricsLogger(accelerator=dummy_accelerator, prefix="train/")
    assert logger.should_log()

    logger.log_step({"loss": 0.5, "lr": 1e-4}, step=0)


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_trainer_initialization(simple_model, simple_optimizer, dummy_accelerator):
    """Test Trainer initialization with Accelerate."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
        log_every_steps=1,
        nan_threshold=0.1,
    )

    assert trainer.accelerator is dummy_accelerator
    assert trainer.loss_config.l1_weight == 1.0
    assert trainer.nan_tracker is not None
    assert trainer.train_logger is not None


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_trainer_train_step(simple_model, simple_optimizer, dummy_accelerator):
    """Test trainer train_step with Accelerate."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
    )

    batch = torch.randn(2, 1, 32, 32)
    loss, metrics = trainer.train_step(batch)

    assert isinstance(loss, float)
    assert isinstance(metrics, dict)


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_trainer_checkpoint_save_load_directory(
    simple_model, simple_optimizer, dummy_accelerator
):
    """Test checkpoint save/load with Accelerate directory format."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoint"
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.save_checkpoint(str(ckpt_dir), metadata={"test": "value"})

        metadata_file = ckpt_dir / "metadata.json"
        assert metadata_file.exists()

        new_trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            accelerator=dummy_accelerator,
            loss_config=loss_config,
        )
        new_trainer.load_checkpoint(str(ckpt_dir))

        assert new_trainer.current_epoch == 5
        assert new_trainer.global_step == 100


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_trainer_checkpoint_load_legacy_fails(
    simple_model, simple_optimizer, dummy_accelerator, simple_dataloader
):
    """Test that loading legacy .pt checkpoints raises clear error."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a legacy .pt checkpoint
        ckpt_file = Path(tmpdir) / "checkpoint.pt"
        torch.save(
            {"epoch": 5, "model_state_dict": simple_model.state_dict()}, ckpt_file
        )

        # Should raise ValueError with clear message
        with pytest.raises(ValueError) as exc_info:
            trainer.load_checkpoint(str(ckpt_file))

        assert "Accelerate directory checkpoints ONLY" in str(exc_info.value)


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_trainer_fit_2_steps(simple_model, simple_optimizer, dummy_accelerator):
    """Test trainer fit for 2 steps with Accelerate."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    dataset = torch.randn(4, 1, 32, 32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
        log_every_steps=1,
    )

    trainer.fit(dataloader, epochs=1, steps_per_epoch=2)

    assert trainer.current_epoch == 0
    assert trainer.global_step >= 1


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_global_step_increments_on_optimizer_steps(
    simple_model, simple_optimizer, dummy_accelerator
):
    """Test that global_step only increments when optimizer steps."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    dataset = torch.randn(8, 1, 32, 32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
        log_every_steps=1,
        gradient_accumulation_steps=2,
    )

    # With 4 batches and gradient_accumulation_steps=2,
    # optimizer should step 2 times per epoch
    trainer.fit(dataloader, epochs=1, steps_per_epoch=4)

    # global_step should equal number of optimizer steps (4 // 2 = 2)
    assert trainer.global_step == 2


@pytest.mark.skipif(not ACCELERATE_AVAILABLE, reason="accelerate not installed")
def test_warmup_scheduler_registration(
    simple_model, simple_optimizer, dummy_accelerator
):
    """Test that warmup_scheduler is registered for checkpointing."""
    loss_config = LossConfig(l1_weight=1.0, quant_weight=0.5)
    dataset = torch.randn(4, 1, 32, 32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        simple_optimizer, start_factor=1e-8, end_factor=1.0, total_iters=5
    )

    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        accelerator=dummy_accelerator,
        loss_config=loss_config,
        warmup_scheduler=warmup_scheduler,
        warmup_steps=5,
    )

    # Fit for a few steps to verify warmup_scheduler works
    trainer.fit(dataloader, epochs=1, steps_per_epoch=2)

    assert trainer.global_step >= 1
