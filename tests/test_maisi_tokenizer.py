"""Tests for :class:`MAISITokenizer`.

The published NVIDIA MAISI weights are distributed under a separate license and
are not available in CI, so these tests exercise the architecture with a small,
randomly-initialised config. They verify instantiation, the encode/decode shape
round-trip on a tiny 3D volume (CPU only), and ``from_pretrained`` error
handling for a missing path. Anything that genuinely needs the real pretrained
weights is gated behind the network marker / skip.
"""

import pytest
import torch

from medtokenizers.networks import MAISITokenizer
from medtokenizers.networks.continuous import NetworkEval


@pytest.fixture
def small_config() -> dict:
    """A tiny, random-init override of ``MAISITokenizer.DEFAULT_CONFIG``.

    Keeps the MAISI architecture (3D VAE, separate quant convs, etc.) while
    shrinking channels/resolution so the model is cheap to build and run on CPU.
    """
    return {
        "channels": 8,
        "channels_mult": (1, 2),
        "num_res_blocks": 1,
        "z_channels": 4,
        "latent_channels": 4,
        "resolution": 32,
        "spatial_compression": 4,
        "decoder_blocks_per_stage": [1, 1, 0],
    }


def test_maisi_instantiation(small_config: dict) -> None:
    """A small MAISI tokenizer builds with MAISI defaults applied."""
    model = MAISITokenizer(**small_config)

    assert model.dim == 3
    assert model.formulation == "VAE"
    # MAISI uses separate mu/log-sigma quant convs.
    assert model.separate_quant_conv is True
    assert model.num_parameters() > 0


def test_maisi_default_config_values() -> None:
    """The MAISI-specific architecture defaults are preserved."""
    config = MAISITokenizer.DEFAULT_CONFIG

    assert config["dim"] == 3
    assert config["formulation"] == "VAE"
    assert config["use_encoder_mid"] is False
    assert config["decoder_blocks_per_stage"] == [2, 2, 0]


def test_maisi_encode_decode_round_trip(small_config: dict) -> None:
    """encode() then decode() returns the original spatial shape."""
    model = MAISITokenizer(**small_config).to(torch.float32)
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32)

    with torch.no_grad():
        latent, distribution_output = model.encode(x)
        recon = model.decode(latent)

    # 4x spatial compression on a 32^3 volume -> 8^3 latent grid.
    assert latent.shape == (1, model.latent_channels, 8, 8, 8)
    assert recon.shape == x.shape
    # VAE encode returns (kl_loss, (mu, log_sigma)).
    assert isinstance(distribution_output, tuple)
    kl_loss, posteriors = distribution_output
    assert kl_loss.ndim == 0
    assert posteriors[0].shape == latent.shape


def test_maisi_forward_eval_returns_namedtuple(small_config: dict) -> None:
    """Eval-mode forward returns a NetworkEval namedtuple with matching shapes."""
    model = MAISITokenizer(**small_config).to(torch.float32)
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, NetworkEval)
    assert output.reconstructions.shape == x.shape
    assert output.latent.shape == (1, model.latent_channels, 8, 8, 8)
    # VAE posteriors are (mean, logvar).
    assert output.posteriors[0].shape == output.latent.shape


def test_maisi_forward_train_returns_dict(small_config: dict) -> None:
    """Training-mode forward returns a dict with reconstructions + kl_loss."""
    model = MAISITokenizer(**small_config).to(torch.float32)
    model.train()

    x = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32)
    output = model(x)

    assert isinstance(output, dict)
    assert output["reconstructions"].shape == x.shape
    assert "kl_loss" in output
    assert "latent" in output


def test_maisi_from_pretrained_missing_path_raises() -> None:
    """from_pretrained on a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        MAISITokenizer.from_pretrained("/nonexistent/maisi/checkpoint/path")


def test_maisi_pretrained_missing_path_raises(small_config: dict) -> None:
    """Constructing with a missing ``pretrained`` path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        MAISITokenizer(pretrained="/nonexistent/maisi/checkpoint/path", **small_config)


def test_maisi_get_training_config() -> None:
    """get_training_config exposes the documented MAISI hyperparameters."""
    config = MAISITokenizer.get_training_config()

    assert config["reconstruction_loss"] == "l1"
    assert config["kl_weight"] == pytest.approx(1e-7)
    assert config["perceptual_weight"] == pytest.approx(0.3)


@pytest.mark.skip(
    reason="Requires NVIDIA MAISI weights (NSCLv1 license), unavailable in CI. "
    "Convert with scripts/convert_maisi_to_hf.py and load locally to enable."
)
def test_maisi_load_real_pretrained_weights() -> None:  # pragma: no cover
    """Documents loading real converted MAISI weights (skipped in CI)."""
    model = MAISITokenizer.from_pretrained("weights/maisi_converted")
    assert model.formulation == "VAE"
