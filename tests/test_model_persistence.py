"""Tests for model saving, loading, and inference patterns."""

import tempfile
from pathlib import Path

import pytest
import torch

from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer


@pytest.fixture
def base_kwargs() -> dict:
    """Minimal kwargs for fast test models."""
    return {
        "in_channels": 1,
        "out_channels": 1,
        "channels": 16,  # Smaller for faster tests
        "channels_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],  # No attention for speed
        "dropout": 0.0,
        "resolution": 16,
        "spatial_compression": 4,
    }


class TestContinuousTokenizerPersistence:
    """Test save/load for ContinuousTokenizer."""

    def test_save_and_load_autoencoder(self, base_kwargs: dict) -> None:
        """Test saving and loading an autoencoder."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            z_factor=1,
            latent_channels=4,
            formulation="AE",
            **base_kwargs,
        ).to(torch.float32)

        x = torch.randn(1, 1, 16, 16, dtype=torch.float32)
        original_output = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save_pretrained(save_path)

            # Verify files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "pytorch_model.bin").exists()

            # Load and compare
            loaded = ContinuousTokenizer.from_pretrained(save_path)
            loaded_output = loaded(x)

            assert torch.allclose(
                original_output["reconstructions"],
                loaded_output["reconstructions"],
                atol=1e-5,
            )

    def test_save_and_load_vae(self, base_kwargs: dict) -> None:
        """Test saving and loading a VAE."""
        model = ContinuousTokenizer(
            dim=3,
            z_channels=4,
            embedding_dim=4,
            z_factor=2,
            latent_channels=4,
            formulation="VAE",
            **base_kwargs,
        ).to(torch.float32)

        x = torch.randn(1, 1, 16, 16, 16, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vae"
            model.save_pretrained(save_path)

            loaded = ContinuousTokenizer.from_pretrained(save_path)

            # Check config preserved - check forward pass works
            loaded_output = loaded(x)
            assert loaded_output["reconstructions"].shape == x.shape


class TestDiscreteTokenizerPersistence:
    """Test save/load for DiscreteTokenizer."""

    def test_save_and_load_vq(self, base_kwargs: dict) -> None:
        """Test saving and loading VQ tokenizer."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="VQ",
            num_embeddings=16,
            levels=[2] * 4,
            num_quantizers=1,
            **base_kwargs,
        ).to(torch.float32)

        x = torch.randn(1, 1, 16, 16, dtype=torch.float32)
        indices_orig, _, _ = model.encode(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vq"
            model.save_pretrained(save_path)

            loaded = DiscreteTokenizer.from_pretrained(save_path)
            indices_loaded, _, _ = loaded.encode(x)

            # Same indices for same input
            assert torch.equal(indices_orig, indices_loaded)

    def test_save_and_load_fsq(self, base_kwargs: dict) -> None:
        """Test saving and loading FSQ tokenizer."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[5, 5, 5, 5],
            num_embeddings=16,  # Ignored for FSQ
            num_quantizers=1,
            **base_kwargs,
        ).to(torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "fsq"
            model.save_pretrained(save_path)

            loaded = DiscreteTokenizer.from_pretrained(save_path)
            assert loaded.quantizer_type == "FSQ"


class TestTokenizeDetokenize:
    """Test tokenize/detokenize interface."""

    def test_tokenize_detokenize_round_trip(self, base_kwargs: dict) -> None:
        """Test that tokenize->detokenize produces valid reconstruction."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[5, 5, 5, 5],
            num_embeddings=16,
            num_quantizers=1,
            **base_kwargs,
        ).to(torch.float32)

        x = torch.randn(2, 1, 16, 16, dtype=torch.float32)

        tokens = model.tokenize(x)
        reconstructed = model.detokenize(tokens)

        # Check shapes
        assert reconstructed.shape == x.shape
        assert tokens.ndim == 3  # (batch, spatial_h, spatial_w) or flat

    def test_tokenize_produces_integers(self, base_kwargs: dict) -> None:
        """Test that tokenize produces integer indices."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="VQ",
            num_embeddings=32,
            levels=[2] * 4,
            num_quantizers=1,
            **base_kwargs,
        ).to(torch.float32)

        x = torch.randn(1, 1, 16, 16, dtype=torch.float32)
        tokens = model.tokenize(x)

        # Should be integer type
        assert tokens.dtype in (torch.int32, torch.int64, torch.long)
        # Should be in valid range
        assert tokens.min() >= 0


class TestEncodeBatch:
    """Test batch encoding functionality."""

    def test_encode_batch_matches_single(self, base_kwargs: dict) -> None:
        """Test that encode_batch produces same results as individual encoding."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            z_factor=1,
            latent_channels=4,
            formulation="AE",
            **base_kwargs,
        ).to(torch.float32)
        model.train(False)  # Set to inference mode

        x = torch.randn(4, 1, 16, 16, dtype=torch.float32)

        # Encode all at once
        with torch.inference_mode():
            all_at_once, _ = model.encode(x)

        # Encode in batches
        batched = model.encode_batch(x, batch_size=2)

        assert torch.allclose(all_at_once, batched, atol=1e-5)

    def test_encode_batch_with_odd_samples(self, base_kwargs: dict) -> None:
        """Test encode_batch handles non-divisible sample counts."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            z_factor=1,
            latent_channels=4,
            formulation="AE",
            **base_kwargs,
        ).to(torch.float32)
        model.train(False)  # Set to inference mode

        x = torch.randn(5, 1, 16, 16, dtype=torch.float32)  # 5 not divisible by 2
        batched = model.encode_batch(x, batch_size=2)

        assert batched.shape[0] == 5


class TestLoadErrors:
    """Test error handling for model loading."""

    def test_from_pretrained_nonexistent_path(self) -> None:
        """Test that loading from nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Could not find model"):
            ContinuousTokenizer.from_pretrained("/nonexistent/path/model")

    def test_save_creates_directory(self, base_kwargs: dict) -> None:
        """Test that save_pretrained creates parent directories."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            z_factor=1,
            latent_channels=4,
            formulation="AE",
            **base_kwargs,
        ).to(torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dirs" / "model"
            model.save_pretrained(save_path)

            assert (save_path / "config.json").exists()
            assert (save_path / "pytorch_model.bin").exists()
