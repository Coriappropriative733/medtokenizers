"""Tests for TiTok 1D tokenizers (2D inputs)."""

import torch

from medtokenizers.networks import TiTokTokenizer


def _make_titok_2d(resolution: int = 64) -> TiTokTokenizer:
    return TiTokTokenizer(
        dim=2,
        in_channels=1,
        out_channels=1,
        num_tokens=8,
        num_embeddings=16,
        embedding_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        patch_size=16,
        resolution=resolution,
    )


class TestTiTok2D:
    def test_encode_shape_and_range(self) -> None:
        tokenizer = _make_titok_2d(resolution=64).eval()
        x = torch.randn(2, 1, 64, 64)

        with torch.no_grad():
            indices, quantized, loss = tokenizer.encode(x)

        assert indices.shape == (2, 8)
        assert quantized.shape == (2, 8, 16)
        assert loss.shape == (2, 1, 1)
        assert indices.dtype == torch.int32
        assert indices.min() >= 0
        assert indices.max() < tokenizer.num_embeddings

    def test_tokenize_decode_shapes(self) -> None:
        tokenizer = _make_titok_2d(resolution=64).eval()
        x = torch.randn(2, 1, 64, 64)

        with torch.no_grad():
            tokens = tokenizer.tokenize(x)
            recon = tokenizer.detokenize(tokens)

        assert tokens.shape == (2, 8)
        assert recon.shape == (2, 1, 64, 64)

    def test_forward_shape_and_finite(self) -> None:
        tokenizer = _make_titok_2d(resolution=64).eval()
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = tokenizer(x)

        assert output.reconstructions.shape == x.shape
        assert torch.isfinite(output.reconstructions).all()

    def test_invalid_resolution_raises(self) -> None:
        tokenizer = _make_titok_2d(resolution=64).eval()
        x = torch.randn(1, 1, 32, 64)

        try:
            _ = tokenizer.encode(x)
        except ValueError as exc:
            assert "resolution" in str(exc)
        else:
            raise AssertionError("Expected ValueError for mismatched resolution.")

    def test_gradients_flow(self) -> None:
        tokenizer = _make_titok_2d(resolution=64)
        tokenizer.train()

        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        output = tokenizer(x)
        output["reconstructions"].sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


class TestTiTok2DDetokenize:
    def test_encode_decode_shapes(self) -> None:
        tokenizer = _make_titok_2d(resolution=64).eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            tokens = tokenizer.tokenize(x)
            recon = tokenizer.detokenize(tokens)

        assert tokens.shape == (1, 8)
        assert recon.shape == x.shape
