"""Tests for 3D TiTok tokenizer."""

import torch

from medtokenizers.networks import TiTokTokenizer


def _make_titok_3d(
    resolution: tuple[int, int, int] = (32, 32, 32),
) -> TiTokTokenizer:
    return TiTokTokenizer(
        dim=3,
        in_channels=1,
        out_channels=1,
        num_tokens=8,
        num_embeddings=16,
        embedding_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        patch_size=8,
        resolution=resolution,
    )


class TestTiTok3D:
    def test_encode_decode_shapes(self) -> None:
        tokenizer = _make_titok_3d().eval()
        x = torch.randn(2, 1, 32, 32, 32)

        with torch.no_grad():
            indices, quantized, loss = tokenizer.encode(x)
            recon = tokenizer.decode(quantized)

        assert indices.shape == (2, 8)
        assert quantized.shape == (2, 8, 16)
        assert loss.shape == (2, 1, 1)
        assert indices.dtype == torch.int32
        assert indices.min() >= 0
        assert indices.max() < tokenizer.num_embeddings
        assert recon.shape == x.shape

    def test_forward_is_finite(self) -> None:
        tokenizer = _make_titok_3d().eval()
        x = torch.randn(1, 1, 32, 32, 32)

        with torch.no_grad():
            output = tokenizer(x)

        assert output.reconstructions.shape == x.shape
        assert torch.isfinite(output.reconstructions).all()

    def test_tokenize_detokenize_shapes(self) -> None:
        tokenizer = _make_titok_3d().eval()
        x = torch.randn(1, 1, 32, 32, 32)

        with torch.no_grad():
            tokens = tokenizer.tokenize(x)
            recon = tokenizer.detokenize(tokens)

        assert tokens.shape == (1, 8)
        assert recon.shape == x.shape

    def test_invalid_resolution_raises(self) -> None:
        tokenizer = _make_titok_3d().eval()
        x = torch.randn(1, 1, 16, 32, 32)

        try:
            _ = tokenizer.encode(x)
        except ValueError as exc:
            assert "resolution" in str(exc)
        else:
            raise AssertionError("Expected ValueError for mismatched resolution.")

    def test_gradients_flow(self) -> None:
        tokenizer = _make_titok_3d()
        tokenizer.train()
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)

        output = tokenizer(x)
        output["reconstructions"].sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
