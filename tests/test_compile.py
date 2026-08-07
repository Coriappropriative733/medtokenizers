"""Tests for torch.compile compatibility with fullgraph=True.

These tests verify that quantizers and tokenizers can be compiled with
fullgraph=True, which requires a single FX graph without graph breaks.
This is a strict gate - any graph breaks will cause compilation to fail.

The tests use model.eval() to test inference paths, which should be
fully compilable after the P0 guards are in place.
"""

import pytest
import torch

from medtokenizers.modules.quant import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)

# Skip all tests in this module if torch.compile is not available
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile not available in this PyTorch version",
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset dynamo cache before each test to avoid recompilation issues."""
    torch._dynamo.reset()
    yield


class TestQuantizerCompileFullgraph:
    """Test that quantizers compile with fullgraph=True in eval mode."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_fsq_compile_fullgraph(self, dim: int) -> None:
        """FSQuantizer should compile with fullgraph=True."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        quantizer.eval()

        compiled = torch.compile(quantizer, fullgraph=True)

        if dim == 2:
            x = torch.randn(1, 4, 8, 8)
        else:
            x = torch.randn(1, 4, 8, 8, 8)

        # Should not raise - if it does, there's a graph break
        codes, loss, indices = compiled(x)

        assert codes.shape == x.shape
        assert indices.shape == (1,) + x.shape[2:]

    @pytest.mark.parametrize("dim", [2, 3])
    def test_resfsq_compile_fullgraph(self, dim: int) -> None:
        """ResidualFSQuantizer should compile with fullgraph=True."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 8, 8, 8], num_quantizers=2, embedding_dim=4
        )
        quantizer.eval()

        compiled = torch.compile(quantizer, fullgraph=True)

        if dim == 2:
            x = torch.randn(1, 4, 8, 8)
        else:
            x = torch.randn(1, 4, 8, 8, 8)

        # Should not raise - if it does, there's a graph break
        codes, loss, indices = compiled(x)

        assert codes.shape == x.shape
        assert indices.shape == (1, 2) + x.shape[2:]  # (B, num_quantizers, *spatial)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_vq_compile_fullgraph(self, dim: int) -> None:
        """VectorQuantizer should compile with fullgraph=True in eval mode.

        Note: VQ with use_ema=True has EMA updates in training mode that are
        guarded with is_compiling(). In eval mode, these don't run.
        """
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=4, dim=dim, use_ema=False
        )
        quantizer.eval()

        compiled = torch.compile(quantizer, fullgraph=True)

        if dim == 2:
            x = torch.randn(1, 4, 8, 8)
        else:
            x = torch.randn(1, 4, 8, 8, 8)

        # Should not raise - if it does, there's a graph break
        codes, loss, indices = compiled(x)

        assert codes.shape == x.shape

    @pytest.mark.parametrize("dim", [2, 3])
    def test_vq_ema_compile_fullgraph_eval(self, dim: int) -> None:
        """VectorQuantizer with EMA should compile in eval mode.

        The EMA updates are training-only and guarded with is_compiling(),
        so eval mode should have no graph breaks.
        """
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=4, dim=dim, use_ema=True
        )
        quantizer.eval()

        compiled = torch.compile(quantizer, fullgraph=True)

        if dim == 2:
            x = torch.randn(1, 4, 8, 8)
        else:
            x = torch.randn(1, 4, 8, 8, 8)

        # Should not raise in eval mode
        codes, loss, indices = compiled(x)

        assert codes.shape == x.shape

    @pytest.mark.parametrize("dim", [2, 3])
    def test_lfq_compile_fullgraph(self, dim: int) -> None:
        """LFQuantizer should compile with fullgraph=True."""
        quantizer = LFQuantizer(
            codebook_size=16, codebook_dim=4, dim=dim, entropy_loss=False
        )
        quantizer.eval()

        compiled = torch.compile(quantizer, fullgraph=True)

        if dim == 2:
            x = torch.randn(1, 4, 8, 8)
        else:
            x = torch.randn(1, 4, 8, 8, 8)

        # Should not raise - if it does, there's a graph break
        codes, loss, indices = compiled(x)

        assert codes.shape == x.shape


class TestQuantizerReturnOrder:
    """Test that all quantizers return (codes, loss, indices) consistently."""

    def test_fsq_return_order(self) -> None:
        """FSQuantizer should return (codes, loss, indices)."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8)

        result = quantizer(x)
        codes, loss, indices = result

        # codes should be float, same shape as input
        assert codes.dtype.is_floating_point
        assert codes.shape == x.shape

        # loss should be float
        assert loss.dtype.is_floating_point

        # indices should be integer
        assert not indices.dtype.is_floating_point
        assert indices.shape == (2, 8, 8)

    def test_resfsq_return_order(self) -> None:
        """ResidualFSQuantizer should return (codes, loss, indices)."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 8, 8, 8], num_quantizers=2, embedding_dim=4
        )
        x = torch.randn(2, 4, 8, 8)

        result = quantizer(x)
        codes, loss, indices = result

        # codes should be float, same shape as input
        assert codes.dtype.is_floating_point
        assert codes.shape == x.shape

        # loss should be float
        assert loss.dtype.is_floating_point

        # indices should be integer, with num_quantizers dimension
        assert not indices.dtype.is_floating_point
        assert indices.shape == (2, 2, 8, 8)  # (B, num_quantizers, H, W)

    def test_vq_return_order(self) -> None:
        """VectorQuantizer should return (codes, loss, indices)."""
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2)
        x = torch.randn(2, 4, 8, 8)

        result = quantizer(x)
        codes, loss, indices = result

        # codes should be float, same shape as input
        assert codes.dtype.is_floating_point
        assert codes.shape == x.shape

        # loss should be float
        assert loss.dtype.is_floating_point

        # indices should be integer
        assert not indices.dtype.is_floating_point

    def test_lfq_return_order(self) -> None:
        """LFQuantizer should return (codes, loss, indices)."""
        quantizer = LFQuantizer(
            codebook_size=16, codebook_dim=4, dim=2, entropy_loss=False
        )
        x = torch.randn(2, 4, 8, 8)

        result = quantizer(x)
        codes, loss, indices = result

        # codes should be float, same shape as input
        assert codes.dtype.is_floating_point
        assert codes.shape == x.shape

        # loss should be float
        assert loss.dtype.is_floating_point

        # indices should be integer
        assert not indices.dtype.is_floating_point
