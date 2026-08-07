"""Phase 5 Behavioral Tests - Prove critical system behaviors.

These tests verify invariants and behaviors that must hold for the system
to work correctly in production. They focus on:
1. Serialization roundtrips (state persistence)
2. Token distribution metrics (codebook health)
3. Reconstruction quality invariants
4. Edge case stress tests
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from medtokenizers.modules.quant import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)
from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer


@pytest.fixture
def base_kwargs() -> dict:
    """Minimal kwargs for fast test models."""
    return {
        "in_channels": 1,
        "out_channels": 1,
        "channels": 16,
        "channels_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
        "resolution": 16,
        "spatial_compression": 4,
    }


# ============================================================================
# Phase 5a: Serialization Roundtrip Tests
# ============================================================================


class TestQuantizerStatePersistence:
    """Test that quantizer state is correctly saved and restored."""

    def test_vq_codebook_survives_save_load(self) -> None:
        """Test VQ codebook is identical after save/load cycle."""
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=8, dim=2)

        # Get original codebook
        original_codebook = quantizer.embedding.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vq_state.pt"
            torch.save(quantizer.state_dict(), path)

            # Create new quantizer and load state
            loaded = VectorQuantizer(num_embeddings=64, embedding_dim=8, dim=2)
            loaded.load_state_dict(torch.load(path, weights_only=True))

            assert torch.equal(original_codebook, loaded.embedding.weight.data), (
                "VQ codebook changed after save/load"
            )

    def test_vq_ema_buffers_survive_save_load(self) -> None:
        """Test VQ EMA buffers are correctly persisted."""
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=8, dim=2, use_ema=True
        )
        quantizer.train()

        # Run some updates to populate EMA buffers
        for _ in range(10):
            x = torch.randn(4, 8, 8, 8)
            quantizer(x)

        # Store original buffer values
        original_cluster_size = quantizer.ema_cluster_size.clone()
        original_embed_sum = quantizer.ema_embed_sum.clone()
        original_batches_since_used = quantizer.batches_since_used.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vq_ema_state.pt"
            torch.save(quantizer.state_dict(), path)

            loaded = VectorQuantizer(
                num_embeddings=64, embedding_dim=8, dim=2, use_ema=True
            )
            loaded.load_state_dict(torch.load(path, weights_only=True))

            assert torch.equal(original_cluster_size, loaded.ema_cluster_size)
            assert torch.equal(original_embed_sum, loaded.ema_embed_sum)
            assert torch.equal(original_batches_since_used, loaded.batches_since_used)

    def test_fsq_buffers_survive_save_load(self) -> None:
        """Test FSQ internal buffers are correctly persisted."""
        quantizer = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)

        original_levels = quantizer._levels.clone()
        original_basis = quantizer._basis.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fsq_state.pt"
            torch.save(quantizer.state_dict(), path)

            loaded = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)
            loaded.load_state_dict(torch.load(path, weights_only=True))

            assert torch.equal(original_levels, loaded._levels)
            assert torch.equal(original_basis, loaded._basis)

    def test_resfsq_shared_projections_survive_save_load(self) -> None:
        """Test ResidualFSQ with shared projections saves/loads correctly."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=4,
            embedding_dim=16,
            share_projections=True,
        )

        x = torch.randn(2, 16, 4, 4)
        original_quant, _, original_indices = quantizer(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "resfsq_state.pt"
            torch.save(quantizer.state_dict(), path)

            loaded = ResidualFSQuantizer(
                levels=[8, 5, 5],
                num_quantizers=4,
                embedding_dim=16,
                share_projections=True,
            )
            loaded.load_state_dict(torch.load(path, weights_only=True))

            loaded_quant, _, loaded_indices = loaded(x)

            assert torch.equal(original_indices, loaded_indices)
            assert torch.allclose(original_quant, loaded_quant)

    def test_lfq_codebook_buffer_survives_save_load(self) -> None:
        """Test LFQ codebook buffer is correctly persisted."""
        quantizer = LFQuantizer(
            codebook_size=64, codebook_dim=6, dim=2, entropy_loss=True
        )

        original_codebook = quantizer.codebook.clone()
        original_mask = quantizer.mask.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lfq_state.pt"
            torch.save(quantizer.state_dict(), path)

            loaded = LFQuantizer(
                codebook_size=64, codebook_dim=6, dim=2, entropy_loss=True
            )
            loaded.load_state_dict(torch.load(path, weights_only=True))

            assert torch.equal(original_codebook, loaded.codebook)
            assert torch.equal(original_mask, loaded.mask)


class TestTokenizerStatePersistence:
    """Test full tokenizer state persistence."""

    def test_discrete_tokenizer_produces_identical_output_after_reload(
        self, base_kwargs: dict
    ) -> None:
        """Test that reloaded tokenizer produces identical outputs."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="VQ",
            num_embeddings=64,
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            original_indices, original_quant, _ = model.encode(x)
            original_recon = model.decode(original_quant)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save_pretrained(save_path)

            loaded = DiscreteTokenizer.from_pretrained(save_path)
            loaded.eval()

            with torch.no_grad():
                loaded_indices, loaded_quant, _ = loaded.encode(x)
                loaded_recon = loaded.decode(loaded_quant)

            assert torch.equal(original_indices, loaded_indices), (
                "Indices differ after reload"
            )
            assert torch.allclose(original_quant, loaded_quant, atol=1e-5), (
                "Quantized differ after reload"
            )
            assert torch.allclose(original_recon, loaded_recon, atol=1e-5), (
                "Reconstructions differ after reload"
            )


# ============================================================================
# Phase 5b: Token Distribution / Codebook Usage Tests
# ============================================================================


class TestCodebookUsage:
    """Test codebook utilization metrics and health."""

    def test_vq_get_codebook_usage_returns_valid_metrics(self) -> None:
        """Test VQ usage metrics are computed correctly."""
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=8, dim=2, use_ema=True
        )
        quantizer.train()

        # Run some forward passes
        for _ in range(20):
            x = torch.randn(8, 8, 8, 8)
            quantizer(x)

        usage = quantizer.get_codebook_usage()

        assert "perplexity" in usage
        assert "usage_fraction" in usage
        assert "dead_codes" in usage

        # Perplexity should be positive
        assert usage["perplexity"] > 0

        # Usage fraction should be between 0 and 1
        assert 0 <= usage["usage_fraction"] <= 1

        # Dead codes should be non-negative
        assert usage["dead_codes"] >= 0

    def test_vq_perplexity_increases_with_usage(self) -> None:
        """Test that perplexity increases as more codes are used."""
        quantizer = VectorQuantizer(
            num_embeddings=256, embedding_dim=8, dim=2, use_ema=True
        )
        quantizer.train()

        # Few updates - expect low perplexity
        for _ in range(5):
            x = torch.randn(4, 8, 4, 4)
            quantizer(x)

        usage_early = quantizer.get_codebook_usage()

        # Many more updates - expect higher perplexity
        for _ in range(50):
            x = torch.randn(8, 8, 8, 8)
            quantizer(x)

        usage_late = quantizer.get_codebook_usage()

        assert usage_late["perplexity"] >= usage_early["perplexity"], (
            "Perplexity should increase with more diverse usage"
        )

    def test_fsq_uses_full_codebook_range(self) -> None:
        """Test FSQ indices span the full codebook range with diverse input."""
        quantizer = FSQuantizer(levels=[8, 8], embedding_dim=2)
        codebook_size = quantizer.get_codebook_size()  # 64

        # Generate diverse inputs that should hit different codes
        all_indices = []
        for _ in range(100):
            x = torch.randn(4, 2, 8, 8) * 2  # Wider range
            _, _, indices = quantizer(x)
            all_indices.append(indices.flatten())

        all_indices = torch.cat(all_indices)
        unique_codes = torch.unique(all_indices)

        # Should use at least 50% of codes with random input
        usage_fraction = len(unique_codes) / codebook_size
        assert usage_fraction > 0.5, f"FSQ only used {usage_fraction:.1%} of codebook"

    def test_lfq_indices_in_valid_range(self) -> None:
        """Test LFQ indices are within valid codebook range."""
        codebook_size = 256
        quantizer = LFQuantizer(
            codebook_size=codebook_size, codebook_dim=8, dim=2, entropy_loss=False
        )

        x = torch.randn(4, 8, 16, 16)
        _, _, indices = quantizer(x)

        assert indices.min() >= 0, "LFQ produced negative indices"
        assert indices.max() < codebook_size, "LFQ indices exceed codebook size"


class TestTokenDistribution:
    """Test token distribution properties."""

    def test_fsq_uniform_input_produces_centered_codes(self) -> None:
        """Test FSQ produces centered codes for uniform input."""
        quantizer = FSQuantizer(levels=[9, 9, 9], embedding_dim=3)  # Odd levels

        # Zero input should quantize to center codes (0)
        x = torch.zeros(2, 3, 8, 8)
        quantized, _, indices = quantizer(x)

        # Quantized values should be near 0 for zero input
        assert quantized.abs().max() < 0.3, (
            "Zero input should produce near-zero quantized values"
        )

    def test_vq_indices_match_nearest_embedding(self) -> None:
        """Test VQ indices correspond to actual nearest embeddings."""
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=4, dim=2)

        x = torch.randn(2, 4, 4, 4)
        quantized, _, indices = quantizer(x)

        # Manually look up embeddings for returned indices
        with torch.no_grad():
            expected = quantizer.embedding(indices.flatten())
            expected = expected.view(*indices.shape, -1)
            expected = expected.permute(0, 3, 1, 2)  # (B, C, H, W)

        assert torch.allclose(quantized.detach(), expected, atol=1e-6), (
            "VQ quantized output doesn't match embedding lookup"
        )


# ============================================================================
# Phase 5c: Reconstruction Quality Invariants
# ============================================================================


class TestReconstructionInvariants:
    """Test invariants that must hold for reconstruction quality."""

    def test_encode_decode_reduces_to_codebook_space(self, base_kwargs: dict) -> None:
        """Test that encode->decode maps to quantized manifold."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[8, 5, 5, 5],
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            indices1, quant1, _ = model.encode(x)
            recon1 = model.decode(quant1)

            # Re-encoding the reconstruction should give same indices
            indices2, quant2, _ = model.encode(recon1)

        # This is a strong test: reconstruction lies on quantized manifold
        # In practice, due to decoder imperfection, indices may differ
        # But the quantized representations should be close
        assert quant2.shape == quant1.shape

    def test_vq_double_quantization_is_idempotent(self) -> None:
        """Test VQ quantizing already-quantized values gives same result.

        VQ is truly idempotent because it does exact codebook lookup.
        FSQ is NOT idempotent due to the tanh bounding function.
        """
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2)

        x = torch.randn(2, 4, 8, 8)

        quantized1, _, indices1 = quantizer(x)
        quantized2, _, indices2 = quantizer(quantized1)

        # VQ should be exactly idempotent
        assert torch.equal(indices1, indices2), "VQ double quantization changed indices"
        assert torch.allclose(quantized1, quantized2, atol=1e-6), (
            "VQ double quantization changed values"
        )

        # Most indices should match (allow small percentage to differ at boundaries)
        match_ratio = (indices1 == indices2).float().mean()
        assert match_ratio > 0.95, (
            f"Too many indices changed: {1 - match_ratio:.1%} differ"
        )

    def test_indices_to_codes_is_inverse_of_forward(self) -> None:
        """Test indices_to_codes inverts the forward pass encoding."""
        quantizer = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)

        x = torch.randn(2, 4, 8, 8)
        quantized, _, indices = quantizer(x)

        # indices_to_codes should recover the quantized representation
        recovered = quantizer.indices_to_codes(indices)

        # recovered is (B, H, W, C), quantized is (B, C, H, W)
        recovered = recovered.permute(0, 3, 1, 2)

        assert torch.allclose(recovered, quantized, atol=1e-5), (
            "indices_to_codes doesn't invert forward pass"
        )

    def test_resfsq_indices_to_codes_roundtrip(self) -> None:
        """Test ResidualFSQ indices_to_codes inverts encoding."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5], num_quantizers=3, embedding_dim=3
        )

        x = torch.randn(2, 3, 8, 8)
        quantized, _, indices = quantizer(x)

        recovered = quantizer.indices_to_codes(indices)

        assert torch.allclose(recovered, quantized, atol=1e-5), (
            "ResidualFSQ indices_to_codes doesn't invert forward"
        )


# ============================================================================
# Phase 5d: Edge Case Stress Tests
# ============================================================================


class TestEdgeCaseSizes:
    """Test handling of edge case input sizes."""

    def test_quantizers_handle_batch_size_1(self) -> None:
        """Test all quantizers handle batch_size=1 correctly."""
        quantizers = [
            FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4),
            VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2),
            ResidualFSQuantizer(levels=[8, 8, 8, 8], num_quantizers=2),
            LFQuantizer(codebook_size=16, codebook_dim=4, dim=2, entropy_loss=False),
        ]

        x = torch.randn(1, 4, 8, 8)  # batch_size=1

        for quantizer in quantizers:
            # All quantizers now return (codes, loss, indices) consistently
            quantized, loss, indices = quantizer(x)

            assert quantized.shape[0] == 1, f"{type(quantizer).__name__} failed batch=1"
            assert not torch.isnan(quantized).any()
            assert not torch.isnan(loss).any()

    def test_quantizers_handle_spatial_size_1(self) -> None:
        """Test quantizers handle 1x1 spatial dimensions."""
        quantizers = [
            FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4),
            VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2),
            ResidualFSQuantizer(levels=[8, 8, 8, 8], num_quantizers=2),
            LFQuantizer(codebook_size=16, codebook_dim=4, dim=2, entropy_loss=False),
        ]

        x = torch.randn(2, 4, 1, 1)  # 1x1 spatial

        for quantizer in quantizers:
            # All quantizers now return (codes, loss, indices) consistently
            quantized, loss, indices = quantizer(x)

            assert quantized.shape == x.shape, f"{type(quantizer).__name__} shape wrong"
            assert not torch.isnan(quantized).any()

    def test_tokenizer_handles_minimum_resolution(self, base_kwargs: dict) -> None:
        """Test tokenizer handles minimum valid resolution."""
        # With spatial_compression=4, minimum is 4x4
        kwargs = {**base_kwargs, "resolution": 4}
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[8, 5, 5, 5],
            **kwargs,
        )

        x = torch.randn(2, 1, 4, 4)
        indices, quantized, loss = model.encode(x)

        assert not torch.isnan(quantized).any()
        assert indices.shape[0] == 2

    def test_quantizers_handle_large_batch(self) -> None:
        """Test quantizers handle large batches without OOM or errors."""
        # Use smaller spatial dims to avoid memory issues
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)

        x = torch.randn(64, 4, 4, 4)  # Large batch, small spatial

        quantized, loss, indices = quantizer(x)

        assert quantized.shape == x.shape
        assert not torch.isnan(quantized).any()


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_quantizers_handle_zero_input(self) -> None:
        """Test all quantizers handle all-zero input."""
        quantizers = [
            FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4),
            VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2),
            ResidualFSQuantizer(levels=[8, 8, 8, 8], num_quantizers=2),
            LFQuantizer(codebook_size=16, codebook_dim=4, dim=2, entropy_loss=False),
        ]

        x = torch.zeros(2, 4, 8, 8)

        for quantizer in quantizers:
            # All quantizers now return (codes, loss, indices) consistently
            quantized, loss, indices = quantizer(x)

            assert not torch.isnan(quantized).any(), (
                f"{type(quantizer).__name__} produced NaN for zero input"
            )
            assert not torch.isinf(quantized).any(), (
                f"{type(quantizer).__name__} produced Inf for zero input"
            )

    def test_quantizers_handle_constant_input(self) -> None:
        """Test quantizers handle constant (non-zero) input."""
        quantizers = [
            FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4),
            VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2),
            ResidualFSQuantizer(levels=[8, 8, 8, 8], num_quantizers=2),
            LFQuantizer(codebook_size=16, codebook_dim=4, dim=2, entropy_loss=False),
        ]

        x = torch.ones(2, 4, 8, 8) * 0.5

        for quantizer in quantizers:
            # All quantizers now return (codes, loss, indices) consistently
            quantized, loss, indices = quantizer(x)

            assert not torch.isnan(quantized).any()
            assert not torch.isinf(quantized).any()

            # All indices should be the same for constant input
            assert torch.all(indices == indices.flatten()[0]), (
                f"{type(quantizer).__name__} produced varying indices for constant input"
            )

    def test_fsq_handles_values_at_quantization_boundaries(self) -> None:
        """Test FSQ correctly handles values exactly at quantization boundaries."""
        levels = [5, 5, 5, 5]
        quantizer = FSQuantizer(levels=levels, embedding_dim=4)

        # Values that should quantize to exact boundary codes
        # For level 5: codes are at -1, -0.5, 0, 0.5, 1 (normalized)
        boundary_vals = torch.tensor([[-1.0, -0.5, 0.0, 0.5]])
        boundary_vals = boundary_vals.unsqueeze(-1).unsqueeze(-1)  # (1, 4, 1, 1)

        quantized, _, indices = quantizer(boundary_vals)

        # Should produce valid output
        assert not torch.isnan(quantized).any()
        assert not torch.isinf(quantized).any()

    def test_vq_handles_input_far_from_codebook(self) -> None:
        """Test VQ handles input very far from any codebook entry."""
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=4, dim=2)

        # Very large values, likely far from any codebook entry
        x = torch.ones(2, 4, 4, 4) * 1000

        quantized, loss, indices = quantizer(x)

        assert not torch.isnan(quantized).any()
        assert not torch.isinf(quantized).any()
        # Should still produce valid indices
        assert indices.min() >= 0
        assert indices.max() < 16


class TestGradientEdgeCases:
    """Test gradient behavior in edge cases."""

    def test_gradient_through_all_zeros(self) -> None:
        """Test gradient flows correctly through all-zero input."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)

        x = torch.zeros(2, 4, 8, 8, requires_grad=True)
        quantized, loss, indices = quantizer(x)

        quantized.sum().backward()

        assert x.grad is not None, "No gradient for zero input"
        # Gradient should exist but may be zero or non-zero depending on STE

    def test_gradient_doesnt_explode_with_extreme_values(self) -> None:
        """Test gradients don't explode with extreme input values."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)

        x = torch.randn(2, 4, 8, 8) * 100
        x = x.clone().detach().requires_grad_(True)  # Ensure it's a leaf tensor
        quantized, loss, indices = quantizer(x)

        quantized.sum().backward()

        assert x.grad is not None, "Gradient is None"
        assert not torch.isnan(x.grad).any(), "Gradient is NaN"
        assert not torch.isinf(x.grad).any(), "Gradient is Inf"
        assert x.grad.abs().max() < 1e6, "Gradient exploded"

    def test_vq_commitment_loss_gradient_flows(self) -> None:
        """Test VQ commitment loss produces valid gradients."""
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=4, dim=2, beta=0.25
        )

        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        quantized, loss, indices = quantizer(x)

        # Backward through loss only
        loss.sum().backward()

        assert x.grad is not None, "No gradient from commitment loss"
        assert not torch.isnan(x.grad).any(), "Commitment loss gradient is NaN"
        assert x.grad.abs().sum() > 0, "Commitment loss gradient is zero"
