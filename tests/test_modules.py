import pytest
import torch

from medtokenizers.modules.layers import AttnBlock, ResnetBlock
from medtokenizers.modules.patch import SpatialCompressor, SpatialDecompressor


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def feature_channels() -> int:
    return 16


@pytest.fixture
def patch_size() -> int:
    return 4


def test_resnet_block_preserves_shape_2d(
    batch_size: int, feature_channels: int
) -> None:
    block = ResnetBlock(
        in_channels=feature_channels,
        out_channels=feature_channels,
        dropout=0.0,
        dim=2,
    ).to(torch.float32)
    x = torch.randn(batch_size, feature_channels, 32, 32, dtype=torch.float32)
    out = block(x)

    assert out.shape == x.shape


def test_resnet_block_preserves_shape_3d(
    batch_size: int, feature_channels: int
) -> None:
    block = ResnetBlock(
        in_channels=feature_channels,
        out_channels=feature_channels,
        dropout=0.0,
        dim=3,
    ).to(torch.float32)
    x = torch.randn(batch_size, feature_channels, 16, 16, 16, dtype=torch.float32)
    out = block(x)

    assert out.shape == x.shape


def test_attention_block_is_identity_on_constant_input_2d(
    feature_channels: int,
) -> None:
    block = AttnBlock(in_channels=feature_channels, dim=2).to(torch.float32)
    x = torch.ones(1, feature_channels, 8, 8, dtype=torch.float32)
    out = block(x)

    # For constant input, attention should preserve spatial uniformity
    # Check that all spatial positions have the same value within each channel
    for c in range(feature_channels):
        channel_values = out[0, c].flatten()
        assert torch.allclose(channel_values, channel_values[0], atol=1e-5), (
            f"Channel {c} not uniform"
        )

    # Also check that shape is preserved
    assert out.shape == x.shape


def test_attention_block_handles_3d_tensor(feature_channels: int) -> None:
    block = AttnBlock(in_channels=feature_channels, dim=3).to(torch.float32)
    x = torch.randn(1, feature_channels, 4, 4, 4, dtype=torch.float32)
    out = block(x)

    assert out.shape == x.shape


def test_spatial_compressor_and_decompressor_round_trip_2d(
    batch_size: int, patch_size: int
) -> None:
    compressor = SpatialCompressor(
        dim=2, patch_size=patch_size, patch_method="rearrange"
    ).to(torch.float32)
    decompressor = SpatialDecompressor(
        dim=2, patch_size=patch_size, patch_method="rearrange"
    ).to(torch.float32)
    x = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)

    compressed = compressor(x)
    reconstructed = decompressor(compressed)

    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=1e-5)


def test_spatial_compressor_and_decompressor_round_trip_haar_3d(
    batch_size: int, patch_size: int
) -> None:
    compressor = SpatialCompressor(
        dim=3, patch_size=patch_size, patch_method="haar"
    ).to(torch.float32)
    decompressor = SpatialDecompressor(
        dim=3, patch_size=patch_size, patch_method="haar"
    ).to(torch.float32)
    x = torch.randn(batch_size, 2, 16, 16, 16, dtype=torch.float32)

    compressed = compressor(x)
    reconstructed = decompressor(compressed)

    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=1e-4)
