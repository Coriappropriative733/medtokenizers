"""Smoke coverage for training entrypoint model types.

These tests ensure OpenMind and MedMNIST training entrypoints can build and run
all supported model variants without shape/runtime regressions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.train_medmnist2d import create_2d_tokenizer
from examples.train_medmnist3d import create_3d_tokenizer
from scripts.train import build_model as build_openmind_model


def _extract_reconstruction(output: object) -> torch.Tensor:
    """Get reconstruction tensor from either dict or namedtuple output."""
    if isinstance(output, dict):
        return output["reconstructions"]
    return output.reconstructions


def _make_openmind_args(model_type: str, twod: bool) -> SimpleNamespace:
    """Build minimal args namespace required by scripts.train.build_model()."""
    return SimpleNamespace(
        type=model_type,
        twod=twod,
        in_channels=1,
        out_channels=1,
        z_channels=8,
        latent_channels=4,
        channels=16,
        channels_mult=[1, 2],
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        resolution=16,
        crop_size=16,
        spatial_compression=4,
        patch_size=1,
        patch_method="haar",
        voronoi_jitter=0.0,
        embedding_dim=4,
        num_codebooks=2,
        num_embeddings=32,
        codebook_dim=4,
        codebook_size=16,
        levels=[4, 4, 4, 4],
        beta=0.25,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        quant_temp=0.01,
        use_norm=False,
        mcq_heads=1,
    )


@pytest.mark.parametrize(
    "model_type",
    [
        "AE",
        "VAE",
        "VQ",
        "LFQ",
        "FSQ",
        "RESFSQ",
    ],
)
@pytest.mark.parametrize("twod", [True, False])
def test_openmind_builder_supports_all_model_types(model_type: str, twod: bool) -> None:
    """OpenMind training model builder should run a forward pass for every type."""
    args = _make_openmind_args(model_type=model_type, twod=twod)
    model, _ = build_openmind_model(args)

    x = torch.randn(1, 1, 16, 16) if twod else torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        output = model(x)

    reconstruction = _extract_reconstruction(output)
    assert reconstruction.shape == x.shape


@pytest.mark.parametrize(
    "quantizer",
    [
        "VQ",
        "FSQ",
        "LFQ",
        "RESFSQ",
    ],
)
def test_medmnist_factory_supports_all_discrete_quantizers(quantizer: str) -> None:
    """MedMNIST 2D/3D factories should support every discrete quantizer."""
    model_2d = create_2d_tokenizer(
        dataset_name="pathmnist",
        in_channels=1,
        resolution=16,
        spatial_compression=4,
        tokenizer_type="discrete",
        quantizer=quantizer,
        levels=[4, 4, 4, 4],
        num_codebooks=2,
        codebook_dim=4,
        codebook_size=16,
    )
    model_3d = create_3d_tokenizer(
        dataset_name="organmnist3d",
        in_channels=1,
        resolution=16,
        spatial_compression=4,
        tokenizer_type="discrete",
        quantizer=quantizer,
        levels=[4, 4, 4, 4],
        num_codebooks=2,
        codebook_dim=4,
        codebook_size=16,
    )

    x2d = torch.randn(1, 1, 16, 16)
    x3d = torch.randn(1, 1, 16, 16, 16)

    with torch.no_grad():
        out2d = model_2d(x2d)
        out3d = model_3d(x3d)

    assert _extract_reconstruction(out2d).shape == x2d.shape
    assert _extract_reconstruction(out3d).shape == x3d.shape


@pytest.mark.parametrize("formulation", ["AE", "VAE"])
def test_medmnist_factory_supports_all_continuous_formulations(
    formulation: str,
) -> None:
    """MedMNIST 2D/3D factories should support AE and VAE continuous variants."""
    model_2d = create_2d_tokenizer(
        dataset_name="pathmnist",
        in_channels=1,
        resolution=16,
        spatial_compression=4,
        tokenizer_type="continuous",
        formulation=formulation,
        latent_channels=4,
    )
    model_3d = create_3d_tokenizer(
        dataset_name="organmnist3d",
        in_channels=1,
        resolution=16,
        spatial_compression=4,
        tokenizer_type="continuous",
        formulation=formulation,
        latent_channels=4,
    )

    x2d = torch.randn(1, 1, 16, 16)
    x3d = torch.randn(1, 1, 16, 16, 16)

    with torch.no_grad():
        out2d = model_2d(x2d)
        out3d = model_3d(x3d)

    assert _extract_reconstruction(out2d).shape == x2d.shape
    assert _extract_reconstruction(out3d).shape == x3d.shape
