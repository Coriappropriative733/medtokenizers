"""Tests for :class:`RAETokenizer`.

The RAE tokenizer wraps a *frozen* representation encoder (a ViT/SigLIP-style
HuggingFace model or a NeuroVFM encoder). Every supported ``encoder_type``
loads its backbone via ``from_pretrained`` / external download, so a real
``RAETokenizer`` cannot be instantiated fully offline in CI.

To still cover the fixed forward contract (a ``NetworkEval`` namedtuple in eval
mode, a ``dict`` in training mode) we substitute a tiny offline stub for the
``ViTEncoderAdapter`` and run the *real* ``RAETokenizer.__init__`` / ``encode``
/ ``decode`` / ``forward`` against it. Pure-logic components (the
``PatchDecoder`` and the constructor's dimensionality validation) are tested
directly without any encoder. Tests that need real backbone downloads are
gated behind ``@pytest.mark.skip`` / the network marker.
"""

import pytest
import torch
import torch.nn as nn

import medtokenizers.networks.rae as rae_mod
from medtokenizers.networks.rae import (
    EncoderOutput,
    NetworkEval,
    PatchDecoder,
    RAETokenizer,
)


class _StubViTEncoderAdapter(nn.Module):
    """Offline stand-in matching the public surface of ``ViTEncoderAdapter``.

    Exposes the attributes ``RAETokenizer.__init__`` reads (``hidden_size``,
    ``patch_size``, ``image_size``) and the ``encode`` / ``denormalize`` methods
    its ``encode`` / ``decode`` call, without touching ``transformers`` or the
    network. It carries a trivial learnable parameter so it shows up in
    ``state_dict`` like the real frozen encoder would.
    """

    def __init__(
        self,
        encoder_name_or_path: str,
        image_size: int | None = None,
        drop_cls_token: bool = True,
        encoder_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = 16
        self.patch_size = 16
        self.image_size = image_size or 32
        self.expected_channels = 3
        self._dummy = nn.Linear(1, 1)

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        grid = self.image_size // self.patch_size
        tokens = torch.randn(x.shape[0], grid * grid, self.hidden_size)
        return EncoderOutput(tokens=tokens, grid_shape=(grid, grid))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def stub_vit(monkeypatch: pytest.MonkeyPatch):
    """Patch ``ViTEncoderAdapter`` so the real RAE ctor runs offline."""
    monkeypatch.setattr(rae_mod, "ViTEncoderAdapter", _StubViTEncoderAdapter)
    return _StubViTEncoderAdapter


@pytest.fixture
def rae_2d(stub_vit) -> RAETokenizer:
    """A 2D RAE tokenizer built with the offline stub encoder."""
    return RAETokenizer(
        dim=2,
        encoder_type="vit",
        encoder_name_or_path="stub-encoder",
        encoder_image_size=32,
        out_channels=1,
        decoder_hidden_dim=32,
        decoder_num_layers=2,
    )


# --------------------------------------------------------------------------- #
# Forward-contract tests (real RAETokenizer, offline stub encoder)
# --------------------------------------------------------------------------- #
def test_rae_instantiation(rae_2d: RAETokenizer) -> None:
    """A 2D RAE builds and infers its latent grid from the stub encoder."""
    assert rae_2d.dim == 2
    assert rae_2d.latent_dim == 16
    assert rae_2d.patch_size == (16, 16)
    assert rae_2d.latent_grid_shape == (2, 2)


def test_rae_forward_eval_returns_namedtuple(rae_2d: RAETokenizer) -> None:
    """Eval mode returns a NetworkEval namedtuple with ``posteriors=None``.

    This is the fixed forward contract that mirrors the sibling continuous
    tokenizer: ``reconstructions`` first, ``posteriors`` always ``None`` for
    RAE, and the spatial ``latent`` grid last.
    """
    rae_2d.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        output = rae_2d(x)

    assert isinstance(output, NetworkEval)
    assert output.posteriors is None
    assert output.reconstructions.shape == x.shape
    assert output.latent.shape == (1, rae_2d.latent_dim, 2, 2)


def test_rae_forward_train_returns_dict(rae_2d: RAETokenizer) -> None:
    """Training mode returns a dict carrying reconstructions and latents."""
    rae_2d.train()
    x = torch.randn(1, 1, 32, 32)

    output = rae_2d(x)

    assert isinstance(output, dict)
    assert output["reconstructions"].shape == x.shape
    assert "latent" in output and "latents" in output
    assert output["latent"].shape == (1, rae_2d.latent_dim, 2, 2)


def test_rae_encode_decode_round_trip(rae_2d: RAETokenizer) -> None:
    """encode() yields a latent grid that decode() maps back to image shape."""
    rae_2d.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        latent, grid_shape = rae_2d.encode(x)
        recon = rae_2d.decode(latent, grid_shape=grid_shape)

    assert grid_shape == (2, 2)
    assert latent.shape == (1, rae_2d.latent_dim, 2, 2)
    assert recon.shape == x.shape


def test_rae_tokenize_returns_latent_grid(rae_2d: RAETokenizer) -> None:
    """tokenize() is encode()[0]: just the latent grid tensor."""
    rae_2d.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        latent = rae_2d.tokenize(x)

    assert latent.shape == (1, rae_2d.latent_dim, 2, 2)


def test_rae_decode_rejects_wrong_rank(rae_2d: RAETokenizer) -> None:
    """decode() requires 4D or 5D latents."""
    with pytest.raises(ValueError):
        rae_2d.decode(torch.randn(1, rae_2d.latent_dim, 4))


# --------------------------------------------------------------------------- #
# Constructor validation (no encoder construction required)
# --------------------------------------------------------------------------- #
def test_rae_vit_requires_2d() -> None:
    """ViT/SigLIP encoders reject dim=3 before any backbone is built."""
    with pytest.raises(ValueError):
        RAETokenizer(dim=3, encoder_type="vit", encoder_name_or_path="stub")


def test_rae_neurovfm_requires_3d() -> None:
    """NeuroVFM encoder rejects dim=2 before any backbone is built."""
    with pytest.raises(ValueError):
        RAETokenizer(dim=2, encoder_type="neurovfm", encoder_name_or_path="stub")


def test_rae_rejects_unknown_encoder_type() -> None:
    """An unsupported encoder_type raises ValueError."""
    with pytest.raises(ValueError):
        RAETokenizer(dim=2, encoder_type="bogus", encoder_name_or_path="stub")


# --------------------------------------------------------------------------- #
# PatchDecoder pure-logic tests (fully offline)
# --------------------------------------------------------------------------- #
def test_patch_decoder_2d_shape() -> None:
    """2D PatchDecoder un-patchifies (B, N, C) tokens to (B, out, H*p, W*p)."""
    decoder = PatchDecoder(
        latent_dim=8,
        out_channels=1,
        patch_size=(4, 4),
        dim=2,
        hidden_dim=16,
        num_layers=2,
    )
    tokens = torch.randn(2, 9, 8)  # 3x3 grid

    out = decoder(tokens, grid_shape=(3, 3))

    assert out.shape == (2, 1, 12, 12)


def test_patch_decoder_3d_shape() -> None:
    """3D PatchDecoder un-patchifies to (B, out, D*p, H*p, W*p)."""
    decoder = PatchDecoder(
        latent_dim=8,
        out_channels=1,
        patch_size=(2, 2, 2),
        dim=3,
        hidden_dim=16,
        num_layers=2,
    )
    tokens = torch.randn(1, 8, 8)  # 2x2x2 grid

    out = decoder(tokens, grid_shape=(2, 2, 2))

    assert out.shape == (1, 1, 4, 4, 4)


def test_patch_decoder_rejects_non_3d_tokens() -> None:
    """PatchDecoder requires tokens of shape (B, N, C)."""
    decoder = PatchDecoder(
        latent_dim=8,
        out_channels=1,
        patch_size=(4, 4),
        dim=2,
        num_layers=1,
    )
    with pytest.raises(ValueError):
        decoder(torch.randn(2, 8), grid_shape=(3, 3))


# --------------------------------------------------------------------------- #
# Real-backbone tests (require downloads / network) -- skipped in CI
# --------------------------------------------------------------------------- #
@pytest.mark.skip(
    reason="Requires downloading a real ViT/SigLIP backbone via "
    "transformers.from_pretrained; not available offline in CI."
)
def test_rae_with_real_vit_encoder() -> None:  # pragma: no cover
    """Documents building RAE on a real HF ViT encoder (skipped in CI)."""
    model = RAETokenizer(
        dim=2,
        encoder_type="vit",
        encoder_name_or_path="google/vit-base-patch16-224",
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 224, 224))
    assert isinstance(out, NetworkEval)


@pytest.mark.skip(
    reason="Requires the optional 'neurovfm' package and a downloaded 3D "
    "encoder checkpoint; not available offline in CI."
)
def test_rae_with_real_neurovfm_encoder() -> None:  # pragma: no cover
    """Documents building RAE on a real NeuroVFM encoder (skipped in CI)."""
    model = RAETokenizer(
        dim=3,
        encoder_type="neurovfm",
        encoder_name_or_path="neurovfm/encoder",
    )
    assert isinstance(model, RAETokenizer)
