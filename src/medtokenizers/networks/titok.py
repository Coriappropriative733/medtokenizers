"""TiTok tokenizer: 1D latent tokens with ViT encoder/decoder."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn

from medtokenizers.modules.base import BaseTokenizer
from medtokenizers.modules.quant import VectorQuantizer
from medtokenizers.modules.utils import validate_tensor_input
from medtokenizers.networks._types import NetworkEval

if TYPE_CHECKING:
    from jaxtyping import Float, Int


def _to_tuple(value: int | Iterable[int], dim: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")
        return (value,) * dim
    items = tuple(int(v) for v in value)
    if len(items) != dim:
        raise ValueError(f"{name} must have length {dim}, got {len(items)}.")
    if any(v <= 0 for v in items):
        raise ValueError(f"{name} must contain positive ints, got {items}.")
    return items


class TiTokTokenizer(BaseTokenizer):
    """Transformer-based 1D image tokenizer (TiTok).

    Implements the TiTok idea (Yu et al., 2024) for 2D and 3D medical images:
    instead of producing a spatial grid of codes, the image is compressed into a
    flat sequence of ``num_tokens`` learnable 1D latent tokens. The input is
    split into non-overlapping patches and embedded; a learnable set of latent
    tokens is appended and a Transformer encoder lets those latent tokens attend
    over all patches. The latent tokens are then vector-quantized into discrete
    indices. Decoding mirrors this: quantized latents plus a learnable mask token
    per patch are passed through a Transformer decoder, and the patch outputs are
    un-embedded and reassembled into an image.

    Because the latent is a fixed-length 1D sequence (independent of spatial
    resolution), TiTok is well suited to autoregressive / sequence models. Note
    that the input spatial size is fixed: it must equal ``resolution`` (validated
    at encode time), and ``resolution`` must be divisible by ``patch_size``.

    Patch ordering:
        - 2D: patches are flattened in (height, width) row-major order, i.e.
          token index ``i = row * grid_w + col``. Each patch vector concatenates
          its pixels as ``(p_h, p_w, channels)``.
        - 3D: patches are flattened in (depth, height, width) order following the
          ``(g_h, g_w, g_d)`` grid, with each patch vector laid out as
          ``(p_h, p_w, p_d, channels)``.

    ``_patchify`` / ``_unpatchify`` are exact inverses, so decode restores
    the original spatial layout.

    Args:
        dim: Spatial dimensionality, ``2`` or ``3``.
        in_channels: Number of input image channels.
        out_channels: Number of reconstructed output channels. Defaults to
            ``in_channels`` when ``None``.
        num_tokens: Number of 1D latent tokens (the compressed sequence length).
        num_embeddings: Codebook size of the vector quantizer (vocabulary).
        embedding_dim: Dimensionality of each quantized code. Defaults to
            ``hidden_dim`` when ``None``.
        hidden_dim: Transformer model width; must be divisible by ``num_heads``.
        num_heads: Number of attention heads in encoder and decoder.
        num_layers: Number of Transformer layers in encoder and decoder.
        patch_size: Patch edge length. Either a single int (applied to every
            spatial axis) or a per-axis iterable of length ``dim``.
        resolution: Expected input spatial size. Either a single int or a
            per-axis iterable of length ``dim``. Must be divisible by
            ``patch_size`` along every axis.
        dropout: Dropout probability in ``[0, 1)`` applied to embeddings and
            within Transformer layers.
        beta: Commitment loss weight for the vector quantizer.
        use_norm: Whether the quantizer L2-normalizes codes/inputs.
        use_ema: Whether the quantizer updates its codebook via EMA.
        ema_decay: EMA decay used when ``use_ema`` is ``True``.
        reset_unused_codes: Whether to reset dead codebook entries.
        dead_code_threshold: Usage count below which a code is considered dead.
        name: Human-readable tokenizer name.

    Shapes:
        - Input: ``(B, in_channels, *resolution)`` where ``*resolution`` is
          ``(H, W)`` for 2D or ``(H, W, D)`` for 3D.
        - Discrete indices: ``(B, num_tokens)`` (integer dtype).
        - Quantized latents: ``(B, num_tokens, embedding_dim)``.
        - Reconstruction: ``(B, out_channels, *resolution)``.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        out_channels: int | None = None,
        num_tokens: int = 32,
        num_embeddings: int = 1024,
        embedding_dim: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_size: int | Iterable[int] = 16,
        resolution: int | Iterable[int] = 128,
        dropout: float = 0.0,
        beta: float = 0.25,
        use_norm: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        reset_unused_codes: bool = False,
        dead_code_threshold: int = 100,
        name: str = "TiTokTokenizer",
    ) -> None:
        super().__init__(dim=dim, name=name)
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}.")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be > 0, got {num_embeddings}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if embedding_dim is None:
            embedding_dim = hidden_dim
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads "
                f"(got hidden_dim={hidden_dim}, num_heads={num_heads})."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_tokens = num_tokens
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.patch_size = _to_tuple(patch_size, dim, "patch_size")
        self.resolution = _to_tuple(resolution, dim, "resolution")
        self.beta = beta

        if any(
            size % patch != 0
            for size, patch in zip(
                self.resolution,
                self.patch_size,
            )
        ):
            raise ValueError(
                "resolution must be divisible by patch_size. "
                f"Got resolution={self.resolution}, patch_size={self.patch_size}."
            )

        self.patch_grid = tuple(
            size // patch
            for size, patch in zip(
                self.resolution,
                self.patch_size,
            )
        )
        self.num_patches = prod(self.patch_grid)
        self.patch_dim = self.in_channels * prod(self.patch_size)
        self.output_patch_dim = self.out_channels * prod(self.patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        self.latent_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_quant = nn.Linear(hidden_dim, embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            dim=1,
            beta=beta,
            use_norm=use_norm,
            use_ema=use_ema,
            ema_decay=ema_decay,
            reset_unused_codes=reset_unused_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.from_quant = nn.Linear(embedding_dim, hidden_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.patch_unembed = nn.Linear(hidden_dim, self.output_patch_dim)
        self.dropout = nn.Dropout(dropout)

        self.config = {
            "dim": dim,
            "in_channels": in_channels,
            "out_channels": self.out_channels,
            "num_tokens": num_tokens,
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "patch_size": list(self.patch_size),
            "resolution": list(self.resolution),
            "dropout": dropout,
            "beta": beta,
            "use_norm": use_norm,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "reset_unused_codes": reset_unused_codes,
            "dead_code_threshold": dead_code_threshold,
            "name": name,
        }

    def _validate_resolution(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> None:
        """Raise if the input spatial size differs from ``self.resolution``."""
        if tuple(x.shape[2:]) != self.resolution:
            raise ValueError(
                "Input spatial size does not match configured resolution. "
                f"Expected {self.resolution}, got {tuple(x.shape[2:])}."
            )

    def _patchify(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch num_patches patch_dim"]:
        """Split the image into a flat sequence of flattened patches."""
        if self.dim == 2:
            b, c, _, _ = x.shape
            p_h, p_w = self.patch_size
            g_h, g_w = self.patch_grid
            patches = x.reshape(b, c, g_h, p_h, g_w, p_w)
            patches = patches.permute(0, 2, 4, 3, 5, 1)
            return patches.reshape(b, g_h * g_w, c * p_h * p_w)

        b, c, _, _, _ = x.shape
        p_h, p_w, p_d = self.patch_size
        g_h, g_w, g_d = self.patch_grid
        patches = x.reshape(b, c, g_h, p_h, g_w, p_w, g_d, p_d)
        patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1)
        return patches.reshape(b, g_h * g_w * g_d, c * p_h * p_w * p_d)

    def _unpatchify(
        self, patches: Float[torch.Tensor, "batch num_patches output_patch_dim"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Reassemble decoded patch vectors back into an image (inverse of patchify)."""
        if patches.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {patches.shape[1]}."
            )
        if self.dim == 2:
            b, _, _ = patches.shape
            p_h, p_w = self.patch_size
            g_h, g_w = self.patch_grid
            patches = patches.reshape(b, g_h, g_w, p_h, p_w, self.out_channels)
            patches = patches.permute(0, 5, 1, 3, 2, 4)
            return patches.reshape(b, self.out_channels, g_h * p_h, g_w * p_w)

        b, _, _ = patches.shape
        p_h, p_w, p_d = self.patch_size
        g_h, g_w, g_d = self.patch_grid
        patches = patches.reshape(b, g_h, g_w, g_d, p_h, p_w, p_d, self.out_channels)
        patches = patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return patches.reshape(b, self.out_channels, g_h * p_h, g_w * p_w, g_d * p_d)

    def encode(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Int[torch.Tensor, "batch num_tokens"],
        Float[torch.Tensor, "batch num_tokens embedding_dim"],
        torch.Tensor,
    ]:
        """Encode an image into discrete 1D latent tokens.

        Args:
            x: Input image of shape ``(B, in_channels, *resolution)``.

        Returns:
            Tuple of ``(indices, quantized, quant_loss)`` where ``indices`` has
            shape ``(B, num_tokens)`` (integer dtype), ``quantized`` has shape
            ``(B, num_tokens, embedding_dim)``, and ``quant_loss`` is the
            quantizer commitment/codebook loss.
        """
        validate_tensor_input(x, self.dim, self.in_channels, "encode")
        self._validate_resolution(x)

        patches = self._patchify(x)
        patch_tokens = self.patch_embed(patches)
        patch_tokens = patch_tokens + self.patch_pos_embed
        patch_tokens = self.dropout(patch_tokens)

        latent_tokens = self.latent_tokens.expand(x.shape[0], -1, -1)
        latent_tokens = latent_tokens + self.latent_pos_embed
        latent_tokens = self.dropout(latent_tokens)

        encoder_input = torch.cat([patch_tokens, latent_tokens], dim=1)
        encoded = self.encoder(encoder_input)
        latent_encoded = encoded[:, -self.num_tokens :, :]

        quant_input = self.to_quant(latent_encoded)
        quantized, quant_loss, indices = self.quantizer(quant_input.transpose(1, 2))
        quantized = quantized.transpose(1, 2)
        return indices, quantized, quant_loss

    def indices_to_codes(
        self, indices: Int[torch.Tensor, "batch num_tokens"]
    ) -> Float[torch.Tensor, "batch num_tokens embedding_dim"]:
        """Look up quantized code vectors for a batch of token indices.

        Args:
            indices: Integer indices of shape ``(B, num_tokens)``.

        Returns:
            Quantized code vectors of shape ``(B, num_tokens, embedding_dim)``.
        """
        if not isinstance(indices, torch.Tensor):
            raise TypeError(
                f"indices_to_codes expects torch.Tensor, got {type(indices).__name__}."
            )
        if indices.is_floating_point() or indices.dtype == torch.bool:
            raise TypeError(
                f"indices_to_codes expects integer indices tensor, got {indices.dtype}."
            )
        if indices.dim() != 2:
            raise ValueError(
                f"indices_to_codes expects (B, num_tokens), got {indices.shape}."
            )
        if indices.shape[1] != self.num_tokens:
            raise ValueError(
                f"indices_to_codes expects num_tokens={self.num_tokens}, "
                f"got {indices.shape[1]}."
            )
        codes = self.quantizer.indices_to_codes(indices)
        if codes.shape != (indices.shape[0], self.num_tokens, self.embedding_dim):
            raise ValueError(
                "indices_to_codes returned unexpected shape "
                f"{codes.shape} for num_tokens={self.num_tokens}."
            )
        return codes

    def decode(
        self, quantized: Float[torch.Tensor, "batch num_tokens embedding_dim"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode quantized 1D latent tokens back into an image.

        Args:
            quantized: Quantized code vectors of shape
                ``(B, num_tokens, embedding_dim)``.

        Returns:
            Reconstructed image of shape ``(B, out_channels, *resolution)``.
        """
        if not isinstance(quantized, torch.Tensor):
            raise TypeError(
                f"decode expects torch.Tensor, got {type(quantized).__name__}."
            )
        if not quantized.is_floating_point():
            raise TypeError(
                f"decode expects floating point tensor, got {quantized.dtype}."
            )
        if quantized.dim() != 3:
            raise ValueError(
                f"decode expects (B, num_tokens, embedding_dim), got {quantized.shape}."
            )
        if quantized.shape[1] != self.num_tokens:
            raise ValueError(
                f"decode expects num_tokens={self.num_tokens}, "
                f"got {quantized.shape[1]}."
            )
        if quantized.shape[2] != self.embedding_dim:
            raise ValueError(
                f"decode expects embedding_dim={self.embedding_dim}, "
                f"got {quantized.shape[2]}."
            )

        latent = self.from_quant(quantized)
        latent = latent + self.latent_pos_embed

        mask_tokens = self.mask_token.expand(quantized.shape[0], self.num_patches, -1)
        mask_tokens = mask_tokens + self.patch_pos_embed

        decoder_input = torch.cat([latent, mask_tokens], dim=1)
        decoded = self.decoder(decoder_input)
        patch_tokens = decoded[:, self.num_tokens :, :]

        patches = self.patch_unembed(patch_tokens)
        return self._unpatchify(patches)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> dict[str, torch.Tensor] | NetworkEval:
        """Encode then decode ``x``, returning training or eval outputs.

        Args:
            x: Input image of shape ``(B, in_channels, *resolution)``.

        Returns:
            In training mode, a dict with ``reconstructions``, ``quant_loss``,
            ``quant_info`` (indices) and ``latents`` (quantized). In eval mode, a
            :class:`NetworkEval` named tuple.
        """
        indices, quantized, quant_loss = self.encode(x)
        reconstructions = self.decode(quantized)

        if self.training:
            return {
                "reconstructions": reconstructions,
                "quant_loss": quant_loss,
                "quant_info": indices,
                "latents": quantized,
            }

        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=indices,
        )

    def tokenize(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Int[torch.Tensor, "batch num_tokens"]:
        """Encode ``x`` to discrete token indices of shape ``(B, num_tokens)``."""
        return self.encode(x)[0]

    def detokenize(
        self, indices: Int[torch.Tensor, "batch num_tokens"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode discrete token indices back into an image.

        Inverse of :meth:`tokenize`. This is the canonical
        index-to-reconstruction decoding path.

        Args:
            indices: Integer indices of shape ``(B, num_tokens)``.

        Returns:
            Reconstructed image of shape ``(B, out_channels, *resolution)``.
        """
        if indices.dim() != 2:
            raise ValueError(
                f"detokenize expects (B, num_tokens), got {indices.shape}."
            )
        if indices.shape[1] != self.num_tokens:
            raise ValueError(
                f"detokenize expects num_tokens={self.num_tokens}, "
                f"got {indices.shape[1]}."
            )
        return self.decode(self.indices_to_codes(indices))


__all__ = ["TiTokTokenizer"]
