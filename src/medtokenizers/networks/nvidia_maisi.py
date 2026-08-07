# Copyright 2026 Liam Chalcroft
# SPDX-License-Identifier: Apache-2.0
#
# This file interoperates with and reproduces the architecture of NVIDIA MAISI / MONAI
# AutoencoderKlMaisi (https://github.com/Project-MONAI/MONAI), licensed under the Apache License 2.0.
# The module mirrors the MAISI VAE layer layout (no encoder mid blocks, decoder_blocks_per_stage
# [2, 2, 0], separate quant_conv_mu / quant_conv_log_sigma) and the convert_nvidia_weights mapping
# converts published NVIDIA MAISI checkpoints into this tokenizer's state dict.
# See THIRD_PARTY_NOTICES.md for details.
"""NVIDIA MAISI VAE with exact architecture match for direct weight loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from medtokenizers.modules.base import BaseTokenizer
from medtokenizers.modules.layers import Decoder, Encoder
from medtokenizers.modules.utils import validate_tensor_input
from medtokenizers.networks._types import NetworkEval

if TYPE_CHECKING:
    from jaxtyping import Float


logger = logging.getLogger(__name__)


class NVIDIAMAISITokenizer(BaseTokenizer):
    """NVIDIA MAISI VAE with exact architecture for direct weight loading.

    Key differences from standard VAE (ContinuousTokenizer):
    - Encoder outputs z_channels (4), not 2*z_channels (8)
    - Separate quant_conv_mu and quant_conv_log_sigma (each 4->4)
    - No encoder mid blocks
    """

    def __init__(
        self,
        dim: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        z_channels: int = 4,
        channels: int = 64,
        channels_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        spatial_compression: int = 4,
        name: str = "NVIDIAMAISITokenizer",
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, name=name)

        self.config = {
            "dim": dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "z_channels": z_channels,
            "channels": channels,
            "channels_mult": list(channels_mult),
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": list(attn_resolutions),
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "name": name,
        }
        self.config.update(kwargs)

        self.z_channels = z_channels
        self.spatial_compression = spatial_compression

        layer_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channels": channels,
            "channels_mult": channels_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "use_encoder_mid": False,
            "decoder_blocks_per_stage": [2, 2, 0],
            "use_output_nonlinearity": False,
        }
        layer_kwargs.update(kwargs)

        self.encoder = Encoder(dim=dim, z_channels=z_channels, **layer_kwargs)
        self.decoder = Decoder(dim=dim, z_channels=z_channels, **layer_kwargs)

        conv_class = nn.Conv3d if dim == 3 else nn.Conv2d
        self.quant_conv_mu = conv_class(z_channels, z_channels, kernel_size=1)
        self.quant_conv_log_sigma = conv_class(z_channels, z_channels, kernel_size=1)
        self.post_quant_conv = conv_class(z_channels, z_channels, kernel_size=1)

    def encode(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Float[torch.Tensor, "batch z_channels *spatial_compressed"],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        validate_tensor_input(x, self.dim, self.config["in_channels"], "encode")

        h = self.encoder(x)
        mu = self.quant_conv_mu(h)
        log_sigma = self.quant_conv_log_sigma(h)

        if self.training:
            std = torch.exp(log_sigma)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        kl_loss = 0.5 * torch.sum(
            mu.pow(2) + log_sigma.exp().pow(2) - 1 - 2 * log_sigma
        )

        return z, (kl_loss, mu, log_sigma)

    def decode(
        self, z: Float[torch.Tensor, "batch z_channels *spatial_compressed"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        validate_tensor_input(z, self.dim, self.z_channels, "decode")
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self, input: Float[torch.Tensor, "batch channels *spatial"]
    ) -> dict[str, torch.Tensor] | NetworkEval:
        z, (kl_loss, mu, log_sigma) = self.encode(input)
        reconstructions = self.decode(z)

        if self.training:
            return {
                "reconstructions": reconstructions,
                "posteriors": (mu, log_sigma),
                "kl_loss": kl_loss,
                "latent": z,
                "latents": z,
            }

        return NetworkEval(
            reconstructions=reconstructions,
            posteriors=(mu, log_sigma),
            latent=z,
        )

    def tokenize(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch z_channels *spatial_compressed"]:
        return self.encode(x)[0]

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        b = input_shape[0]
        compression = self.spatial_compression
        if self.dim == 2:
            h, w = input_shape[2], input_shape[3]
            return (b, self.z_channels, h // compression, w // compression)
        else:
            h, w, d = input_shape[2], input_shape[3], input_shape[4]
            return (
                b,
                self.z_channels,
                h // compression,
                w // compression,
                d // compression,
            )

    @classmethod
    def from_nvidia_weights(
        cls,
        weights_path: str | Path,
        map_location: str = "cpu",
        **kwargs,
    ) -> "NVIDIAMAISITokenizer":
        """Load directly from NVIDIA checkpoint with automatic conversion."""
        model = cls(**kwargs)

        # Published NVIDIA MAISI checkpoints are plain state dicts (tensors only),
        # optionally wrapped under a "unet_state_dict" / "state_dict" key. They carry
        # no pickled Python objects, so weights_only=True is both safe and sufficient.
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
        if "unet_state_dict" in state:
            state = state["unet_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

        converted = convert_nvidia_weights(state)
        missing, unexpected = model.load_state_dict(converted, strict=False)

        if missing:
            logger.info("Missing keys: %s...", missing[:5])
        if unexpected:
            logger.info("Unexpected keys: %s...", unexpected[:5])

        return model


def convert_nvidia_weights(nvidia_state: dict) -> dict:
    """Convert NVIDIA MAISI weights to NVIDIAMAISITokenizer format."""
    from collections import OrderedDict

    new_state = OrderedDict()

    for suffix in ["weight", "bias"]:
        old_key = f"encoder.blocks.0.conv.conv.{suffix}"
        new_key = f"encoder.conv_in.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for block_idx, (level, block_num) in [(1, (0, 0)), (2, (0, 1))]:
        _convert_resblock(
            nvidia_state,
            new_state,
            f"encoder.blocks.{block_idx}",
            f"encoder.down.{level}.block.{block_num}",
        )

    for suffix in ["weight", "bias"]:
        old_key = f"encoder.blocks.3.conv.conv.conv.{suffix}"
        new_key = f"encoder.down.0.downsample.conv.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for block_idx, (level, block_num) in [(4, (1, 0)), (5, (1, 1))]:
        _convert_resblock(
            nvidia_state,
            new_state,
            f"encoder.blocks.{block_idx}",
            f"encoder.down.{level}.block.{block_num}",
        )

    for suffix in ["weight", "bias"]:
        old_key = f"encoder.blocks.6.conv.conv.conv.{suffix}"
        new_key = f"encoder.down.1.downsample.conv.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for block_idx, (level, block_num) in [(7, (2, 0)), (8, (2, 1))]:
        _convert_resblock(
            nvidia_state,
            new_state,
            f"encoder.blocks.{block_idx}",
            f"encoder.down.{level}.block.{block_num}",
        )

    for suffix in ["weight", "bias"]:
        old_key = f"encoder.blocks.9.{suffix}"
        new_key = f"encoder.norm_out.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for suffix in ["weight", "bias"]:
        old_key = f"encoder.blocks.10.conv.conv.{suffix}"
        new_key = f"encoder.conv_out.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for suffix in ["weight", "bias"]:
        for src, dst in [
            ("quant_conv_mu.conv", "quant_conv_mu"),
            ("quant_conv_log_sigma.conv", "quant_conv_log_sigma"),
            ("post_quant_conv.conv", "post_quant_conv"),
        ]:
            old_key = f"{src}.{suffix}"
            new_key = f"{dst}.{suffix}"
            if old_key in nvidia_state:
                new_state[new_key] = nvidia_state[old_key]

    for suffix in ["weight", "bias"]:
        old_key = f"decoder.blocks.0.conv.conv.{suffix}"
        new_key = f"decoder.conv_in.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    _convert_resblock(
        nvidia_state, new_state, "decoder.blocks.1", "decoder.mid.block_1"
    )
    _convert_resblock(
        nvidia_state, new_state, "decoder.blocks.2", "decoder.mid.block_2"
    )

    for suffix in ["weight", "bias"]:
        old_key = f"decoder.blocks.3.conv.conv.conv.{suffix}"
        new_key = f"decoder.up.2.upsample.conv.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for block_idx, (level, block_num) in [(4, (1, 0)), (5, (1, 1))]:
        _convert_resblock(
            nvidia_state,
            new_state,
            f"decoder.blocks.{block_idx}",
            f"decoder.up.{level}.block.{block_num}",
        )

    for suffix in ["weight", "bias"]:
        old_key = f"decoder.blocks.6.conv.conv.conv.{suffix}"
        new_key = f"decoder.up.1.upsample.conv.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for block_idx, (level, block_num) in [(7, (0, 0)), (8, (0, 1))]:
        _convert_resblock(
            nvidia_state,
            new_state,
            f"decoder.blocks.{block_idx}",
            f"decoder.up.{level}.block.{block_num}",
        )

    for suffix in ["weight", "bias"]:
        old_key = f"decoder.blocks.9.{suffix}"
        new_key = f"decoder.norm_out.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    for suffix in ["weight", "bias"]:
        old_key = f"decoder.blocks.10.conv.conv.{suffix}"
        new_key = f"decoder.conv_out.{suffix}"
        if old_key in nvidia_state:
            new_state[new_key] = nvidia_state[old_key]

    return new_state


def _convert_resblock(src: dict, dst: dict, src_prefix: str, dst_prefix: str):
    mappings = [
        ("norm1.weight", "norm1.weight"),
        ("norm1.bias", "norm1.bias"),
        ("conv1.conv.conv.weight", "conv1.weight"),
        ("conv1.conv.conv.bias", "conv1.bias"),
        ("norm2.weight", "norm2.weight"),
        ("norm2.bias", "norm2.bias"),
        ("conv2.conv.conv.weight", "conv2.weight"),
        ("conv2.conv.conv.bias", "conv2.bias"),
        ("nin_shortcut.conv.conv.weight", "nin_shortcut.weight"),
        ("nin_shortcut.conv.conv.bias", "nin_shortcut.bias"),
    ]
    for src_suffix, dst_suffix in mappings:
        src_key = f"{src_prefix}.{src_suffix}"
        dst_key = f"{dst_prefix}.{dst_suffix}"
        if src_key in src:
            dst[dst_key] = src[src_key]


# NVIDIAMAISITokenizer and convert_nvidia_weights are internal checkpoint-conversion
# utilities: they reproduce the MAISI architecture and translate published NVIDIA MAISI
# weights into this tokenizer's state dict. The supported public path is to convert NVIDIA
# weights into a ContinuousTokenizer via scripts/convert_maisi_to_hf.py and load the result
# with MAISITokenizer.from_pretrained. They are intentionally not re-exported from
# networks/__init__ or marked public here; import them directly from this module when needed.
__all__: list[str] = []
