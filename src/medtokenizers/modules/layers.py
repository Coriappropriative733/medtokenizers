# Copyright 2026 Liam Chalcroft
# SPDX-License-Identifier: MIT
#
# This file contains code derived from CompVis latent-diffusion / Stable Diffusion
# (https://github.com/CompVis/latent-diffusion), originally licensed under the MIT License.
# The Encoder/Decoder/ResnetBlock/AttnBlock/Upsample/Downsample structure and the
# nonlinearity/Normalize helpers follow that implementation (here generalized to 2D/3D).
# See THIRD_PARTY_NOTICES.md for details.
"""Neural network layers for encoder-decoder architectures.

This module provides the building blocks for VAE/VQ-VAE encoder and decoder
networks, with full support for both 2D and 3D volumetric data. The design
prioritizes memory efficiency and throughput for large medical imaging volumes.

Architecture Philosophy
-----------------------
The encoder-decoder follows a U-Net-like progressive resolution structure:
- Encoder: progressively downsamples while increasing channel depth
- Decoder: progressively upsamples while decreasing channel depth
- Skip connections are NOT used (unlike U-Net) to enable latent manipulation

Memory Optimization
-------------------
For 3D medical images, memory is often the bottleneck. This module implements:
1. **Channels-last memory format**: Improves cache locality for convolutions
2. **Gradient checkpointing**: Trades compute for VRAM during training
3. **Efficient upsampling**: Uses F.interpolate instead of repeat_interleave

Performance Tips
----------------
- For 3D volumes, always use `torch.channels_last_3d` memory format
- Enable checkpointing for volumes > 128³ during training
- Use `torch.backends.cudnn.benchmark = True` for consistent input sizes
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .patch import SpatialCompressor, SpatialDecompressor
from .utils import Normalize, nonlinearity

if TYPE_CHECKING:
    from jaxtyping import Float

# Type alias for spatial dimensions
SpatialDim = Literal[2, 3]


def get_conv(
    dim: SpatialDim,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    **kwargs,
) -> nn.Module:
    """Factory for 2D/3D convolutions with optimal memory format.

    Creates Conv2d or Conv3d based on spatial dimensionality. For 3D,
    initializes weights in channels_last_3d format for improved cuDNN
    kernel selection and cache efficiency on NVIDIA GPUs.

    The channels-last format stores tensors as NHWDC instead of NCHWD,
    which aligns better with cuDNN's internal representations and can
    provide 20-40% throughput improvements for 3D convolutions.

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride (default: 1)
        padding: Convolution padding (default: 0)
        **kwargs: Additional arguments passed to Conv2d/Conv3d

    Returns:
        Conv2d or Conv3d module with optimal memory layout

    Example:
        >>> conv = get_conv(3, 64, 128, kernel_size=3, padding=1)
        >>> x = torch.randn(1, 64, 32, 32, 32)
        >>> x = x.to(memory_format=torch.channels_last_3d)
        >>> y = conv(x)  # Stays in channels_last_3d format
    """
    if dim == 2:
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, **kwargs
        )
        # Convert to channels_last for optimal cuDNN performance on CUDA
        return conv.to(memory_format=torch.channels_last)
    else:
        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, **kwargs
        )
        # Convert to channels_last_3d for optimal cuDNN performance
        return conv.to(memory_format=torch.channels_last_3d)


def get_padding(dim: SpatialDim) -> tuple[int, ...]:
    """Get asymmetric padding for strided convolutions.

    When using stride=2 with kernel_size=3, we need asymmetric padding
    to ensure output size is exactly input_size // 2. This pads only
    the "end" of each spatial dimension.

    Args:
        dim: Spatial dimensionality (2 or 3)

    Returns:
        Padding tuple for F.pad (reversed order: last dim first)
    """
    # F.pad format: (left, right, top, bottom[, front, back])
    if dim == 2:
        return (0, 1, 0, 1)  # H and W padding
    else:
        return (0, 1, 0, 1, 0, 1)  # D, H, W padding


class Upsample(nn.Module):
    """Learnable upsampling via nearest-neighbor interpolation + convolution.

    Upsamples spatial dimensions by 2x using nearest-neighbor interpolation,
    then applies a 3x3 convolution to learn smooth upsampling patterns.
    This is preferred over transposed convolution to avoid checkerboard
    artifacts.

    Why not TransposedConv?
    -----------------------
    Transposed convolutions with stride=2 create overlapping output regions
    that can cause checkerboard artifacts. Nearest-neighbor + conv avoids
    this by ensuring uniform spatial coverage.

    Implementation Note
    -------------------
    Uses F.interpolate instead of repeat_interleave for efficiency.
    F.interpolate dispatches to optimized CUDA kernels and avoids
    creating intermediate tensors.

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input/output channels
    """

    def __init__(self, dim: SpatialDim, in_channels: int) -> None:
        super().__init__()
        self.conv = get_conv(dim, in_channels, in_channels, 3, 1, 1)
        self.dim = dim
        self._scale_factor: tuple[int, ...] = (2, 2, 2) if dim == 3 else (2, 2)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial_upsampled"]:
        """Upsample input by 2x in all spatial dimensions.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, H, W, D)

        Returns:
            Upsampled tensor with spatial dims doubled
        """
        # F.interpolate is faster than repeat_interleave and uses less memory
        x = F.interpolate(x, scale_factor=self._scale_factor, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """Strided convolution for 2x spatial downsampling.

    Downsamples by factor of 2 using stride-2 convolution with asymmetric
    padding. This learns spatial aggregation patterns rather than using
    fixed pooling operations.

    Why Strided Conv Over Pooling?
    ------------------------------
    1. Learnable: Can adapt to data-specific downsampling patterns
    2. More expressive: Not limited to max/avg operations
    3. Consistent with encoder design: Uses same conv blocks throughout

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input/output channels
    """

    def __init__(self, dim: SpatialDim, in_channels: int) -> None:
        super().__init__()
        # stride=2, no padding (we pad asymmetrically in forward)
        self.conv = get_conv(dim, in_channels, in_channels, 3, 2, 0)
        self.dim = dim
        self._padding = get_padding(dim)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial_downsampled"]:
        """Downsample input by 2x in all spatial dimensions.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, H, W, D)

        Returns:
            Downsampled tensor with spatial dims halved
        """
        # Asymmetric padding for exact 2x downsampling
        x = F.pad(x, self._padding, mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Pre-activation residual block with GroupNorm.

    Implements the "full pre-activation" residual block design where
    normalization and activation precede each convolution. This provides
    better gradient flow and is standard in modern autoencoder architectures.

    Block Structure
    ---------------
    ```
    x ─────────────────────────────┐
    │                              │
    ├─> GroupNorm -> Swish -> Conv ├──> (+)
    │                              │
    └─> GroupNorm -> Swish -> Conv ┘
           └── Dropout ──┘
    ```

    If in_channels != out_channels, a 1x1 conv is applied to the shortcut.

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels (default: same as input)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        *,
        dim: SpatialDim,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = get_conv(dim, in_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = get_conv(dim, out_channels, out_channels, 3, 1, 1)

        # Shortcut projection if channel count changes
        self.nin_shortcut = (
            get_conv(dim, in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels_out *spatial"]:
        """Apply residual block transformation.

        Args:
            x: Input tensor

        Returns:
            Residual output: shortcut(x) + conv2(conv1(x))
        """
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """Self-attention block with efficient scaled dot-product attention.

    Applies multi-head self-attention over spatial dimensions to capture
    long-range dependencies. Uses PyTorch's optimized scaled_dot_product_attention
    which automatically selects the best kernel (Flash Attention when available).

    When to Use Attention
    ---------------------
    Attention is computationally expensive (O(n²) in spatial size), so it's
    typically only used at lower resolutions (e.g., 32x32 or 16x16) where
    the spatial dimension is small enough to be tractable.

    Memory Considerations
    ---------------------
    For 3D volumes, attention at full resolution is usually infeasible.
    Consider using attention only at the bottleneck resolution or using
    more efficient variants like linear attention.

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input/output channels
    """

    def __init__(self, dim: SpatialDim, in_channels: int) -> None:
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = get_conv(dim, in_channels, in_channels, 1, 1, 0)
        self.k = get_conv(dim, in_channels, in_channels, 1, 1, 0)
        self.v = get_conv(dim, in_channels, in_channels, 1, 1, 0)
        self.proj_out = get_conv(dim, in_channels, in_channels, 1, 1, 0)
        self.dim = dim

        # Precompute scale for attention (as plain float to avoid torch.compile graph breaks)
        self.scale: float = in_channels ** (-0.5)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Apply self-attention over spatial dimensions.

        Args:
            x: Input tensor

        Returns:
            Attention output with residual connection: x + attn(x)
        """
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Flatten spatial dimensions for attention
        # SDPA requires 4D: (B, num_heads, seq_len, head_dim)
        # We use 1 head with head_dim = channels
        if self.dim == 2:
            b, c, h, w = q.shape
            spatial_shape = (h, w)
            # Reshape to (B, 1, N, C) for attention where N = H*W
            q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
            k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
            v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        else:
            b, c, h, w, d = q.shape
            spatial_shape = (h, w, d)
            # Reshape to (B, 1, N, C) for attention where N = H*W*D
            q = q.reshape(b, c, h * w * d).transpose(1, 2).unsqueeze(1)
            k = k.reshape(b, c, h * w * d).transpose(1, 2).unsqueeze(1)
            v = v.reshape(b, c, h * w * d).transpose(1, 2).unsqueeze(1)

        # Use PyTorch's SDPA - let it choose the best backend automatically
        h_ = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back to spatial: (B, 1, N, C) -> (B, C, *spatial)
        h_ = h_.squeeze(1).transpose(1, 2).reshape(b, c, *spatial_shape)
        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """Progressive encoder for VAE/VQ-VAE architectures.

    The encoder progressively downsamples the input while increasing
    channel depth, producing a low-resolution, high-channel latent
    representation suitable for quantization or sampling.

    Architecture Overview
    ---------------------
    ::

        Input (H, W, D) -> conv_in -> [ResBlock x N] -> Downsample ->
                                      [ResBlock x N] -> Downsample ->
                                      ...
                                      [ResBlock x N] -> (optional mid) ->
                                      norm -> Swish -> conv_out -> Latent

    Each resolution level contains:
    - N residual blocks (increasing channel depth)
    - Optional attention blocks (at specified resolutions)
    - Downsampling (except at the last level)

    Memory Optimization
    -------------------
    For large 3D volumes, enable gradient checkpointing to trade compute
    for memory. This recomputes intermediate activations during backward
    pass instead of storing them.

    Args:
        dim: Spatial dimensionality (2 or 3)
        in_channels: Number of input channels
        channels: Base channel count (multiplied by channels_mult)
        channels_mult: Channel multiplier for each resolution level
        num_res_blocks: Number of residual blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
        resolution: Input resolution (for attention calculations)
        z_channels: Output latent channels
        spatial_compression: Total spatial compression factor
        use_checkpointing: Enable gradient checkpointing (default: False)
        **kwargs: Additional config (use_encoder_mid, patch_size, etc.)
    """

    def __init__(
        self,
        dim: SpatialDim,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        use_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.use_checkpointing = use_checkpointing
        self.dim = dim

        # NVIDIA MAISI encoder doesn't use mid blocks
        self.use_encoder_mid = kwargs.get("use_encoder_mid", True)

        # NVIDIA MAISI doesn't use nonlinearity between norm_out and conv_out
        self.use_output_nonlinearity = kwargs.get("use_output_nonlinearity", True)

        # Optional spatial compression (wavelets, patching)
        patch_size = kwargs.get("patch_size", 1)
        if patch_size > 1:
            self.compressor = SpatialCompressor(
                patch_size,
                kwargs.get("patch_method", "haar"),
                dim,
                resolution=kwargs.get("resolution", None),
                voronoi_jitter=kwargs.get("voronoi_jitter", 0.0),
            )
        else:
            self.compressor = nn.Identity()
        in_channels = in_channels * (patch_size**dim)

        # Calculate number of downsample operations
        self.num_downsamples = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )

        # Input convolution
        self.conv_in = get_conv(dim, in_channels, channels, 3, 1, 1)

        # Build downsampling blocks
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        dim=dim,
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(dim, block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = Downsample(dim, block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle blocks (optional - NVIDIA MAISI doesn't use them)
        if self.use_encoder_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(
                dim=dim, in_channels=block_in, out_channels=block_in, dropout=dropout
            )
            self.mid.attn_1 = (
                AttnBlock(dim, block_in)
                if len(attn_resolutions) > 0 and "mid" in str(attn_resolutions)
                else None
            )
            self.mid.block_2 = ResnetBlock(
                dim=dim, in_channels=block_in, out_channels=block_in, dropout=dropout
            )
        else:
            self.mid = None

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = get_conv(dim, block_in, z_channels, 3, 1, 1)

    def _checkpoint_forward(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply module with optional gradient checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch z_channels *spatial_compressed"]:
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, H, W, D)

        Returns:
            Latent tensor with reduced spatial dimensions
        """
        # Note: channels_last conversion removed - it confuses torch.compile
        # Use model.to(memory_format=torch.channels_last_3d) at model level instead

        # Optional spatial compression
        x = self.compressor(x)

        # Initial convolution
        h = self.conv_in(x)

        # Progressive downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self._checkpoint_forward(self.down[i_level].block[i_block], h)
                if len(self.down[i_level].attn) > i_block:
                    h = self.down[i_level].attn[i_block](h)

            if i_level < self.num_downsamples:
                h = self.down[i_level].downsample(h)

        # Middle blocks
        if self.mid is not None:
            h = self._checkpoint_forward(self.mid.block_1, h)
            if self.mid.attn_1 is not None:
                h = self.mid.attn_1(h)
            h = self._checkpoint_forward(self.mid.block_2, h)

        # Output
        h = self.norm_out(h)
        if self.use_output_nonlinearity:
            h = nonlinearity(h)
        h = self.conv_out(h)

        # NOTE: We intentionally do NOT call h.contiguous() here.
        # The tensor remains in channels_last_3d format, which is optimal for:
        # 1. The downstream quant_conv (1x1 conv, format-agnostic)
        # 2. The decoder (also uses channels_last_3d internally)
        # Only the final model output should be converted to contiguous if needed.

        return h


class Decoder(nn.Module):
    """Progressive decoder for VAE/VQ-VAE architectures.

    The decoder progressively upsamples the latent representation while
    decreasing channel depth, reconstructing the original input dimensions.
    Mirrors the encoder structure in reverse.

    Architecture Overview
    ---------------------
    ::

        Latent -> conv_in -> [mid blocks] ->
                  [ResBlock x N] -> Upsample ->
                  [ResBlock x N] -> Upsample ->
                  ...
                  [ResBlock x N] -> norm -> Swish -> conv_out -> Output

    NVIDIA MAISI Compatibility
    --------------------------
    The MAISI decoder has non-uniform block counts per stage ([2, 2, 0]).
    This is controlled by the `decoder_blocks_per_stage` argument.

    Args:
        dim: Spatial dimensionality (2 or 3)
        out_channels: Number of output channels
        channels: Base channel count
        channels_mult: Channel multiplier for each resolution level
        num_res_blocks: Default number of residual blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
        resolution: Output resolution
        z_channels: Input latent channels
        spatial_compression: Total spatial compression factor
        use_checkpointing: Enable gradient checkpointing (default: False)
        **kwargs: Additional config (decoder_blocks_per_stage, patch_size, etc.)
    """

    def __init__(
        self,
        dim: SpatialDim,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        use_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.use_checkpointing = use_checkpointing
        self.dim = dim

        # NVIDIA MAISI has different blocks per stage: [2, 2, 0]
        self.decoder_blocks_per_stage = kwargs.get("decoder_blocks_per_stage", None)

        # NVIDIA MAISI doesn't use nonlinearity between norm_out and conv_out
        self.use_output_nonlinearity = kwargs.get("use_output_nonlinearity", True)

        # Optional spatial decompression
        patch_size = kwargs.get("patch_size", 1)
        if patch_size > 1:
            self.decompressor = SpatialDecompressor(
                patch_size,
                kwargs.get("patch_method", "haar"),
                dim,
                resolution=kwargs.get("resolution", None),
                voronoi_jitter=kwargs.get("voronoi_jitter", 0.0),
                out_channels=out_channels,
            )
        else:
            self.decompressor = nn.Identity()
        out_ch = out_channels * (patch_size**dim)

        # Calculate number of upsample operations
        self.num_upsamples = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )

        # Input convolution (from latent)
        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.conv_in = get_conv(dim, z_channels, block_in, 3, 1, 1)

        # Middle blocks (always present in decoder)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            dim=dim, in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = (
            AttnBlock(dim, block_in)
            if len(attn_resolutions) > 0 and "mid" in str(attn_resolutions)
            else None
        )
        self.mid.block_2 = ResnetBlock(
            dim=dim, in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # Build upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]

            # Get blocks for this stage (MAISI uses different counts)
            if self.decoder_blocks_per_stage is not None:
                stage_blocks = self.decoder_blocks_per_stage[i_level]
            else:
                stage_blocks = self.num_res_blocks

            for _ in range(stage_blocks):
                block.append(
                    ResnetBlock(
                        dim=dim,
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(dim, block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= (self.num_resolutions - self.num_upsamples):
                up.upsample = Upsample(dim, block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = get_conv(dim, block_in, out_ch, 3, 1, 1)

    def _checkpoint_forward(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply module with optional gradient checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(
        self, z: Float[torch.Tensor, "batch z_channels *spatial_compressed"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode latent representation to output.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed output tensor
        """
        # Note: channels_last conversion removed - it confuses torch.compile
        # Use model.to(memory_format=torch.channels_last_3d) at model level instead

        h = self.conv_in(z)

        # Middle blocks
        h = self._checkpoint_forward(self.mid.block_1, h)
        if self.mid.attn_1 is not None:
            h = self.mid.attn_1(h)
        h = self._checkpoint_forward(self.mid.block_2, h)

        # Progressive upsampling
        for i_level in reversed(range(self.num_resolutions)):
            num_blocks = len(self.up[i_level].block)
            for i_block in range(num_blocks):
                h = self._checkpoint_forward(self.up[i_level].block[i_block], h)
                if len(self.up[i_level].attn) > i_block:
                    h = self.up[i_level].attn[i_block](h)

            if i_level >= (self.num_resolutions - self.num_upsamples):
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        if self.use_output_nonlinearity:
            h = nonlinearity(h)
        h = self.conv_out(h)

        # Apply decompression (if any) then convert to contiguous.
        # We convert to contiguous here because this is the FINAL model output
        # that users will receive. Unlike the Encoder output (which stays in
        # channels_last_3d for the decoder), the Decoder output should be in
        # standard contiguous format for user convenience and compatibility.
        h = self.decompressor(h)
        if self.dim == 3:
            h = h.contiguous()

        return h
