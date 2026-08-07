# Copyright 2026 Liam Chalcroft
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code derived from NVIDIA Cosmos-Tokenizer
# (https://github.com/NVIDIA/Cosmos-Tokenizer), originally licensed under the Apache License 2.0.
# The _WAVELETS table and the Haar DWT/IDWT routines (_dwt_2d/_dwt_3d/_idwt_2d/_idwt_3d) used by
# SpatialCompressor/SpatialDecompressor are adapted from cosmos_tokenizer/modules/patching.py.
# The Voronoi tiling path is an original addition.
# See THIRD_PARTY_NOTICES.md for details.
"""Spatial compression modules for handling wavelet, rearrange, and Voronoi tilings."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False


class SpatialCompressorBase(nn.Module):
    """Base class for SpatialCompressor and SpatialDecompressor with shared functionality."""

    def __init__(
        self,
        patch_size=2,
        patch_method="haar",
        dim=2,
        resolution=None,
        voronoi_jitter=0.0,
    ):
        super().__init__()
        if patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of 2")
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.dim = dim
        self.voronoi_jitter = voronoi_jitter

        if resolution is not None:
            self.resolution = (
                (resolution,) * dim
                if isinstance(resolution, int)
                else tuple(resolution)
            )
        else:
            self.resolution = None

        # Register wavelets buffer
        if patch_method in ("haar", "rearrange"):
            self.register_buffer(
                "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
            )
            self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
            self.register_buffer(
                "_arange",
                torch.arange(_WAVELETS[patch_method].shape[0]),
                persistent=_PERSISTENT,
            )
        elif patch_method == "voronoi":
            if self.resolution is None:
                raise ValueError("resolution must be provided for voronoi patching")
            self._setup_voronoi_boundaries()
        else:
            raise ValueError(f"Unknown patch method: {patch_method}")

        for param in self.parameters():
            param.requires_grad = False

    def _setup_voronoi_boundaries(self):
        """Setup Voronoi boundaries (shared between SpatialCompressor and SpatialDecompressor)."""
        jitter_scale = max(self.voronoi_jitter, 0.0)
        for axis, length in enumerate(self.resolution):
            cells = length // self.patch_size
            if cells * self.patch_size != length:
                raise ValueError(
                    "resolution must be divisible by patch_size for voronoi tiling"
                )
            boundaries = torch.linspace(0, length, steps=cells + 1)
            if jitter_scale > 0 and cells > 1:
                offsets = (
                    torch.sin(torch.linspace(0, math.pi, steps=cells - 1))
                    * jitter_scale
                    * self.patch_size
                )
                mids = boundaries[1:-1] + offsets
                mids = torch.clamp(mids, 1.0, length - 1.0)
                boundaries[1:-1] = mids
            boundaries = boundaries.round().long()
            boundaries[0] = 0
            boundaries[-1] = length
            for idx in range(1, boundaries.numel()):
                if boundaries[idx] <= boundaries[idx - 1]:
                    boundaries[idx] = min(length, boundaries[idx - 1] + 1)
            boundaries[-1] = length
            self.register_buffer(
                f"voronoi_boundaries_{axis}", boundaries, persistent=_PERSISTENT
            )

    def _voronoi_bounds(self, axis: int):
        return getattr(self, f"voronoi_boundaries_{axis}")


class SpatialCompressor(SpatialCompressorBase):
    """Compress spatial dimensions into channel dimensions using several tilings."""

    def _dwt_2d(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n, g = h.shape[0], x.shape[1]

        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
        hh = (
            (h * ((-1) ** self._arange))
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype)
        )

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        return out / 2 if rescale else out

    def _dwt_3d(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n, g = h.shape[0], x.shape[1]

        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
        hh = (
            (h * ((-1) ** self._arange))
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype)
        )

        x = F.pad(
            x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode
        ).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        return out / (2 * torch.sqrt(torch.tensor(2.0))) if rescale else out

    def _haar(self, x):
        for _ in self.range:
            x = (
                self._dwt_2d(x, rescale=True)
                if self.dim == 2
                else self._dwt_3d(x, rescale=True)
            )
        if self.dim == 3 and x.size(2) - x.size(3) == 1:
            x = x[:, :, :-1, :, :]
        return x

    def _arrange(self, x):
        if self.dim == 2:
            return rearrange(
                x,
                "b c (h p1) (w p2) -> b (c p1 p2) h w",
                p1=self.patch_size,
                p2=self.patch_size,
            ).contiguous()
        return rearrange(
            x,
            "b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        ).contiguous()

    def _voronoi_2d(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.resolution:
            raise ValueError(
                "input resolution does not match configured voronoi resolution"
            )
        y_bounds, x_bounds = self._voronoi_bounds(0), self._voronoi_bounds(1)
        num_y, num_x = y_bounds.numel() - 1, x_bounds.numel() - 1
        patches = []
        for iy in range(num_y):
            for ix in range(num_x):
                patch = x[
                    :,
                    :,
                    y_bounds[iy] : y_bounds[iy + 1],
                    x_bounds[ix] : x_bounds[ix + 1],
                ]
                patch = F.interpolate(
                    patch,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                patches.append(patch)
        patches = torch.stack(patches, dim=2).view(
            b, c, self.patch_size**self.dim, num_y, num_x
        )
        return rearrange(patches, "b c p hy hx -> b (c p) hy hx")

    def _voronoi_3d(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.shape
        if (h, w, d) != self.resolution:
            raise ValueError(
                "input resolution does not match configured voronoi resolution"
            )
        y_bounds, x_bounds, z_bounds = (
            self._voronoi_bounds(0),
            self._voronoi_bounds(1),
            self._voronoi_bounds(2),
        )
        num_y, num_x, num_z = (
            y_bounds.numel() - 1,
            x_bounds.numel() - 1,
            z_bounds.numel() - 1,
        )
        patches = []
        for iy in range(num_y):
            for ix in range(num_x):
                for iz in range(num_z):
                    patch = x[
                        :,
                        :,
                        y_bounds[iy] : y_bounds[iy + 1],
                        x_bounds[ix] : x_bounds[ix + 1],
                        z_bounds[iz] : z_bounds[iz + 1],
                    ]
                    patch = patch.permute(0, 1, 4, 2, 3)
                    patch = F.interpolate(
                        patch,
                        size=(self.patch_size, self.patch_size, self.patch_size),
                        mode="trilinear",
                        align_corners=False,
                    )
                    patch = patch.permute(0, 1, 3, 4, 2)
                    patches.append(patch)
        patches = torch.stack(patches, dim=2).view(
            b, c, self.patch_size**self.dim, num_y, num_x, num_z
        )
        return rearrange(patches, "b c p hy hx hz -> b (c p) hy hx hz")

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        if self.patch_method == "rearrange":
            return self._arrange(x)
        if self.patch_method == "voronoi":
            return self._voronoi_2d(x) if self.dim == 2 else self._voronoi_3d(x)
        raise ValueError(f"Unknown patch method: {self.patch_method}")


class SpatialDecompressor(SpatialCompressorBase):
    """Reconstruct original spatial layout from compressed channel representation."""

    def __init__(
        self,
        patch_size=2,
        patch_method="haar",
        dim=2,
        resolution=None,
        voronoi_jitter=0.0,
        out_channels=None,
    ):
        super().__init__(patch_size, patch_method, dim, resolution, voronoi_jitter)
        self.base_channels = out_channels

        if patch_method == "voronoi" and out_channels is None:
            raise ValueError("out_channels must be provided for voronoi unpatching")

    def _idwt_2d(self, x, rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n, g = h.shape[0], x.shape[1] // 4

        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype=dtype)
        hh = (
            (h * ((-1) ** self._arange))
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype)
        )

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        yl = F.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yl += F.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh = F.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh += F.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        y = F.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )
        y += F.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )

        return y * 2 if rescale else y

    def _idwt_3d(self, x, rescale=False):
        dtype = x.dtype
        h = self.wavelets
        g = x.shape[1] // 8

        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype=dtype)
        hh = (
            (h * ((-1) ** self._arange))
            .reshape(1, 1, -1)
            .repeat(g, 1, 1)
            .to(dtype=dtype)
        )

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        xll = F.conv_transpose3d(
            xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xll += F.conv_transpose3d(
            xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xlh = F.conv_transpose3d(
            xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xlh += F.conv_transpose3d(
            xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhl = F.conv_transpose3d(
            xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhl += F.conv_transpose3d(
            xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhh = F.conv_transpose3d(
            xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhh += F.conv_transpose3d(
            xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xl = F.conv_transpose3d(
            xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xl += F.conv_transpose3d(
            xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh = F.conv_transpose3d(
            xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh += F.conv_transpose3d(
            xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )

        x = F.conv_transpose3d(
            xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )
        x += F.conv_transpose3d(
            xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )

        return x * (2 * torch.sqrt(torch.tensor(2.0))) if rescale else x

    def _ihaar(self, x):
        for _ in self.range:
            x = (
                self._idwt_2d(x, rescale=True)
                if self.dim == 2
                else self._idwt_3d(x, rescale=True)
            )
        if self.dim == 3 and x.size(2) - x.size(3) == 1:
            x = x[:, :, :-1, :, :]
        return x

    def _iarrange(self, x):
        if self.dim == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.patch_size,
                p2=self.patch_size,
            )
        return rearrange(
            x,
            "b (c p1 p2 p3) h w d -> b c (h p1) (w p2) (d p3)",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )

    def _inverse_voronoi_2d(self, x: torch.Tensor) -> torch.Tensor:
        b, _, hy, hx = x.shape
        num_y, num_x = (
            len(self._voronoi_bounds(0)) - 1,
            len(self._voronoi_bounds(1)) - 1,
        )
        if (hy, hx) != (num_y, num_x):
            raise ValueError("Patch grid does not match Voronoi configuration")
        patches = x.view(b, self.base_channels, self.patch_size**self.dim, hy, hx)
        patches = patches.view(
            b, self.base_channels, self.patch_size, self.patch_size, hy, hx
        )
        output = torch.zeros(
            b, self.base_channels, *self.resolution, device=x.device, dtype=x.dtype
        )
        for iy in range(hy):
            for ix in range(hx):
                y0, y1 = (
                    self._voronoi_bounds(0)[iy].item(),
                    self._voronoi_bounds(0)[iy + 1].item(),
                )
                x0, x1 = (
                    self._voronoi_bounds(1)[ix].item(),
                    self._voronoi_bounds(1)[ix + 1].item(),
                )
                patch = F.interpolate(
                    patches[:, :, :, :, iy, ix],
                    size=(y1 - y0, x1 - x0),
                    mode="bilinear",
                    align_corners=False,
                )
                output[:, :, y0:y1, x0:x1] = patch
        return output

    def _inverse_voronoi_3d(self, x: torch.Tensor) -> torch.Tensor:
        b, _, hy, hx, hz = x.shape
        dims = [len(self._voronoi_bounds(axis)) - 1 for axis in range(3)]
        if (hy, hx, hz) != tuple(dims):
            raise ValueError("Patch grid does not match Voronoi configuration")
        patches = x.view(b, self.base_channels, self.patch_size**self.dim, hy, hx, hz)
        patches = patches.view(
            b,
            self.base_channels,
            self.patch_size,
            self.patch_size,
            self.patch_size,
            hy,
            hx,
            hz,
        )
        output = torch.zeros(
            b, self.base_channels, *self.resolution, device=x.device, dtype=x.dtype
        )
        for iy in range(hy):
            for ix in range(hx):
                for iz in range(hz):
                    y0, y1 = (
                        self._voronoi_bounds(0)[iy].item(),
                        self._voronoi_bounds(0)[iy + 1].item(),
                    )
                    x0, x1 = (
                        self._voronoi_bounds(1)[ix].item(),
                        self._voronoi_bounds(1)[ix + 1].item(),
                    )
                    z0, z1 = (
                        self._voronoi_bounds(2)[iz].item(),
                        self._voronoi_bounds(2)[iz + 1].item(),
                    )
                    patch = patches[:, :, :, :, :, iy, ix, iz].permute(0, 1, 4, 2, 3)
                    patch = F.interpolate(
                        patch,
                        size=(z1 - z0, y1 - y0, x1 - x0),
                        mode="trilinear",
                        align_corners=False,
                    )
                    output[:, :, y0:y1, x0:x1, z0:z1] = patch.permute(0, 1, 3, 4, 2)
        return output

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        if self.patch_method == "rearrange":
            return self._iarrange(x)
        if self.patch_method == "voronoi":
            return (
                self._inverse_voronoi_2d(x)
                if self.dim == 2
                else self._inverse_voronoi_3d(x)
            )
        raise ValueError(f"Unknown patch method: {self.patch_method}")
