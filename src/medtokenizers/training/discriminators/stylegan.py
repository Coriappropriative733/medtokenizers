"""StyleGAN2-inspired discriminator."""

import torch
import torch.nn as nn
from jaxtyping import Float

from .patch import spectral_norm


class StyleGAN(nn.Module):
    """StyleGAN2-style discriminator with progressive resolution."""

    def __init__(
        self,
        dim: int = 2,
        input_channels: int = 1,
        base_channels: int = 64,
        max_channels: int = 512,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.dim = dim
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d

        self.from_rgb = Conv(input_channels, base_channels, 1)

        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = min(base_channels * (2 ** (i + 1)), max_channels)
            self.blocks.append(self._make_block(in_ch, out_ch))
            in_ch = out_ch

        self.final_conv = Conv(in_ch, 1, 4, 1, 0)

    def _make_block(self, in_ch, out_ch):
        Conv = nn.Conv2d if self.dim == 2 else nn.Conv3d
        return nn.Sequential(
            spectral_norm(Conv(in_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(Conv(out_ch, out_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels ..."]
    ) -> Float[torch.Tensor, "batch 1 ..."]:
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)
