"""PatchGAN discriminator for local image realism."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float

from ...modules.utils import jaxtyped_compile_safe


def spectral_norm(module, use_sn: bool = True):
    """Apply spectral normalization if requested."""
    return nn.utils.spectral_norm(module) if use_sn else module


class PatchGAN(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(
        self,
        dim: int = 2,
        input_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d

        sequence = [
            spectral_norm(Conv(input_channels, ndf, 4, 2, 1), use_spectral_norm),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(
                    Conv(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1), use_spectral_norm
                ),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(
                Conv(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1), use_spectral_norm
            ),
            nn.LeakyReLU(0.2, True),
            Conv(ndf * nf_mult, 1, 4, 1, 1),
        ]

        self.model = nn.Sequential(*sequence)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels ..."]
    ) -> Float[torch.Tensor, "batch 1 ..."]:
        return self.model(x)
