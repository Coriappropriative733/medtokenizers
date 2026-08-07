"""Multi-scale discriminator for hierarchical image analysis."""

import logging

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float

from ...modules.utils import jaxtyped_compile_safe
from .patch import PatchGAN

logger = logging.getLogger(__name__)


def _patchgan_min_input(n_layers: int) -> int:
    """Compute the smallest spatial size a PatchGAN can accept.

    A PatchGAN applies ``n_layers`` stride-2 convolutions followed by two
    stride-1 convolutions, all with kernel size 4 and padding 1. Each
    convolution requires its (padded) input to be at least as large as the
    kernel, so the network only produces a valid output once the input is big
    enough for the final stride-1 convolution to see a full receptive field.

    Args:
        n_layers: Number of stride-2 convolutional layers in the PatchGAN.

    Returns:
        The minimum spatial size (per dimension) for which the PatchGAN
        produces a non-empty output.
    """

    def conv_out(size: int, stride: int) -> int:
        # Kernel 4, padding 1; output is empty once the padded size < kernel.
        if size + 2 < 4:
            return 0
        return (size + 2 - 4) // stride + 1

    size = 1
    while True:
        cur = size
        valid = True
        for _ in range(n_layers):
            cur = conv_out(cur, stride=2)
            if cur < 1:
                valid = False
                break
        if valid:
            for _ in range(2):
                cur = conv_out(cur, stride=1)
                if cur < 1:
                    valid = False
                    break
        if valid and cur >= 1:
            return size
        size += 1


class MultiScale(nn.Module):
    """Multi-scale discriminator operating at multiple resolutions."""

    def __init__(
        self,
        dim: int = 2,
        input_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        # Honor the requested number of scales: build exactly ``num_scales``
        # PatchGAN discriminators, each with the requested ``n_layers``. Small
        # inputs are handled at runtime in ``forward`` rather than by shrinking
        # the configuration here, where the real input size is unknown.
        self.discriminators = nn.ModuleList(
            [
                PatchGAN(dim, input_channels, ndf, n_layers, use_spectral_norm)
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool2d(3, 2, 1) if dim == 2 else nn.AvgPool3d(3, 2, 1)
        self._min_input = _patchgan_min_input(n_layers)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels ..."]
    ) -> list[Float[torch.Tensor, "batch 1 ..."]]:
        """Run each scale while the downsampled input remains large enough.

        For normal inputs every scale runs and the outputs have strictly
        decreasing spatial size. For genuinely small inputs the scales whose
        progressively-downsampled resolution would underflow the PatchGAN
        receptive field are skipped, so only the valid outputs are returned.

        Args:
            x: Input tensor of shape ``(batch, channels, *spatial)``.

        Returns:
            A list of per-scale discriminator outputs, ordered from the
            highest resolution to the lowest.
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            if min(x.shape[2:]) < self._min_input:
                logger.debug(
                    "Stopping MultiScale at scale %d: spatial size %s is below "
                    "the PatchGAN minimum input %d.",
                    i,
                    tuple(x.shape[2:]),
                    self._min_input,
                )
                break
            outputs.append(disc(x))
        return outputs
