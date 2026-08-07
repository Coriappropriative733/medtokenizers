"""Discriminator networks for adversarial training."""

from .multiscale import MultiScale
from .patch import PatchGAN
from .stylegan import StyleGAN

__all__ = [
    "PatchGAN",
    "MultiScale",
    "StyleGAN",
]
