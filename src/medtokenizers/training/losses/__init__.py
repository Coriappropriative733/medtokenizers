"""Loss functions for tokenizer training."""

from .adversarial import Adversarial, LeCAM, R1Penalty
from .basic import Combined, CombinedPerceptual, Reconstruction
from .compound import VAEGANLoss, VQGANLoss
from .perceptual import SSIM3D, GramLoss, VGGFeatureLoss

__all__ = [
    "Reconstruction",
    "Combined",
    "CombinedPerceptual",
    "Adversarial",
    "R1Penalty",
    "LeCAM",
    "GramLoss",
    "VGGFeatureLoss",
    "SSIM3D",
    "VQGANLoss",
    "VAEGANLoss",
]
