"""Adversarial loss functions for GAN training."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float

from ...modules.utils import jaxtyped_compile_safe


class Adversarial(nn.Module):
    """Adversarial loss with multiple formulations."""

    def __init__(
        self, loss_type: Literal["hinge", "non_saturating", "vanilla"] = "hinge"
    ):
        super().__init__()
        self.loss_type = loss_type

    @jaxtyped_compile_safe(beartype)
    def discriminator_loss(
        self,
        real_logits: list[Float[torch.Tensor, "..."]],
        fake_logits: list[Float[torch.Tensor, "..."]],
    ) -> Float[torch.Tensor, ""]:
        if self.loss_type == "hinge":
            real_loss = torch.stack(
                [F.relu(1.0 - r).mean() for r in real_logits]
            ).sum() / len(real_logits)
            fake_loss = torch.stack(
                [F.relu(1.0 + f).mean() for f in fake_logits]
            ).sum() / len(fake_logits)
            return real_loss + fake_loss

        elif self.loss_type == "non_saturating":
            real_loss = torch.stack(
                [F.softplus(-r).mean() for r in real_logits]
            ).sum() / len(real_logits)
            fake_loss = torch.stack(
                [F.softplus(f).mean() for f in fake_logits]
            ).sum() / len(fake_logits)
            return real_loss + fake_loss

        else:  # vanilla
            real_loss = torch.stack(
                [
                    F.binary_cross_entropy_with_logits(r, torch.ones_like(r))
                    for r in real_logits
                ]
            ).sum() / len(real_logits)
            fake_loss = torch.stack(
                [
                    F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
                    for f in fake_logits
                ]
            ).sum() / len(fake_logits)
            return real_loss + fake_loss

    @jaxtyped_compile_safe(beartype)
    def generator_loss(
        self,
        fake_logits: list[Float[torch.Tensor, "..."]],
    ) -> Float[torch.Tensor, ""]:
        if self.loss_type == "hinge":
            return -torch.stack([f.mean() for f in fake_logits]).sum() / len(
                fake_logits
            )

        elif self.loss_type == "non_saturating":
            return torch.stack(
                [F.softplus(-f).mean() for f in fake_logits]
            ).sum() / len(fake_logits)

        else:  # vanilla
            return torch.stack(
                [
                    F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
                    for f in fake_logits
                ]
            ).sum() / len(fake_logits)


class R1Penalty(nn.Module):
    """R1 gradient penalty for discriminator regularization."""

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        real_images: Float[torch.Tensor, "batch channels ..."],
        real_logits: list[Float[torch.Tensor, "..."]],
    ) -> Float[torch.Tensor, ""]:
        grad_penalty: torch.Tensor = torch.tensor(0.0, device=real_images.device)
        for logits in real_logits:
            grad_real = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=real_images,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_penalty = (
                grad_penalty
                + (grad_real**2).reshape(grad_real.size(0), -1).sum(1).mean()
            )
        return self.weight * grad_penalty / len(real_logits)


class LeCAM(nn.Module):
    """LeCAM regularization for GAN training stability.

    Penalizes divergence between running means of real/fake discriminator logits,
    preventing mode collapse and training oscillation.
    Reference: Tseng et al., "Regularizing Generative Adversarial Networks under
    Limited Data" (CVPR 2021).
    """

    def __init__(self, weight: float = 0.001, decay: float = 0.999):
        super().__init__()
        self.weight = weight
        self.decay = decay
        self.register_buffer("real_mean", torch.tensor(0.0))
        self.register_buffer("fake_mean", torch.tensor(0.0))

    def forward(
        self,
        real_logits: list[Float[torch.Tensor, "..."]],
        fake_logits: list[Float[torch.Tensor, "..."]],
    ) -> Float[torch.Tensor, ""]:
        real_mean = torch.stack([r.mean() for r in real_logits]).mean()
        fake_mean = torch.stack([f.mean() for f in fake_logits]).mean()
        self.real_mean.lerp_(real_mean.detach(), 1 - self.decay)
        self.fake_mean.lerp_(fake_mean.detach(), 1 - self.decay)
        lecam = (
            F.relu(real_mean - self.fake_mean).mean()
            + F.relu(self.real_mean - fake_mean).mean()
        )
        return self.weight * lecam
