# Copyright 2026 Liam Chalcroft
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code derived from the MONAI / NVIDIA MAISI VAE Gaussian posterior
# (https://github.com/Project-MONAI/MONAI), originally licensed under the Apache License 2.0.
# GaussianDistribution follows the MAISI-style diagonal-Gaussian formulation (logvar clamping,
# reparameterized sampling, and KL divergence against an N(0, I) prior).
# See THIRD_PARTY_NOTICES.md for details.
"""Latent distribution modules for tokenizer regularization.

These modules sit between the encoder and the latent bottleneck and define how
the encoder output is interpreted as a (possibly stochastic) latent and which
regularization loss, if any, is applied to it.

Available distributions
------------------------
- ``GaussianDistribution``: MAISI-style diagonal-Gaussian VAE posterior. The
  encoder output is split into a mean and a log-variance, a latent is drawn via
  the reparameterization trick, and a per-sample KL divergence against an
  ``N(0, I)`` prior is returned as a regularization loss.
- ``IdentityDistribution``: A pass-through distribution for deterministic
  (autoencoder / discrete) tokenizers. It returns the encoder output unchanged
  and emits a zero regularization loss so downstream code can treat all
  distributions with a uniform interface.

Common interface
----------------
Each distribution is an ``nn.Module`` whose ``forward`` accepts the raw encoder
output ``parameters`` of shape ``(B, C, *spatial)`` and returns a tuple

    ``(latent, (kl_loss, extra))``

where ``latent`` is the latent passed to the decoder/quantizer, ``kl_loss`` is a
per-sample regularization loss, and ``extra`` carries any auxiliary tensors
(e.g. ``(mean, logvar)`` for the Gaussian, or a zero ``logvar`` placeholder for
the identity).
"""

from typing import Tuple

import torch


class IdentityDistribution(torch.nn.Module):
    """Pass-through distribution with zero regularization.

    Used by deterministic tokenizers (plain autoencoders and discrete
    quantizers) that do not impose a probabilistic prior on the latent. The
    encoder output is returned unchanged and the regularization loss is zero,
    preserving the same ``(latent, (kl_loss, extra))`` interface as
    :class:`GaussianDistribution`.
    """

    def __init__(self) -> None:
        """Initialize the identity distribution.

        Registers zero-valued buffers so that the regularization loss and the
        placeholder log-variance follow the module's device and dtype.
        """
        super().__init__()
        # Register buffers for consistent device/dtype handling
        self.register_buffer("zero_kl_div", torch.tensor([0.0]))
        self.register_buffer("zero_logvar", torch.tensor([0.0]))

    def forward(
        self, parameters: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return the input latent unchanged with a zero regularization loss.

        Args:
            parameters: Encoder output of shape ``(B, C, *spatial)``. Returned
                verbatim as the latent.

        Returns:
            A tuple ``(latent, (kl_loss, zero_logvar))`` where ``latent`` is the
            unmodified input of shape ``(B, C, *spatial)``, and both
            ``kl_loss`` and ``zero_logvar`` are zero tensors of shape
            ``(B, 1)`` on the same device and dtype as ``parameters``.
        """
        device = parameters.device
        dtype = parameters.dtype
        batch_size = parameters.shape[0]

        zero_kl = self.zero_kl_div.to(device=device, dtype=dtype).expand(batch_size, -1)
        zero_log = self.zero_logvar.to(device=device, dtype=dtype).expand(
            batch_size, -1
        )

        return parameters, (zero_kl, zero_log)


class GaussianDistribution(torch.nn.Module):
    """Diagonal-Gaussian VAE posterior with KL regularization.

    MAISI-style VAE regularization:
    - Splits the encoder output into a mean and a log-variance.
    - Draws a latent via the reparameterization trick.
    - Computes a per-sample KL divergence from an ``N(0, I)`` prior.

    The log-variance is clamped to ``[min_logvar, max_logvar]`` for numerical
    stability before sampling and KL computation.
    """

    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0) -> None:
        """Initialize the Gaussian distribution.

        Args:
            min_logvar: Lower clamp bound for the log-variance. Default ``-30.0``.
            max_logvar: Upper clamp bound for the log-variance. Default ``20.0``.
        """
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw a latent using the reparameterization trick.

        Computes ``z = mean + std * eps`` where ``std = exp(0.5 * logvar)`` and
        ``eps ~ N(0, I)``, keeping the sampling differentiable with respect to
        ``mean`` and ``logvar``.

        Args:
            mean: Posterior mean of shape ``(B, C, *spatial)``.
            logvar: Posterior log-variance of shape ``(B, C, *spatial)``.

        Returns:
            A latent sample of shape ``(B, C, *spatial)`` on the same device and
            dtype as ``mean``.
        """
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from an ``N(0, 1)`` prior.

        ``KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)``,
        summed over all non-batch dimensions.

        Args:
            mean: Posterior mean of shape ``(B, C, *spatial)``.
            logvar: Posterior log-variance of shape ``(B, C, *spatial)``.

        Returns:
            Per-sample KL divergence of shape ``(B,)`` (not reduced over the
            batch dimension).
        """
        # KL per element: -0.5 * (1 + logvar - mean^2 - exp(logvar))
        # Sum over all non-batch dimensions to avoid torch.compile graph breaks
        # from runtime .dim() checks
        kl_element = 1 + logvar - mean.pow(2) - logvar.exp()
        # Flatten all non-batch dimensions and sum
        kl = -0.5 * kl_element.flatten(start_dim=1).sum(dim=1)
        return kl

    def forward(
        self, parameters: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Split parameters, sample a latent, and compute the KL loss.

        The channel dimension of ``parameters`` is split in half into a mean and
        a log-variance; the log-variance is clamped to
        ``[min_logvar, max_logvar]`` before sampling and KL computation.

        Args:
            parameters: Encoder output of shape ``(B, 2 * C, *spatial)``, where
                the first ``C`` channels are the mean and the remaining ``C``
                channels are the log-variance.

        Returns:
            A tuple ``(latent, (kl_loss, (mean, logvar)))`` where:
            - ``latent`` is the sampled latent of shape ``(B, C, *spatial)``,
            - ``kl_loss`` is the per-sample KL divergence of shape ``(B,)``,
            - ``mean`` and ``logvar`` are the (clamped) posterior parameters,
              each of shape ``(B, C, *spatial)``.
        """
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        # Compute KL divergence
        kl_loss = self.kl_divergence(mean, logvar)

        # Sample latent
        z = self.sample(mean, logvar)

        # Return: (latent, (kl_loss, (mean, logvar)))
        return z, (kl_loss, (mean, logvar))
