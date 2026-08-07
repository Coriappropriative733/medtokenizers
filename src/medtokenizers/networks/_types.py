"""Shared output types for tokenizer networks.

This module defines the single canonical :class:`NetworkEval` structure
returned by every tokenizer's ``forward`` in evaluation mode. Consolidating
the type here keeps the eval-mode contract identical across the continuous,
discrete, TiTok, RAE, and NVIDIA MAISI tokenizers.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch


class NetworkEval(NamedTuple):
    """Output structure for tokenizer evaluation mode.

    The superset of fields used by all tokenizer variants. ``reconstructions``
    is always populated; the remaining fields are populated only by the
    tokenizer families that produce them, and default to ``None`` otherwise.

    Attributes:
        reconstructions: Decoded output tensor (always present).
        posteriors: Distribution parameters for VAE-style tokenizers
            (e.g. ``(mean, logvar)``), ``None`` for tokenizers without a
            probabilistic posterior (AE/RAE/discrete).
        latent: Continuous latent tensor for continuous/RAE tokenizers,
            ``None`` for discrete tokenizers.
        quant_loss: Quantization loss (commitment, entropy, etc.) for discrete
            tokenizers, ``None`` for continuous tokenizers.
        quant_info: Discrete codebook indices for discrete tokenizers, ``None``
            for continuous tokenizers.
    """

    reconstructions: torch.Tensor
    posteriors: Optional[Union[tuple[torch.Tensor, ...], torch.Tensor]] = None
    latent: Optional[torch.Tensor] = None
    quant_loss: Optional[torch.Tensor] = None
    quant_info: Optional[torch.Tensor] = None


__all__ = ["NetworkEval"]
