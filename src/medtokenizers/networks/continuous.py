"""Continuous latent space tokenizers (VAE/AE).

This module implements tokenizers that map images to continuous latent
representations, optionally with variational regularization (VAE).

Continuous vs Discrete Tokenizers
---------------------------------
Continuous tokenizers (this module):
- Produce real-valued latent vectors z ∈ ℝ^d
- Suitable for diffusion model latent spaces
- VAE variant regularizes toward N(0,1) prior

Discrete tokenizers (see discrete.py):
- Produce integer codes from finite vocabulary
- Suitable for autoregressive/transformer models
- Require quantization layer (VQ, FSQ, etc.)

VAE Theory
----------
The Variational Autoencoder learns:
- Encoder q(z|x): Maps input to latent distribution
- Decoder p(x|z): Reconstructs input from latent

Training objective (ELBO):
    L = E_q[log p(x|z)] - β * KL(q(z|x) || p(z))

Where:
- First term: reconstruction quality
- Second term: regularization toward prior p(z) = N(0,1)
- β: KL weight (β=1 is standard VAE, β<1 for better reconstruction)

Architecture
------------
```
Input -> Encoder -> μ, σ² -> Sample z -> Decoder -> Reconstruction
                           └── Reparameterization trick
                               z = μ + σ * ε, ε ~ N(0,1)
```

For AE (no KL), encoder outputs z directly without sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
import torch.nn as nn

from medtokenizers.modules.base import BaseTokenizer
from medtokenizers.modules.distributions import (
    GaussianDistribution,
    IdentityDistribution,
)
from medtokenizers.modules.layers import Decoder, Encoder
from medtokenizers.modules.utils import validate_tensor_input
from medtokenizers.networks._types import NetworkEval

if TYPE_CHECKING:
    from jaxtyping import Float


FormulationType = Literal["VAE", "AE"]


class ContinuousTokenizer(BaseTokenizer):
    """Continuous latent tokenizer for medical imaging (VAE/AE).

    This tokenizer learns a continuous latent representation using either:
    - **VAE**: Variational Autoencoder with KL divergence regularization
    - **AE**: Standard Autoencoder without probabilistic modeling

    The VAE variant is particularly useful for:
    - Latent diffusion models (LDM)
    - Interpolation in latent space
    - Generative modeling with controllable sampling

    Architecture Details
    --------------------
    The network follows a symmetric encoder-decoder design::

        Encoder Path:
        Input(H,W,D) -> Conv_in -> ResBlocks -> Downsample -> ... -> Conv_out -> mu, sigma^2

        Decoder Path:
        z -> Conv_in -> ResBlocks -> Upsample -> ... -> Conv_out -> Output(H,W,D)

    Key architectural choices:
    - **quant_conv**: 1x1 conv reducing encoder output to latent dimension
    - **post_quant_conv**: 1x1 conv expanding latent to decoder input
    - **GroupNorm**: Batch-size independent normalization
    - **Swish activation**: Smooth, non-monotonic activation

    Memory Optimization
    -------------------
    For 3D volumes, the model automatically:
    - Uses channels_last_3d memory format for better cache efficiency
    - Supports gradient checkpointing (via use_checkpointing kwarg)

    Args:
        dim: Spatial dimensionality (2 for 2D images, 3 for 3D volumes)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (usually same as in_channels)
        z_channels: Intermediate channels after encoder, before quant_conv
        z_factor: Multiplier for encoder output channels. Default: 2 for VAE
                 (outputs μ and σ²), 1 for AE (outputs z directly)
        latent_channels: Final latent dimension (e.g., 4 for SD-style VAE)
        channels: Base channel count (scaled by channels_mult)
        channels_mult: Channel multipliers at each resolution level.
                      Example: (1, 2, 4) means channels → 2*channels → 4*channels
        num_res_blocks: Number of residual blocks per resolution level
        attn_resolutions: Spatial resolutions where self-attention is applied
        dropout: Dropout probability in residual blocks
        resolution: Input spatial resolution (for attention position info)
        spatial_compression: Total downsampling factor (e.g., 8 = 3 downsamples)
        formulation: "VAE" for variational, "AE" for deterministic
        name: Model identifier for saving/loading
        **kwargs: Additional args passed to Encoder/Decoder (e.g., use_checkpointing)

    Example:
        >>> # Create a 3D VAE with 4-channel latent (like Stable Diffusion)
        >>> model = ContinuousTokenizer(
        ...     dim=3,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     z_channels=128,
        ...     latent_channels=4,
        ...     channels=64,
        ...     channels_mult=(1, 2, 4),
        ...     spatial_compression=8,
        ...     formulation='VAE'
        ... )
        >>>
        >>> # Forward pass returns dict with reconstructions and KL loss
        >>> volume = torch.randn(1, 1, 128, 128, 128)
        >>> output = model(volume)
        >>> recon = output['reconstructions']
        >>> kl_loss = output.get('kl_loss')  # Only for VAE
        >>>
        >>> # For inference, use tokenize/detokenize
        >>> with model.inference_mode():
        ...     latents = model.tokenize(volume)  # (1, 4, 16, 16, 16)
        ...     reconstructed = model.detokenize(latents)

    References:
        Kingma & Welling "Auto-Encoding Variational Bayes" (2013)
        Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
    """

    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        z_channels: int = 4,
        z_factor: Optional[int] = None,
        latent_channels: int = 4,
        channels: int = 64,
        channels_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        spatial_compression: int = 4,
        formulation: FormulationType = "VAE",
        use_encoder_mid: bool = False,
        use_output_nonlinearity: bool = False,
        decoder_blocks_per_stage: Optional[list[int]] = None,
        separate_quant_conv: bool = True,
        name: str = "ContinuousTokenizer",
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, name=name)

        # Validate inputs
        if dim not in [2, 3]:
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        if formulation not in ["AE", "VAE"]:
            raise ValueError(f"formulation must be 'AE' or 'VAE', got {formulation}")

        # Default decoder_blocks_per_stage to MAISI-compatible [2, 2, 0] if not provided
        if decoder_blocks_per_stage is None:
            decoder_blocks_per_stage = [2, 2, 0]

        # Store config for serialization
        self.config = {
            "dim": dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "z_channels": z_channels,
            "z_factor": z_factor,
            "latent_channels": latent_channels,
            "channels": channels,
            "channels_mult": list(channels_mult),
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": list(attn_resolutions),
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "formulation": formulation,
            "use_encoder_mid": use_encoder_mid,
            "use_output_nonlinearity": use_output_nonlinearity,
            "decoder_blocks_per_stage": decoder_blocks_per_stage,
            "separate_quant_conv": separate_quant_conv,
            "name": name,
        }
        self.config.update(kwargs)

        self.latent_channels = latent_channels
        self.spatial_compression = spatial_compression
        self.formulation = formulation
        self.separate_quant_conv = separate_quant_conv and (formulation == "VAE")

        # Prepare kwargs for encoder/decoder
        layer_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channels": channels,
            "channels_mult": channels_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "use_encoder_mid": use_encoder_mid,
            "use_output_nonlinearity": use_output_nonlinearity,
            "decoder_blocks_per_stage": decoder_blocks_per_stage,
        }

        # Validate that spatial_compression divides resolution evenly
        if resolution % spatial_compression != 0:
            raise ValueError(
                f"Resolution {resolution}x{resolution} is not divisible by "
                f"spatial_compression={spatial_compression}. Expected latent size would be "
                f"({resolution // spatial_compression}x{resolution // spatial_compression}) but this causes "
                f"shape mismatch with target. Use compression that divides resolution evenly "
                f"(powers of 2: 1, 2, 4, 8, 16)."
            )
        layer_kwargs.update(kwargs)

        conv_class = nn.Conv2d if dim == 2 else nn.Conv3d

        if self.separate_quant_conv:
            # MAISI-style: encoder outputs z_channels, separate 1x1 convs for mu/sigma
            self.encoder = Encoder(dim=dim, z_channels=z_channels, **layer_kwargs)
            self.quant_conv_mu = conv_class(z_channels, latent_channels, kernel_size=1)
            self.quant_conv_log_sigma = conv_class(
                z_channels, latent_channels, kernel_size=1
            )
            self.quant_conv = None
        else:
            # Standard VAE: encoder outputs 2*z_channels, combined quant_conv
            if z_factor is None:
                z_factor = 2 if formulation == "VAE" else 1
            self.encoder = Encoder(
                dim=dim, z_channels=z_factor * z_channels, **layer_kwargs
            )
            self.quant_conv = conv_class(
                z_factor * z_channels, z_factor * latent_channels, kernel_size=1
            )
            self.quant_conv_mu = None
            self.quant_conv_log_sigma = None

        self.decoder = Decoder(dim=dim, z_channels=z_channels, **layer_kwargs)
        self.post_quant_conv = conv_class(latent_channels, z_channels, kernel_size=1)

        # Distribution for VAE vs AE (only used when not separate_quant_conv)
        self.distribution = (
            GaussianDistribution() if formulation == "VAE" else IdentityDistribution()
        )

    def encode(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Float[torch.Tensor, "batch latent_channels *spatial_compressed"],
        tuple[torch.Tensor, ...],
    ]:
        """Encode input to latent representation.

        For VAE:
        - Encoder outputs (μ, log σ²)
        - Samples z using reparameterization: z = μ + σ * ε
        - Returns (z, (kl_loss, (mean, logvar)))

        For AE:
        - Encoder outputs z directly
        - Returns (z, (zero_kl, zero_logvar))

        Args:
            x: Input tensor of shape ``(B, C, *spatial)`` where:
                - B: batch size
                - C: number of channels (must match model's in_channels)
                - spatial: (H, W) for 2D or (H, W, D) for 3D

        Returns:
            Tuple of:
            - latent: Sampled or deterministic latent tensor
            - distribution_output: KL loss and posterior parameters

        Raises:
            TypeError: If x is not a floating point tensor
            ValueError: If x has wrong shape or contains NaN/Inf
        """
        validate_tensor_input(x, self.dim, self.config["in_channels"], "encode")

        h = self.encoder(x)

        if self.separate_quant_conv:
            assert self.quant_conv_mu is not None
            assert self.quant_conv_log_sigma is not None
            mu = self.quant_conv_mu(h)
            log_sigma = self.quant_conv_log_sigma(h)

            if self.training:
                std = torch.exp(log_sigma)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu

            kl_loss = 0.5 * torch.sum(
                mu.pow(2) + log_sigma.exp().pow(2) - 1 - 2 * log_sigma
            )
            return z, (kl_loss, (mu, log_sigma))
        else:
            assert self.quant_conv is not None
            moments = self.quant_conv(h)
            return self.distribution(moments)

    def decode(
        self, z: Float[torch.Tensor, "batch latent_channels *spatial_compressed"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode latent representation to output.

        Args:
            z: Latent tensor from encode() or external source.
               Shape: ``(B, latent_channels, *spatial_compressed)``

        Returns:
            Reconstructed output with original spatial dimensions

        Raises:
            TypeError: If z is not a floating point tensor
            ValueError: If z has wrong shape or contains NaN/Inf
        """
        validate_tensor_input(z, self.dim, self.latent_channels, "decode")

        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self, input: Float[torch.Tensor, "batch channels *spatial"]
    ) -> dict[str, torch.Tensor] | NetworkEval:
        """Full forward pass: encode -> decode.

        During training, returns a dict with all outputs for loss computation.
        During evaluation, returns a NetworkEval namedtuple.

        Args:
            input: Input tensor

        Returns:
            Training mode (dict):
                - 'reconstructions': Decoded output
                - 'posteriors': (mean, logvar) for VAE, or identity for AE
                - 'latent'/'latents': Sampled latent tensor
                - 'kl_loss': KL divergence (VAE only)

            Eval mode (NetworkEval):
                - reconstructions: Decoded output
                - posteriors: Distribution parameters
                - latent: Sampled latent
        """
        latent, distribution_output = self.encode(input)
        reconstructions = self.decode(latent)

        # Parse distribution output based on formulation
        # GaussianDistribution: (kl_loss, (mean, logvar))
        # IdentityDistribution: (zero_kl, zero_logvar)
        if isinstance(distribution_output, tuple) and len(distribution_output) == 2:
            if isinstance(distribution_output[1], tuple):
                # VAE: GaussianDistribution format
                kl_loss, (mean, logvar) = distribution_output
                posteriors = (mean, logvar)

                if self.training:
                    return {
                        "reconstructions": reconstructions,
                        "posteriors": posteriors,
                        "kl_loss": kl_loss,
                        "latent": latent,
                        "latents": latent,
                    }
            else:
                # AE: IdentityDistribution format
                posteriors = distribution_output

        if self.training:
            return {
                "reconstructions": reconstructions,
                "posteriors": posteriors,
                "latent": latent,
                "latents": latent,
            }

        return NetworkEval(
            reconstructions=reconstructions, posteriors=posteriors, latent=latent
        )

    def tokenize(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch latent_channels *spatial_compressed"]:
        """Encode input to latent representation (convenience method).

        For inference, this is the primary encoding method. For VAE, returns
        the sampled latent (not the mean), enabling diverse reconstructions.

        Args:
            x: Input tensor

        Returns:
            Latent tensor suitable for storage, manipulation, or decoding
        """
        return self.encode(x)[0]

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output latent shape for given input shape.

        Useful for pre-allocating memory or understanding compression ratio.

        Args:
            input_shape: Input tensor shape (B, C, H, W) or (B, C, H, W, D)

        Returns:
            Expected latent shape (B, latent_channels, H', W') or (B, C', H', W', D')
            where spatial dims are compressed by spatial_compression factor
        """
        b = input_shape[0]
        compression = self.spatial_compression

        if self.dim == 2:
            h, w = input_shape[2], input_shape[3]
            return (b, self.latent_channels, h // compression, w // compression)
        else:
            h, w, d = input_shape[2], input_shape[3], input_shape[4]
            return (
                b,
                self.latent_channels,
                h // compression,
                w // compression,
                d // compression,
            )
