"""Representation Autoencoder (RAE) tokenizer.

Implements a frozen representation encoder + trainable decoder pipeline,
following the RAE formulation (Zheng et al., 2025). The encoder is kept
frozen and provides token-level embeddings, while the decoder learns to
reconstruct images or volumes from those embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import torch
import torch.nn as nn

from medtokenizers.modules.base import BaseTokenizer
from medtokenizers.networks._types import NetworkEval

EncoderType = Literal["vit", "medsiglip", "neurovfm"]


@dataclass
class EncoderOutput:
    """Container for encoder outputs."""

    tokens: torch.Tensor
    grid_shape: tuple[int, ...] | None


class ViTEncoderAdapter(nn.Module):
    """Adapter for frozen HuggingFace ViT/SigLIP-style encoders."""

    def __init__(
        self,
        encoder_name_or_path: str,
        image_size: int | None = None,
        drop_cls_token: bool = True,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        encoder_kwargs = encoder_kwargs or {}
        self.config = AutoConfig.from_pretrained(encoder_name_or_path)
        self.processor = AutoImageProcessor.from_pretrained(encoder_name_or_path)
        self.model = AutoModel.from_pretrained(encoder_name_or_path, **encoder_kwargs)
        self.model.requires_grad_(False)
        self.drop_cls_token = drop_cls_token

        self.hidden_size = getattr(self.config, "hidden_size", None) or getattr(
            self.config, "hidden_dim", None
        )
        if self.hidden_size is None:
            raise ValueError(
                "Unable to infer hidden size from encoder config. "
                "Please provide a compatible ViT-style encoder."
            )

        self.patch_size = getattr(self.config, "patch_size", None)
        if isinstance(self.patch_size, (list, tuple)):
            self.patch_size = int(self.patch_size[0])

        if self.patch_size is None:
            raise ValueError(
                "Unable to infer patch size from encoder config. "
                "Please provide a ViT-style encoder with patch_size in the config."
            )

        if image_size is None:
            image_size = getattr(self.config, "image_size", None)
            if isinstance(image_size, (list, tuple)):
                image_size = int(image_size[0])
            if image_size is None:
                size_from_proc = getattr(self.processor, "size", None)
                if isinstance(size_from_proc, dict):
                    image_size = int(size_from_proc.get("shortest_edge", 0)) or int(
                        size_from_proc.get("height", 0)
                    )
                elif isinstance(size_from_proc, (list, tuple)):
                    image_size = int(size_from_proc[0])
        if image_size is None:
            raise ValueError(
                "Unable to infer image size from encoder config or processor. "
                "Please pass image_size explicitly."
            )

        self.image_size = image_size
        self.expected_channels = getattr(self.config, "num_channels", 3)

        mean = getattr(self.processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(self.processor, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer(
            "encoder_mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "encoder_std", torch.tensor(std).view(1, -1, 1, 1), persistent=False
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.encoder_mean.shape[1]:
            if x.shape[1] == 1 and self.encoder_mean.shape[1] == 3:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise ValueError(
                    "Input channels do not match encoder expectation. "
                    f"Got {x.shape[1]} channels, expected {self.encoder_mean.shape[1]}."
                )
        return (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        if x.dim() != 4:
            raise ValueError("ViT encoder expects 4D input (B, C, H, W).")

        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )

        x = self._normalize(x)
        outputs = self.model(pixel_values=x)
        tokens = outputs.last_hidden_state

        if self.drop_cls_token and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]

        grid_dim = self.image_size // self.patch_size
        grid_shape = (grid_dim, grid_dim)
        return EncoderOutput(tokens=tokens, grid_shape=grid_shape)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.encoder_std.to(x.device) + self.encoder_mean.to(x.device)


class NeuroVFMEncoderAdapter(nn.Module):
    """Adapter for NeuroVFM encoders loaded from HuggingFace Hub."""

    def __init__(
        self,
        encoder_name_or_path: str,
        device: str | None = None,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        try:
            from neurovfm.pipelines import load_encoder
        except ImportError as err:
            raise ImportError(
                "neurovfm is required to use the NeuroVFM encoder adapter. "
                "Install it with: pip install neurovfm"
            ) from err

        encoder_pipeline, preprocessor = load_encoder(
            encoder_name_or_path, device=device
        )
        self.encoder_pipeline = encoder_pipeline
        self.preprocessor = preprocessor
        self.use_amp = use_amp

        model = encoder_pipeline.model
        self.hidden_size = getattr(model, "embed_dim", None) or getattr(
            model, "hidden_size", None
        )
        if self.hidden_size is None:
            raise ValueError("Unable to infer hidden size from NeuroVFM encoder.")

        self.patch_size = getattr(preprocessor, "patch_size", None)
        if self.patch_size is None:
            raise ValueError("Unable to infer patch size from NeuroVFM preprocessor.")

    @torch.no_grad()
    def encode(self, batch: dict[str, torch.Tensor]) -> EncoderOutput:
        tokens = self.encoder_pipeline.embed(batch, use_amp=self.use_amp)

        if "series_cu_seqlens" in batch:
            cu = batch["series_cu_seqlens"].tolist()
            coords = batch["coords"].to(tokens.device)
        else:
            cu = [0, tokens.shape[0]]
            coords = batch["coords"].to(tokens.device)

        grid_shapes: list[tuple[int, int, int]] = []
        series_tokens: list[torch.Tensor] = []

        for start, end in zip(
            cu[:-1],
            cu[1:],
        ):
            series_tokens_flat = tokens[start:end]
            series_coords = coords[start:end]

            grid_shape = tuple(
                int(series_coords[:, dim].max().item()) + 1 for dim in range(3)
            )
            num_patches = grid_shape[0] * grid_shape[1] * grid_shape[2]
            ordered_tokens = series_tokens_flat.new_zeros(
                (num_patches, tokens.shape[-1])
            )

            flat_indices = (
                series_coords[:, 0] * (grid_shape[1] * grid_shape[2])
                + series_coords[:, 1] * grid_shape[2]
                + series_coords[:, 2]
            )
            ordered_tokens[flat_indices] = series_tokens_flat

            grid_shapes.append(grid_shape)
            series_tokens.append(ordered_tokens)

        if len(set(grid_shapes)) != 1:
            raise ValueError(
                "NeuroVFM encoder returned mixed grid shapes in the same batch. "
                "Please batch series with consistent spatial sizes."
            )

        tokens = torch.stack(series_tokens, dim=0)
        return EncoderOutput(tokens=tokens, grid_shape=grid_shapes[0])


class PatchDecoder(nn.Module):
    """Patch-wise MLP decoder for RAE latents."""

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        patch_size: tuple[int, ...],
        dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        output_dim = out_channels * prod(patch_size)

        layers: list[nn.Module] = []
        current_dim = latent_dim
        if num_layers <= 1:
            layers.append(nn.Linear(current_dim, output_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, tokens: torch.Tensor, grid_shape: tuple[int, ...]
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError("Decoder expects tokens of shape (B, N, C).")

        batch_size, num_tokens, _ = tokens.shape
        patch_dim = prod(self.patch_size)
        projected = self.mlp(tokens).view(
            batch_size, num_tokens, self.out_channels, patch_dim
        )

        if self.dim == 2:
            height, width = grid_shape
            p_h, p_w = self.patch_size
            projected = projected.view(
                batch_size,
                height,
                width,
                self.out_channels,
                p_h,
                p_w,
            )
            projected = projected.permute(0, 3, 1, 4, 2, 5)
            return projected.reshape(
                batch_size, self.out_channels, height * p_h, width * p_w
            )

        depth, height, width = grid_shape
        p_d, p_h, p_w = self.patch_size
        projected = projected.view(
            batch_size,
            depth,
            height,
            width,
            self.out_channels,
            p_d,
            p_h,
            p_w,
        )
        projected = projected.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return projected.reshape(
            batch_size,
            self.out_channels,
            depth * p_d,
            height * p_h,
            width * p_w,
        )


class RAETokenizer(BaseTokenizer):
    """Representation Autoencoder (RAE) tokenizer with a frozen encoder."""

    def __init__(
        self,
        dim: int,
        encoder_type: EncoderType,
        encoder_name_or_path: str,
        out_channels: int = 1,
        latent_dim: int | None = None,
        patch_size: int | tuple[int, ...] | None = None,
        encoder_image_size: int | None = None,
        encoder_drop_cls_token: bool = True,
        encoder_kwargs: dict[str, Any] | None = None,
        decoder_hidden_dim: int = 1024,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.0,
        noise_tau: float = 0.0,
        latent_stats_path: str | None = None,
        latent_eps: float = 1e-5,
        name: str = "RAETokenizer",
    ) -> None:
        super().__init__(dim=dim, name=name)

        self.encoder_type = encoder_type
        self.encoder_name_or_path = encoder_name_or_path
        self.out_channels = out_channels
        self.noise_tau = noise_tau
        self.latent_eps = latent_eps

        if encoder_type in {"vit", "medsiglip"}:
            if dim != 2:
                raise ValueError(
                    "ViT/SigLIP encoder adapters only support dim=2 inputs."
                )
            self.encoder = ViTEncoderAdapter(
                encoder_name_or_path=encoder_name_or_path,
                image_size=encoder_image_size,
                drop_cls_token=encoder_drop_cls_token,
                encoder_kwargs=encoder_kwargs,
            )
            inferred_patch = self.encoder.patch_size
            grid_shape = self.encoder.image_size // inferred_patch
            self.latent_grid_shape = (grid_shape, grid_shape)
        elif encoder_type == "neurovfm":
            if dim != 3:
                raise ValueError("NeuroVFM encoder adapter only supports dim=3 inputs.")
            self.encoder = NeuroVFMEncoderAdapter(
                encoder_name_or_path=encoder_name_or_path,
            )
            self.latent_grid_shape = None
            inferred_patch = self.encoder.patch_size
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if patch_size is None:
            patch_size = inferred_patch

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * dim
        self.patch_size = patch_size

        encoder_hidden = self.encoder.hidden_size
        self.latent_dim = latent_dim or encoder_hidden
        if self.latent_dim != encoder_hidden:
            self.latent_projection = nn.Linear(encoder_hidden, self.latent_dim)
        else:
            self.latent_projection = nn.Identity()

        self.decoder = PatchDecoder(
            latent_dim=self.latent_dim,
            out_channels=out_channels,
            patch_size=self.patch_size,
            dim=dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
        )

        self.latent_mean: torch.Tensor | None = None
        self.latent_var: torch.Tensor | None = None
        if latent_stats_path is not None:
            # Latent stats are a plain dict of tensors (mean/var), so
            # weights_only=True is both safe and sufficient here. It blocks
            # arbitrary pickle code execution when loading untrusted files.
            stats = torch.load(latent_stats_path, map_location="cpu", weights_only=True)
            self.latent_mean = stats.get("mean")
            self.latent_var = stats.get("var")

        self.config = {
            "dim": dim,
            "encoder_type": encoder_type,
            "encoder_name_or_path": encoder_name_or_path,
            "out_channels": out_channels,
            "latent_dim": latent_dim,
            "patch_size": patch_size,
            "encoder_image_size": encoder_image_size,
            "encoder_drop_cls_token": encoder_drop_cls_token,
            "encoder_kwargs": encoder_kwargs or {},
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "decoder_dropout": decoder_dropout,
            "noise_tau": noise_tau,
            "latent_stats_path": latent_stats_path,
            "latent_eps": latent_eps,
            "name": name,
        }

        self._last_grid_shape: tuple[int, ...] | None = None

    def _infer_grid_shape(self, tokens: torch.Tensor) -> tuple[int, ...]:
        if self.latent_grid_shape is not None:
            return self.latent_grid_shape

        num_tokens = tokens.shape[1]
        if self.dim == 2:
            grid = int(round(num_tokens**0.5))
            if grid * grid != num_tokens:
                raise ValueError(
                    "Unable to infer square grid shape from token count. "
                    "Please provide a fixed encoder image size."
                )
            return (grid, grid)

        grid = round(num_tokens ** (1 / 3))
        if grid**3 != num_tokens:
            raise ValueError(
                "Unable to infer cubic grid shape from token count. "
                "Please provide a fixed grid shape."
            )
        return (grid, grid, grid)

    def _apply_latent_normalization(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_var is None:
            return tokens
        return (tokens - self.latent_mean.to(tokens.device)) / torch.sqrt(
            self.latent_var.to(tokens.device) + self.latent_eps
        )

    def _remove_latent_normalization(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_var is None:
            return tokens
        return tokens * torch.sqrt(
            self.latent_var.to(tokens.device) + self.latent_eps
        ) + self.latent_mean.to(tokens.device)

    def encode(
        self, x: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        output = self.encoder.encode(x)
        tokens = self.latent_projection(output.tokens)

        if self.training and self.noise_tau > 0:
            noise_sigma = self.noise_tau * torch.rand(
                (tokens.size(0), 1, 1), device=tokens.device
            )
            tokens = tokens + noise_sigma * torch.randn_like(tokens)

        tokens = self._apply_latent_normalization(tokens)
        grid_shape = output.grid_shape or self._infer_grid_shape(tokens)
        self._last_grid_shape = grid_shape

        latent = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.latent_dim, *grid_shape
        )
        return latent, grid_shape

    def decode(
        self, z: torch.Tensor, grid_shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        if z.dim() not in (4, 5):
            raise ValueError("Latents must be 4D or 5D tensors.")

        if grid_shape is None:
            grid_shape = z.shape[2:]
        tokens = z.reshape(z.shape[0], self.latent_dim, -1).transpose(1, 2)
        tokens = self._remove_latent_normalization(tokens)

        recon = self.decoder(tokens, grid_shape=grid_shape)
        if isinstance(self.encoder, ViTEncoderAdapter):
            recon = self.encoder.denormalize(recon)
        return recon

    def forward(
        self, input: torch.Tensor | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | NetworkEval:
        """Full forward pass: encode -> decode.

        Follows the same return contract as the sibling tokenizers
        (``ContinuousTokenizer``/``DiscreteTokenizer``): a dict during
        training (for loss computation) and a ``NetworkEval`` namedtuple
        during evaluation.

        Args:
            input: Input tensor (or encoder batch dict for NeuroVFM).

        Returns:
            Training mode (dict):
                - ``reconstructions``: Decoded output.
                - ``latent``/``latents``: Latent grid tensor.

            Eval mode (NetworkEval):
                - ``reconstructions``: Decoded output.
                - ``posteriors``: Always ``None`` (RAE has no posterior).
                - ``latent``: Latent grid tensor.
        """
        latent, grid_shape = self.encode(input)
        reconstructions = self.decode(latent, grid_shape=grid_shape)

        if self.training:
            return {
                "reconstructions": reconstructions,
                "latent": latent,
                "latents": latent,
            }

        return NetworkEval(
            reconstructions=reconstructions, posteriors=None, latent=latent
        )

    def tokenize(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(x)[0]


__all__ = ["RAETokenizer"]
