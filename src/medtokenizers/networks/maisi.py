"""MAISI tokenizer with NVIDIA model configuration."""

from pathlib import Path
from typing import Optional, Union

import torch

from .continuous import ContinuousTokenizer


class MAISITokenizer(ContinuousTokenizer):
    """MAISI VAE tokenizer matching NVIDIA NV-Generate-MR architecture.

    To use published NVIDIA MAISI weights, first convert them with
    ``scripts/convert_maisi_to_hf.py`` and then load the converted checkpoint via
    :meth:`from_pretrained`. Note that NVIDIA MAISI model weights are distributed
    under the NVIDIA Source Code License (NSCLv1), separate from this repository's
    Apache-2.0 code license.
    """

    DEFAULT_CONFIG = {
        "dim": 3,
        "in_channels": 1,
        "out_channels": 1,
        "z_channels": 4,
        "z_factor": 2,
        "latent_channels": 4,
        "channels": 64,
        "channels_mult": (1, 2, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (),
        "dropout": 0.0,
        "resolution": 256,
        "spatial_compression": 4,
        "formulation": "VAE",
        "name": "MAISITokenizer",
        # NVIDIA MAISI-specific architecture settings
        "use_encoder_mid": False,  # NVIDIA encoder has no mid blocks
        "decoder_blocks_per_stage": [2, 2, 0],  # NVIDIA decoder: up.0=2, up.1=2, up.2=0
    }

    def __init__(self, pretrained: Optional[str] = None, **kwargs):
        config = {**self.DEFAULT_CONFIG, **kwargs}
        super().__init__(**config)

        if pretrained is not None:
            self._load_pretrained(pretrained)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> "MAISITokenizer":
        """Load from local checkpoint."""
        return cls(pretrained=pretrained_model_name_or_path, **kwargs)

    def _load_pretrained(self, path: Union[str, Path]):
        """Load weights from local path."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. To use NVIDIA MAISI weights, first convert them "
                "with scripts/convert_maisi_to_hf.py, then point this loader at the "
                "converted checkpoint."
            )

        # Check if path is a directory (traditional HuggingFace format)
        if path.is_dir():
            checkpoint_path = path / "pytorch_model.pt"
            if not checkpoint_path.exists():
                for name in ["model.pt", "tokenizer.pt", "vae.pt"]:
                    if (path / name).exists():
                        checkpoint_path = path / name
                        break
                else:
                    raise FileNotFoundError(f"No checkpoint found in {path}")
        else:
            # Path is directly a checkpoint file
            checkpoint_path = path

        # Converted MAISI checkpoints are plain state dicts (tensors only), optionally
        # wrapped under a "state_dict" / "model" key, so weights_only=True is safe.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def get_training_config() -> dict:
        """Recommended training hyperparameters from MAISI paper."""
        return {
            "reconstruction_loss": "l1",
            "reconstruction_weight": 1.0,
            "perceptual_weight": 0.3,
            "kl_weight": 1e-7,
            "adversarial_weight": 0.1,
            "discriminator_start_iter": 5000,
            "target_std_min": 0.9,
            "target_std_max": 1.1,
            "kl_adaptation_rate": 0.1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "batch_size": 1,
            "patch_size_stage1": 64,
            "patch_size_stage2": 128,
            "epochs_stage1": 100,
            "epochs_stage2": 200,
            "normalization": "hounsfield",
            "random_crop": True,
            "spacing_type": "rand_zoom",
            "amp": True,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
        }


__all__ = ["MAISITokenizer"]
