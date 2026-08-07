"""Base classes for the medtokenizers architecture.

This module defines the abstract base classes that establish the contract
for all tokenizers and quantizers in the library. It provides:

1. **BaseQuantizer**: Abstract interface for discrete quantization methods
2. **BaseTokenizer**: Abstract interface for tokenizers with HuggingFace Hub integration

Design Philosophy
-----------------
The base classes enforce a consistent API across all implementations while
remaining flexible enough to support diverse architectures (VAE, VQ-VAE, FSQ,
RESFSQ, etc.). Key design decisions:

- **Unified encode/decode interface**: All tokenizers expose the same methods
- **HuggingFace Hub integration**: Easy model sharing and loading
- **Batch processing utilities**: Built-in support for large volume processing
- **Sliding window inference**: For volumes larger than GPU memory

Type Safety
-----------
Uses jaxtyping for tensor shape annotations and beartype for runtime
validation. This catches shape mismatches early and documents expected
tensor shapes clearly.
"""

from __future__ import annotations

import itertools
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int

from .utils import jaxtyped_compile_safe

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class BaseQuantizer(ABC, nn.Module):
    """Abstract base class for all quantization methods.

    Quantizers map continuous encoder outputs to discrete codes from a
    finite vocabulary. This is the key component that enables discrete
    latent representations in VQ-VAE and related architectures.

    The Quantization Contract
    -------------------------
    All quantizers must implement:

    1. **forward(z)** -> (codes, loss, indices)
       - codes: Continuous codes (for decoder input)
       - loss: Auxiliary quantization loss (commitment, entropy, etc.)
       - indices: Integer codebook indices

    2. **indices_to_codes(indices)** -> codes
       - Decode indices back to continuous codes

    3. **get_codebook_size()** -> int
       - Return the total number of codes

    Implementation Notes
    --------------------
    - The `dtype` attribute controls output precision
    - Subclasses should call super().__init__() first
    - Use @jaxtyped decorator for shape validation

    Example Subclass:
        >>> class MyQuantizer(BaseQuantizer):
        ...     def forward(self, z):
        ...         # Quantization logic
        ...         return codes, loss, indices
        ...
        ...     def indices_to_codes(self, indices):
        ...         # Decoding logic
        ...         return codes
        ...
        ...     def get_codebook_size(self):
        ...         return self.num_codes
    """

    def __init__(self) -> None:
        super().__init__()
        self.dtype: torch.dtype = torch.float32

    @abstractmethod
    @jaxtyped_compile_safe(beartype)
    def forward(
        self, z: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Float[torch.Tensor, "batch channels *spatial"],
        Float[torch.Tensor, "batch *loss_dims"],
        Int[torch.Tensor, "batch *spatial"],
    ]:
        """Quantize continuous input to discrete codes.

        This is the core quantization operation. It maps continuous encoder
        outputs to the nearest codes in the quantizer's vocabulary.

        Args:
            z: Continuous input tensor from encoder.
               Shape: ``(batch, channels, *spatial)`` where spatial is (H, W) or (H, W, D)

        Returns:
            codes: Continuous codes to pass to decoder.
                   Shape: same as input z
            loss: Auxiliary quantization loss (commitment, codebook, entropy, etc.)
                  Shape varies by quantizer
            indices: Integer indices into the codebook.
                    Shape varies by quantizer (typically ``batch, *spatial``)

        Note:
            The loss should be added to the reconstruction loss during training.
            For implicit codebook methods like FSQ, this may be a dummy zero tensor.
        """
        pass

    @abstractmethod
    def indices_to_codes(
        self, indices: Int[torch.Tensor, "batch *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Convert discrete indices back to continuous codes.

        Used during inference to decode from stored indices without
        re-running the encoder.

        Args:
            indices: Quantized indices from forward() or stored tokens

        Returns:
            Continuous codes suitable for decoder input
        """
        pass

    def get_codebook_size(self) -> int:
        """Get the total number of codes in the vocabulary.

        For learned codebooks (VQ), this is the number of embedding vectors.
        For implicit codebooks (FSQ), this is the product of quantization levels.

        Returns:
            Total number of discrete codes
        """
        raise NotImplementedError("Subclass must implement get_codebook_size()")


class BaseTokenizer(ABC, nn.Module):
    """Abstract base class for all tokenizers with HuggingFace Hub integration.

    Tokenizers transform input images/volumes into latent representations
    (continuous or discrete) and back. This class provides:

    1. **Abstract encode/decode interface** - Subclasses implement specifics
    2. **HuggingFace Hub integration** - save_pretrained, from_pretrained, push_to_hub
    3. **Batch processing utilities** - For large datasets
    4. **Sliding window inference** - For volumes larger than GPU memory

    Architecture Overview
    ---------------------
    A tokenizer consists of:
    - **Encoder**: Compresses input to low-dimensional latent
    - **Quantizer** (discrete only): Maps continuous latent to discrete codes
    - **Decoder**: Reconstructs input from latent

    ```
    Input -> Encoder -> [Quantizer] -> Latent -> Decoder -> Reconstruction
    ```

    HuggingFace Hub Integration
    ---------------------------
    All tokenizers can be saved and loaded from the HuggingFace Hub:

    >>> model.save_pretrained("./my-tokenizer")
    >>> model = ContinuousTokenizer.from_pretrained("./my-tokenizer")
    >>> model.push_to_hub("username/my-tokenizer")

    Thread Safety
    -------------
    Models should be used in single-threaded contexts or with appropriate
    synchronization. The eval()/train() mode switching is NOT thread-safe.

    Args:
        dim: Spatial dimensionality (2 for images, 3 for volumes)
        name: Human-readable model name
    """

    config_name: str = "config.json"
    weights_name: str = "pytorch_model.bin"

    def __init__(self, dim: Optional[int] = None, name: str = "BaseTokenizer") -> None:
        super().__init__()
        if dim is not None:
            if dim not in [2, 3]:
                raise ValueError(f"dim must be 2 or 3, got {dim}")
            self.dim = dim
        self.name = name
        self.config: dict[str, Any] = {}

    # ==================== Abstract Methods ====================

    @abstractmethod
    @jaxtyped_compile_safe(beartype)
    def encode(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[torch.Tensor, ...]:
        """Encode input to latent representation.

        Args:
            x: Input image/volume tensor

        Returns:
            Tuple containing latent representation and additional outputs
            (varies by subclass - e.g., KL divergence for VAE)
        """
        pass

    @abstractmethod
    @jaxtyped_compile_safe(beartype)
    def decode(
        self, z: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode latent representation to output.

        Args:
            z: Latent tensor (continuous codes or quantized codes)

        Returns:
            Reconstructed output with same spatial shape as original input
        """
        pass

    @abstractmethod
    @jaxtyped_compile_safe(beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode -> [quantize] -> decode.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing at minimum:
            - 'reconstructions': Reconstructed output
            - Additional keys vary by subclass (posteriors, quant_loss, etc.)
        """
        pass

    # ==================== Context Managers ====================

    @contextmanager
    def inference_mode(self) -> Generator[BaseTokenizer, None, None]:
        """Context manager for optimized inference.

        Configures the model for maximum inference performance:
        - Sets model to eval mode
        - Disables gradient computation
        - Uses torch.inference_mode for additional optimizations

        The model is restored to its previous state upon exit.

        Example:
            >>> with model.inference_mode():
            ...     latents = model.tokenize(volume)
            ...     recon = model.detokenize(latents)

        Yields:
            self: The model instance for method chaining
        """
        was_training = self.training
        try:
            self.eval()
            with torch.inference_mode():
                yield self
        finally:
            if was_training:
                self.train()

    # ==================== Utility Methods ====================

    def num_parameters(self) -> int:
        """Get total number of learnable parameters.

        Returns:
            Sum of numel() for all parameters
        """
        return sum(p.numel() for p in self.parameters())

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (convenience method).

        Subclasses should override with appropriate return type.

        Args:
            x: Input tensor

        Returns:
            Latent representation (continuous or discrete indices)
        """
        raise NotImplementedError("Subclasses must implement tokenize()")

    def detokenize(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space (convenience method).

        Args:
            z: Latent tensor

        Returns:
            Reconstructed output
        """
        return self.decode(z)

    # ==================== Performance Optimization ====================

    def compile(
        self, mode: str = "reduce-overhead", fullgraph: bool = False, **kwargs
    ) -> "BaseTokenizer":
        """Compile model with torch.compile() for faster inference.

        Uses PyTorch 2.0+ compilation to optimize the model graph.
        The compiled model maintains the same interface but runs faster,
        especially for repeated inference calls.

        Args:
            mode: Compilation mode. Options:
                - "reduce-overhead": Best for small batches (default)
                - "max-autotune": Best throughput, longer warmup
                - "default": Balanced compilation
            fullgraph: If True, require full graph compilation (stricter)
            **kwargs: Additional arguments passed to torch.compile()

        Returns:
            Compiled model (self, for method chaining)

        Example:
            >>> model = ContinuousTokenizer.from_pretrained("path/to/model")
            >>> model = model.compile(mode="reduce-overhead")
            >>> # First call triggers compilation (slower)
            >>> latents = model.tokenize(batch)
            >>> # Subsequent calls are faster
            >>> latents = model.tokenize(batch2)

        Note:
            - Compilation happens on first forward pass
            - Different input shapes may trigger recompilation
            - Use dynamic=False (default) for fixed input sizes
        """
        if not hasattr(torch, "compile"):
            import warnings

            warnings.warn(
                "torch.compile() requires PyTorch 2.0+. Returning uncompiled model.",
                stacklevel=2,
            )
            return self

        return torch.compile(
            self, mode=mode, dynamic=False, fullgraph=fullgraph, **kwargs
        )

    # ==================== HuggingFace Hub Integration ====================

    def save_pretrained(
        self, save_directory: str | Path, push_to_hub: bool = False, **kwargs
    ) -> None:
        """Save model weights and configuration to directory.

        Creates a directory structure compatible with from_pretrained():
        ```
        save_directory/
        ├── config.json      # Model configuration
        └── pytorch_model.bin # Model weights
        ```

        Args:
            save_directory: Path to save model
            push_to_hub: If True, also upload to HuggingFace Hub
            **kwargs: Additional arguments for push_to_hub

        Example:
            >>> model.save_pretrained("./my-tokenizer")
            >>> # Later...
            >>> model = ContinuousTokenizer.from_pretrained("./my-tokenizer")
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_directory / self.config_name
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Save model weights
        weights_path = save_directory / self.weights_name
        torch.save(self.state_dict(), weights_path)

        logger.info("Model saved to %s", save_directory)

        if push_to_hub:
            self.push_to_hub(str(save_directory), **kwargs)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, map_location: Optional[str] = None, **kwargs
    ) -> BaseTokenizer:
        """Load model from local directory or HuggingFace Hub.

        Automatically detects whether path is local or a Hub repository.
        For Hub repos, downloads config and weights to cache.

        Args:
            model_name_or_path: Local path or HuggingFace Hub repo ID
            map_location: Device to load weights to (default: auto-detect)
            **kwargs: Override config parameters

        Returns:
            Loaded model instance

        Example:
            >>> # From local path
            >>> model = ContinuousTokenizer.from_pretrained("./my-tokenizer")
            >>>
            >>> # From HuggingFace Hub
            >>> model = ContinuousTokenizer.from_pretrained("username/my-tokenizer")
            >>>
            >>> # Override config
            >>> model = ContinuousTokenizer.from_pretrained(
            ...     "./my-tokenizer",
            ...     dropout=0.1  # Override saved dropout value
            ... )
        """
        model_path = Path(model_name_or_path)

        # Check if local path exists
        if model_path.exists():
            config_path = model_path / cls.config_name
            weights_path = model_path / cls.weights_name
        else:
            # Try HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download

                config_path = Path(
                    hf_hub_download(
                        repo_id=model_name_or_path, filename=cls.config_name
                    )
                )
                weights_path = Path(
                    hf_hub_download(
                        repo_id=model_name_or_path, filename=cls.weights_name
                    )
                )
            except ImportError as err:
                raise ImportError(
                    "huggingface_hub is required to load from HuggingFace Hub. "
                    "Install it with: pip install huggingface_hub"
                ) from err
            except Exception as e:
                raise ValueError(
                    f"Could not find model at {model_name_or_path}. "
                    f"Ensure it exists locally or on HuggingFace Hub. Error: {e}"
                ) from e

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Merge with kwargs (kwargs take precedence)
        config.update(kwargs)

        # Create model instance
        model = cls(**config)

        # Determine device
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        # Load weights
        state_dict = torch.load(
            weights_path, map_location=map_location, weights_only=True
        )
        model.load_state_dict(state_dict)

        logger.info("Model loaded from %s", model_name_or_path)
        return model

    def push_to_hub(
        self,
        repo_id: str,
        save_directory: Optional[str] = None,
        commit_message: str = "Upload model",
        private: bool = False,
        **kwargs,
    ) -> None:
        """Upload model to HuggingFace Hub.

        Creates or updates a repository on the Hub with model weights
        and configuration.

        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            save_directory: Local directory to save before upload (auto-created if None)
            commit_message: Commit message for the upload
            private: Whether to create a private repository
            **kwargs: Additional arguments for HfApi.upload_folder

        Example:
            >>> model.push_to_hub("username/my-vae-tokenizer")
            >>> # Creates https://huggingface.co/username/my-vae-tokenizer
        """
        try:
            from huggingface_hub import HfApi
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required for Hub upload. "
                "Install it with: pip install huggingface_hub"
            ) from err

        # Save to temp directory if not provided
        save_path: Path
        if save_directory is None:
            save_path = Path(f"./{repo_id.split('/')[-1]}")
        else:
            save_path = Path(save_directory)

        # Save if not already saved
        if not (save_path / self.config_name).exists():
            self.save_pretrained(save_path)

        # Upload
        api = HfApi()
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(save_path),
            repo_id=repo_id,
            commit_message=commit_message,
            **kwargs,
        )

        logger.info("Model uploaded to https://huggingface.co/%s", repo_id)

    # ==================== Pretrained Weight Loading ====================

    def load_encoder_decoder_weights(
        self,
        pretrained_path: str | Path,
        strict: bool = False,
        verbose: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load encoder/decoder weights from a pretrained checkpoint.

        This method enables transfer learning by loading encoder and decoder
        weights from any compatible tokenizer (VAE, VQ-VAE, FSQ, etc.) while
        leaving quantizer-specific layers randomly initialized.

        Use Cases
        ---------
        1. **Initialize discrete tokenizer from VAE**: Load MAISI VAE encoder/decoder,
           then train only the quantizer layers (VQ, FSQ, etc.)
        2. **Fine-tune on new domain**: Start from pretrained weights, fine-tune all
        3. **Encoder-only transfer**: Use pretrained encoder for downstream tasks

        How It Works
        ------------
        The method identifies encoder/decoder weights by key prefixes and loads
        only those that match. Quantizer-specific layers (quant_conv, post_quant_conv,
        quantizer) may or may not be loaded depending on architecture compatibility.

        Weight Matching Strategy:
        - `encoder.*` keys: Always attempted
        - `decoder.*` keys: Always attempted
        - `quant_conv.*` keys: Loaded if shapes match
        - `post_quant_conv.*` keys: Loaded if shapes match
        - `quantizer.*` keys: Skipped (architecture-specific)

        Args:
            pretrained_path: Path to pretrained weights file (.pt, .bin) or
                           directory containing 'pytorch_model.bin'
            strict: If True, raise error on shape mismatches. If False (default),
                   skip mismatched weights and log warnings.
            verbose: If True, print summary of loaded/skipped weights

        Returns:
            Tuple of (loaded_keys, skipped_keys) for inspection

        Example:
            >>> # Initialize FSQ tokenizer from MAISI VAE weights
            >>> model = DiscreteTokenizer(
            ...     dim=3,
            ...     quantizer='FSQ',
            ...     levels=[8, 5, 5, 5],
            ...     # ... other args matching MAISI architecture
            ... )
            >>> loaded, skipped = model.load_encoder_decoder_weights(
            ...     'weights/maisi_converted.pt',
            ...     verbose=True
            ... )
            >>> print(f"Loaded {len(loaded)} keys, skipped {len(skipped)}")
            >>>
            >>> # Now train with frozen encoder if desired
            >>> for param in model.encoder.parameters():
            ...     param.requires_grad = False

        Note:
            For best results, ensure the pretrained model has the same:
            - `channels`, `channels_mult`, `num_res_blocks`
            - `spatial_compression`

            The following can differ:
            - `z_channels`, `latent_channels`, `embedding_dim`
            - Quantizer type and configuration
        """
        pretrained_path = Path(pretrained_path)

        if pretrained_path.is_dir():
            weights_file = pretrained_path / self.weights_name
            if not weights_file.exists():
                weights_file = pretrained_path / "pytorch_model.bin"
            if not weights_file.exists():
                raise FileNotFoundError(
                    f"No weights file found in {pretrained_path}. "
                    f"Expected '{self.weights_name}' or 'pytorch_model.bin'"
                )
        else:
            weights_file = pretrained_path

        pretrained_state = torch.load(
            weights_file, map_location="cpu", weights_only=True
        )

        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        elif "model" in pretrained_state:
            pretrained_state = pretrained_state["model"]
        elif "unet_state_dict" in pretrained_state:
            pretrained_state = pretrained_state["unet_state_dict"]

        model_state = self.state_dict()

        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        shape_mismatches: list[tuple[str, tuple, tuple]] = []

        transfer_prefixes = ("encoder.", "decoder.")
        optional_prefixes = ("quant_conv.", "post_quant_conv.")
        skip_prefixes = ("quantizer.", "distribution.")

        for key, pretrained_tensor in pretrained_state.items():
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                skipped_keys.append(key)
                continue

            if key not in model_state:
                skipped_keys.append(key)
                if verbose:
                    logger.debug(f"Skipping {key}: not in model")
                continue

            model_tensor = model_state[key]

            if pretrained_tensor.shape != model_tensor.shape:
                shape_mismatches.append(
                    (key, tuple(pretrained_tensor.shape), tuple(model_tensor.shape))
                )

                if any(key.startswith(prefix) for prefix in optional_prefixes):
                    skipped_keys.append(key)
                    if verbose:
                        logger.info(
                            f"Skipping {key}: shape mismatch "
                            f"{pretrained_tensor.shape} vs {model_tensor.shape}"
                        )
                    continue

                if any(key.startswith(prefix) for prefix in transfer_prefixes):
                    if strict:
                        raise ValueError(
                            f"Shape mismatch for {key}: "
                            f"pretrained {pretrained_tensor.shape} vs "
                            f"model {model_tensor.shape}. "
                            f"Ensure encoder/decoder architecture matches."
                        )
                    skipped_keys.append(key)
                    if verbose:
                        logger.warning(
                            f"Skipping {key}: shape mismatch "
                            f"{pretrained_tensor.shape} vs {model_tensor.shape}"
                        )
                    continue

            model_state[key] = pretrained_tensor
            loaded_keys.append(key)

        self.load_state_dict(model_state, strict=False)

        if verbose:
            encoder_loaded = sum(1 for k in loaded_keys if k.startswith("encoder."))
            decoder_loaded = sum(1 for k in loaded_keys if k.startswith("decoder."))
            other_loaded = len(loaded_keys) - encoder_loaded - decoder_loaded

            logger.info(f"Loaded pretrained weights from {weights_file}")
            logger.info(
                f"  Encoder: {encoder_loaded} keys, "
                f"Decoder: {decoder_loaded} keys, "
                f"Other: {other_loaded} keys"
            )
            logger.info(f"  Skipped: {len(skipped_keys)} keys")

            if shape_mismatches and len(shape_mismatches) <= 5:
                logger.info("  Shape mismatches:")
                for key, pre_shape, model_shape in shape_mismatches:
                    logger.info(f"    {key}: {pre_shape} vs {model_shape}")

        return loaded_keys, skipped_keys

    @classmethod
    def from_pretrained_encoder_decoder(
        cls,
        pretrained_path: str | Path,
        strict: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> "BaseTokenizer":
        """Create model and load encoder/decoder weights from pretrained checkpoint.

        Factory method that creates a new model instance and loads encoder/decoder
        weights in one step. Useful for initializing new architectures from
        pretrained backbones.

        Args:
            pretrained_path: Path to pretrained weights
            strict: If True, raise on shape mismatches
            verbose: If True, print loading summary
            **kwargs: Model configuration (passed to __init__)

        Returns:
            New model instance with loaded encoder/decoder weights

        Example:
            >>> # Create FSQ model initialized from VAE weights
            >>> model = DiscreteTokenizer.from_pretrained_encoder_decoder(
            ...     'weights/maisi_converted.pt',
            ...     dim=3,
            ...     quantizer='FSQ',
            ...     levels=[8, 5, 5, 5],
            ...     z_channels=256,
            ...     channels=64,
            ...     channels_mult=(1, 2, 4),
            ... )
        """
        model = cls(**kwargs)
        model.load_encoder_decoder_weights(
            pretrained_path, strict=strict, verbose=verbose
        )
        return model

    # ==================== Batch Processing ====================

    @torch.inference_mode()
    def encode_batch(
        self, x: torch.Tensor, batch_size: int = 8, show_progress: bool = False
    ) -> torch.Tensor:
        """Encode large batch with automatic mini-batching.

        Processes input in chunks to avoid OOM for large datasets.

        Args:
            x: Full input tensor of shape ``(N, C, *spatial)``
            batch_size: Mini-batch size for processing
            show_progress: Whether to show tqdm progress bar

        Returns:
            Concatenated latents for all inputs

        Example:
            >>> dataset = torch.randn(1000, 1, 64, 64, 64)
            >>> latents = model.encode_batch(dataset, batch_size=4)
        """
        self.eval()

        num_samples = x.shape[0]
        all_latents: list[torch.Tensor] = []

        indices: Iterable[int]
        if show_progress:
            try:
                from tqdm import tqdm

                indices = tqdm(range(0, num_samples, batch_size), desc="Encoding")
            except ImportError:
                indices = range(0, num_samples, batch_size)
        else:
            indices = range(0, num_samples, batch_size)

        for i in indices:
            batch = x[i : i + batch_size]
            latents = self.tokenize(batch)
            all_latents.append(latents)

        return torch.cat(all_latents, dim=0)

    @torch.inference_mode()
    def decode_batch(
        self, z: torch.Tensor, batch_size: int = 8, show_progress: bool = False
    ) -> torch.Tensor:
        """Decode large batch with automatic mini-batching.

        Args:
            z: Full latent tensor
            batch_size: Mini-batch size for processing
            show_progress: Whether to show tqdm progress bar

        Returns:
            Concatenated reconstructions for all inputs
        """
        self.eval()

        num_samples = z.shape[0]
        all_outputs: list[torch.Tensor] = []

        indices: Iterable[int]
        if show_progress:
            try:
                from tqdm import tqdm

                indices = tqdm(range(0, num_samples, batch_size), desc="Decoding")
            except ImportError:
                indices = range(0, num_samples, batch_size)
        else:
            indices = range(0, num_samples, batch_size)

        for i in indices:
            batch = z[i : i + batch_size]
            outputs = self.detokenize(batch)
            all_outputs.append(outputs)

        return torch.cat(all_outputs, dim=0)

    # ==================== Sliding Window Inference ====================

    @torch.inference_mode()
    def reconstruct(
        self,
        x: torch.Tensor,
        roi_size: tuple[int, ...] | Optional[int] = None,
        overlap: float = 0.5,
        sw_batch_size: int = 1,
    ) -> torch.Tensor:
        """Full encode-decode reconstruction with optional sliding window.

        For large 3D volumes that exceed GPU memory, this method implements
        sliding window inference with Gaussian importance weighting to
        seamlessly blend overlapping patches.

        The Algorithm
        -------------
        1. Pad input to multiple of stride + window size
        2. Extract overlapping windows with specified stride
        3. Process windows in batches through tokenize -> detokenize
        4. Weight each window's contribution by Gaussian importance
        5. Normalize by accumulated importance and crop to original size

        Gaussian Weighting
        ------------------
        Uses a Gaussian importance map (2D or 3D based on input) that gives
        higher weight to the center of each window, reducing boundary artifacts
        when blending.

        Args:
            x: Input tensor of shape (B, C, H, W, D) for 3D or (B, C, H, W) for 2D
            roi_size: Size of sliding window. If None, processes entire volume.
                     Can be int (isotropic) or tuple (anisotropic).
            overlap: Fraction of overlap between windows (0.0 to 0.9).
                    Higher overlap = smoother blending but more compute.
            sw_batch_size: Number of windows to process in parallel per batch.
                          Higher values use more GPU memory but are faster.
                          Default is 1 (sequential processing).

        Returns:
            Reconstructed tensor with same shape as input

        Example:
            >>> volume = torch.randn(1, 1, 256, 256, 256)  # 256³ volume
            >>> # Process in 128³ windows with 50% overlap, 4 windows at a time
            >>> recon = model.reconstruct(volume, roi_size=128, overlap=0.5, sw_batch_size=4)

        Note:
            - For volumes that fit in memory, omit roi_size for faster processing
            - Overlap of 0.5 is a good default; higher values reduce artifacts
              but increase computation proportionally
            - sw_batch_size > 1 can significantly speed up inference on GPUs
              with sufficient memory
        """
        self.eval()

        # Simple case: process entire volume at once
        if roi_size is None:
            return self.detokenize(self.tokenize(x))

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"reconstruct() expected torch.Tensor, got {type(x).__name__}."
            )
        if not x.is_floating_point():
            raise TypeError(
                f"reconstruct() expected floating point tensor, got {x.dtype}."
            )

        dim = getattr(self, "dim", None)
        if dim is None:
            if x.ndim == 4:
                dim = 2
            elif x.ndim == 5:
                dim = 3
            else:
                raise ValueError(
                    "reconstruct() expected input with 4D (B, C, H, W) or "
                    f"5D (B, C, H, W, D) shape, got {x.ndim}D."
                )
        else:
            expected_ndim = dim + 2
            if x.ndim != expected_ndim:
                raise ValueError(
                    f"reconstruct() expected {expected_ndim}D input for dim={dim}, "
                    f"got {x.ndim}D with shape {tuple(x.shape)}."
                )

        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}.")

        if isinstance(roi_size, int):
            if roi_size <= 0:
                raise ValueError(f"roi_size must be > 0, got {roi_size}.")
            roi_size = (roi_size,) * dim
        else:
            if not isinstance(roi_size, (tuple, list)):
                raise TypeError(
                    "roi_size must be an int or a tuple/list of ints, "
                    f"got {type(roi_size).__name__}."
                )
            roi_size = tuple(roi_size)
            if len(roi_size) != dim:
                raise ValueError(
                    f"roi_size must have length {dim} for dim={dim}, "
                    f"got {len(roi_size)}."
                )
            if any((not isinstance(size, int) or size <= 0) for size in roi_size):
                raise ValueError(
                    f"roi_size must contain positive ints, got {roi_size}."
                )

        device = next(self.parameters()).device

        def gaussian_1d(length: int) -> torch.Tensor:
            coords = torch.linspace(-1, 1, length)
            g = torch.exp(-(coords**2) / 0.5)
            return g / g.max()

        def calc_padding(dim_size: int, stride: int, win: int) -> int:
            if dim_size < win:
                return win - dim_size
            n_windows = (dim_size - win + stride - 1) // stride
            padded_size = n_windows * stride + win
            return max(0, padded_size - dim_size)

        # Sliding window inference (dim-generic over 2D / 3D spatial axes).
        batch = x.shape[0]
        orig_spatial = tuple(x.shape[2:])
        win = tuple(roi_size)

        # Compute stride from overlap for each spatial axis.
        strides = tuple(int(w * (1 - overlap)) for w in win)
        if any(s < 1 for s in strides):
            raise ValueError(
                "overlap is too high for roi_size; computed stride must be >= 1."
            )

        # Calculate padding for each spatial axis.
        pads = tuple(
            calc_padding(size, stride, w)
            for size, stride, w in zip(orig_spatial, strides, win)
        )

        # Pad input. nn.functional.pad expects pad sizes in reverse spatial order,
        # i.e. (last_dim_low, last_dim_high, ..., first_dim_low, first_dim_high).
        if any(p > 0 for p in pads):
            pad_arg: list[int] = []
            for p in reversed(pads):
                pad_arg.extend((0, p))
            x = nn.functional.pad(x, pad_arg, mode="constant", value=0)

        spatial = tuple(x.shape[2:])

        # Gaussian importance map as the outer product of 1D windows. Each 1D
        # window is reshaped to occupy its own spatial axis so the product
        # broadcasts to the full (win[0], ..., win[dim - 1]) shape.
        importance = torch.ones(win)
        for axis in range(dim):
            shape = [1] * dim
            shape[axis] = win[axis]
            importance = importance * gaussian_1d(win[axis]).reshape(shape)
        importance = importance.to(device)
        # Broadcast over leading batch and channel dimensions.
        weight = importance[(None, None) + (slice(None),) * dim]

        # Enumerate window start positions. Iteration order matches nested loops
        # with the first spatial axis varying slowest.
        position_ranges = [
            range(0, size - w + 1, stride)
            for size, w, stride in zip(spatial, win, strides)
        ]
        positions: list[tuple[int, ...]] = list(itertools.product(*position_ranges))

        assert len(positions) > 0, (
            "Sliding window produced no output - check roi_size and stride"
        )

        def window_slices(start: tuple[int, ...]) -> tuple[slice, ...]:
            spatial_slices = tuple(slice(s, s + w) for s, w in zip(start, win))
            return (slice(None), slice(None)) + spatial_slices

        first_slices = (slice(None), slice(None)) + tuple(slice(0, w) for w in win)
        first_window = x[first_slices].to(device)
        first_out = self.detokenize(self.tokenize(first_window))
        out_channels = first_out.shape[1]

        output = torch.zeros(batch, out_channels, *spatial, device=device)
        importance_map = torch.zeros(batch, 1, *spatial, device=device)

        first_dst = window_slices(positions[0])
        output[first_dst] += first_out * weight
        importance_map[first_dst] += weight

        remaining_positions = positions[1:]
        for batch_start in range(0, len(remaining_positions), sw_batch_size):
            batch_positions = remaining_positions[
                batch_start : batch_start + sw_batch_size
            ]

            windows = [x[window_slices(start)] for start in batch_positions]
            batch_windows = torch.cat(windows, dim=0).to(device)

            batch_out = self.detokenize(self.tokenize(batch_windows))

            for i, start in enumerate(batch_positions):
                window_out = batch_out[i * batch : (i + 1) * batch]
                dst = window_slices(start)
                output[dst] += window_out * weight
                importance_map[dst] += weight

        output = output / importance_map

        # Crop to original size.
        if any(p > 0 for p in pads):
            crop = (slice(None), slice(None)) + tuple(
                slice(0, size) for size in orig_spatial
            )
            output = output[crop]

        return output

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate latent shape for given input shape.

        Args:
            input_shape: Input tensor shape ``(B, C, *spatial)``

        Returns:
            Expected latent tensor shape

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_latent_shape()")
