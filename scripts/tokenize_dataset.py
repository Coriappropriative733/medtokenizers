#!/usr/bin/env python3
"""Tokenize a MedMNIST dataset using all trained tokenizers.

Encodes every sample (train/val/test) through each model and saves the
latent representations for downstream generative modelling.

Output structure:
    <output_dir>/<dataset>_<tokenizer>/
        train.npz   -- codes/latents + labels
        val.npz
        test.npz
        metadata.json

For discrete tokenizers (VQ, FSQ, RESFSQ, LFQ):
    npz keys: "codes" (int16), "labels" (uint8)
For continuous tokenizers (AE, VAE):
    npz keys: "latents" (float16), "labels" (uint8)

Usage:
    # Tokenize chestmnist with all available checkpoints
    uv run python scripts/tokenize_dataset.py --dataset chestmnist

    # Single tokenizer
    uv run python scripts/tokenize_dataset.py --dataset chestmnist --tokenizer VQ

    # Custom paths
    uv run python scripts/tokenize_dataset.py --dataset chestmnist \
        --checkpoint_dir ./checkpoints/medmnist --output_dir ./tokenized
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import medmnist
import numpy as np
import torch
from safetensors.torch import load_file

from medtokenizers.inference import load_tokenizer
from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MedMNIST dataset groupings (dimensionality / channels).
DATASETS_3D = [
    "organmnist3d",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
]
DATASETS_2D_1CH = [
    "chestmnist",
    "octmnist",
    "pneumoniamnist",
    "breastmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]
DATASETS_2D_3CH = ["pathmnist", "dermamnist", "retinamnist", "bloodmnist"]

TOKENIZER_TYPES = ["VQ", "FSQ", "RESFSQ", "LFQ", "VAE", "AE"]


def get_dataset_info(dataset_name: str) -> dict:
    """Return dim, channels, and image size for a MedMNIST dataset."""
    if dataset_name in DATASETS_3D:
        return {"dim": 3, "channels": 1, "size": 64}
    elif dataset_name in DATASETS_2D_3CH:
        return {"dim": 2, "channels": 3, "size": 64}
    elif dataset_name in DATASETS_2D_1CH:
        return {"dim": 2, "channels": 1, "size": 64}
    raise ValueError(f"Unknown dataset: {dataset_name}")


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    """Normalize data to the [0, 1] range."""
    data = data.astype(np.float32)
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val > 1e-8:
        data = (data - min_val) / (max_val - min_val)
    return data


def _build_model_config(dataset_name: str, tokenizer_type: str) -> dict:
    """Reconstruct model constructor kwargs from dataset + tokenizer type.

    Mirrors the settings used by the train_medmnist{2,3}d examples, for loading
    Accelerate-format checkpoints that do not carry a config.json.
    """
    info = get_dataset_info(dataset_name)
    dim = info["dim"]
    in_channels = info["channels"]
    channels = 48 if dim == 3 else 64

    common = {
        "dim": dim,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "z_channels": 64,
        "channels": channels,
        "channels_mult": (1, 2, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.0,
        "resolution": 64,
        "spatial_compression": 8,
    }

    if tokenizer_type in ("VAE", "AE"):
        return dict(
            **common,
            latent_channels=4,
            formulation=tokenizer_type,
            name=f"{dataset_name}_{tokenizer_type}_tokenizer",
            cls="continuous",
        )

    quantizer_kwargs: dict = {}
    if tokenizer_type == "FSQ":
        quantizer_kwargs["levels"] = [8, 8, 8, 5, 5, 5]
    elif tokenizer_type == "RESFSQ":
        quantizer_kwargs["levels"] = [8, 8, 8]
        quantizer_kwargs["num_codebooks"] = 2
    elif tokenizer_type == "VQ":
        quantizer_kwargs["num_embeddings"] = 8192
        quantizer_kwargs["use_ema"] = True
        quantizer_kwargs["use_norm"] = True
    elif tokenizer_type == "LFQ":
        quantizer_kwargs["codebook_dim"] = 12
        quantizer_kwargs["codebook_size"] = 4096
        quantizer_kwargs["num_codebooks"] = 1

    return dict(
        **common,
        embedding_dim=16,
        quantizer=tokenizer_type,
        name=f"{dataset_name}_{tokenizer_type}_tokenizer",
        cls="discrete",
        **quantizer_kwargs,
    )


def _load_model(
    checkpoint_path: str, dataset_name: str, tokenizer_type: str, device: str
):
    """Load a tokenizer from a checkpoint directory.

    Supports HuggingFace format (config.json) and Accelerate format
    (model.safetensors, with config reconstructed from the dataset + tokenizer).
    """
    ckpt = Path(checkpoint_path)

    if (ckpt / "config.json").exists():
        return load_tokenizer(checkpoint_path, device=device)

    safetensors_path = ckpt / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"No config.json or model.safetensors found in {checkpoint_path}"
        )

    cfg = _build_model_config(dataset_name, tokenizer_type)
    cls_name = cfg.pop("cls")
    if cls_name == "continuous":
        model = ContinuousTokenizer(**cfg)
    else:
        model = DiscreteTokenizer(**cfg)

    state_dict = load_file(str(safetensors_path), device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_split(
    dataset_name: str, split: str, data_dir: str
) -> tuple[torch.Tensor, np.ndarray]:
    """Load a single split, return (images_tensor, labels_array)."""
    info = get_dataset_info(dataset_name)
    dim = info["dim"]
    size = info["size"]

    ds_data_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_data_dir, exist_ok=True)

    npz_path = os.path.join(ds_data_dir, f"{dataset_name}_{size}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"]
    else:
        mn_info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, mn_info["python_class"])
        dataset = DataClass(root=ds_data_dir, split=split, download=True, size=size)
        if hasattr(dataset, "imgs"):
            images = dataset.imgs
        elif hasattr(dataset, "data"):
            images = dataset.data
        else:
            raise ValueError(f"Cannot access images for {dataset_name}")
        labels = dataset.labels

    # Squeeze trailing singleton channel dim
    if dim == 3 and images.ndim == 5 and images.shape[-1] == 1:
        images = images.squeeze(-1)
    if dim == 2 and images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    images = normalize_to_01(images)

    # Convert to (N, C, ...) tensor
    if dim == 2:
        if images.ndim == 4:  # (N, H, W, C) -> (N, C, H, W)
            tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        else:  # (N, H, W) -> (N, 1, H, W)
            tensor = torch.from_numpy(images).unsqueeze(1)
    else:
        # 3D: (N, D, H, W) -> (N, 1, D, H, W)
        tensor = torch.from_numpy(images).unsqueeze(1)

    return tensor, labels


@torch.inference_mode()
def tokenize_split(
    model: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode all images through the model, return codes/latents as numpy."""
    is_discrete = isinstance(model, DiscreteTokenizer)
    all_outputs = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        out = model.tokenize(batch)
        all_outputs.append(out.cpu())

    result = torch.cat(all_outputs, dim=0)

    if is_discrete:
        arr = result.numpy()
        # Use int32 if values exceed int16 range (e.g. FSQ with 64K vocab)
        if arr.max() > np.iinfo(np.int16).max or arr.min() < np.iinfo(np.int16).min:
            return arr.astype(np.int32)
        return arr.astype(np.int16)
    else:
        return result.numpy().astype(np.float16)


def tokenize_model(
    model: torch.nn.Module,
    dataset_name: str,
    tokenizer_type: str,
    data_dir: str,
    output_dir: str,
    batch_size: int,
    device: str,
) -> dict:
    """Tokenize all splits for one model, save to disk, return metadata."""
    is_discrete = isinstance(model, DiscreteTokenizer)
    out_path = Path(output_dir) / f"{dataset_name}_{tokenizer_type}"
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset_name,
        "tokenizer": tokenizer_type,
        "type": "discrete" if is_discrete else "continuous",
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        logger.info(f"  Encoding {split} split...")
        images, labels = load_split(dataset_name, split, data_dir)
        result = tokenize_split(model, images, batch_size, device)

        key = "codes" if is_discrete else "latents"
        save_path = out_path / f"{split}.npz"
        np.savez_compressed(str(save_path), **{key: result, "labels": labels})

        metadata["splits"][split] = {
            "num_samples": len(images),
            "input_shape": list(images.shape),
            f"{key}_shape": list(result.shape),
            f"{key}_dtype": str(result.dtype),
            "file": str(save_path),
            "file_size_mb": round(save_path.stat().st_size / 1024 / 1024, 2),
        }
        logger.info(
            f"    {split}: {len(images)} samples -> {key} {result.shape} "
            f"({save_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )

    # Save metadata
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize a MedMNIST dataset with trained tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="MedMNIST dataset name (e.g. chestmnist)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenize with only this type (e.g. VQ). Default: all available.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/medmnist",
        help="Directory containing *_final checkpoint dirs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="MedMNIST data cache directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint_dir>/tokenized)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "tokenized")

    print("=" * 60)
    print("MedMNIST Dataset Tokenization")
    print("=" * 60)
    print(f"Dataset        : {args.dataset}")
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Device         : {device}")

    # Find available checkpoints for this dataset
    tokenizers = [args.tokenizer.upper()] if args.tokenizer else TOKENIZER_TYPES
    available = []
    for tok in tokenizers:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{tok}_final")
        if os.path.isdir(ckpt_path):
            available.append((tok, ckpt_path))

    if not available:
        print(f"\nNo checkpoints found for {args.dataset} in {args.checkpoint_dir}")
        return 1

    print(f"Tokenizers     : {', '.join(t for t, _ in available)}")
    print("=" * 60)

    all_metadata = []
    for tok, ckpt_path in available:
        print(f"\n[{tok}] Loading model from {ckpt_path}...")
        model = _load_model(ckpt_path, args.dataset, tok, device)

        meta = tokenize_model(
            model=model,
            dataset_name=args.dataset,
            tokenizer_type=tok,
            data_dir=args.data_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device,
        )
        all_metadata.append(meta)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for meta in all_metadata:
        tok = meta["tokenizer"]
        typ = meta["type"]
        key = "codes" if typ == "discrete" else "latents"
        train_info = meta["splits"]["train"]
        shape = train_info[f"{key}_shape"]
        total_mb = sum(s["file_size_mb"] for s in meta["splits"].values())
        print(f"  {tok:8s} ({typ:10s}): {key} shape {shape}, total {total_mb:.1f} MB")

    print(f"\nTokenized data saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
