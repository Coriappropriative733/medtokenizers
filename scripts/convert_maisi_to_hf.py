#!/usr/bin/env python3
"""Convert NVIDIA MAISI VAE weights to HuggingFace-compatible format.

This script converts the NVIDIA MAISI autoencoder weights to our ContinuousTokenizer
format, which can then be loaded via `from_pretrained()` or pushed to HuggingFace Hub.

Usage:
    # Convert and save locally
    python scripts/convert_maisi_to_hf.py \
        --maisi_weights weights/autoencoder_v2.pt \
        --output_dir ./maisi-vae

    # Convert and push to HuggingFace Hub
    python scripts/convert_maisi_to_hf.py \
        --maisi_weights weights/autoencoder_v2.pt \
        --output_dir ./maisi-vae \
        --push_to_hub \
        --repo_id username/maisi-vae-3d
"""

import argparse
from pathlib import Path

import torch

from medtokenizers.networks.continuous import ContinuousTokenizer
from medtokenizers.networks.nvidia_maisi import convert_nvidia_weights


def load_nvidia_weights(weights_path: str) -> dict:
    # weights_only=True: MAISI checkpoints are plain tensor state dicts.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "unet_state_dict" in state:
        state = state["unet_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    return state


def create_maisi_compatible_vae() -> ContinuousTokenizer:
    return ContinuousTokenizer(
        dim=3,
        in_channels=1,
        out_channels=1,
        z_channels=4,
        latent_channels=4,
        channels=64,
        channels_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resolution=256,
        spatial_compression=4,
        formulation="VAE",
        use_encoder_mid=False,
        use_output_nonlinearity=False,
        decoder_blocks_per_stage=[2, 2, 0],
        separate_quant_conv=True,
        name="MAISI-VAE",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA MAISI weights to HuggingFace format"
    )
    parser.add_argument(
        "--maisi_weights",
        type=str,
        required=True,
        help="Path to NVIDIA MAISI autoencoder weights (autoencoder_v2.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID (required if --push_to_hub)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace repo private",
    )
    args = parser.parse_args()

    if args.push_to_hub and not args.repo_id:
        parser.error("--repo_id is required when using --push_to_hub")

    print(f"Loading NVIDIA MAISI weights from {args.maisi_weights}")
    nvidia_state = load_nvidia_weights(args.maisi_weights)
    print(f"Loaded {len(nvidia_state)} keys from NVIDIA checkpoint")

    print("Converting weights to medtokenizers format...")
    converted_state = convert_nvidia_weights(nvidia_state)
    print(f"Converted to {len(converted_state)} keys")

    print("Creating MAISI-compatible VAE model...")
    model = create_maisi_compatible_vae()

    print("Loading converted weights...")
    missing, unexpected = model.load_state_dict(converted_state, strict=False)

    if missing:
        print(f"Warning: {len(missing)} missing keys:")
        for k in missing[:10]:
            print(f"  - {k}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            print(f"  - {k}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected) - 10} more")

    if not missing and not unexpected:
        print("All weights loaded successfully!")

    print("Running quick validation...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 32, 32, 32)
        output = model(test_input)
        recon = output.reconstructions
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {recon.shape}")
        print(f"Latent shape: {output.latent.shape}")

    output_path = Path(args.output_dir)
    print(f"\nSaving model to {output_path}")
    model.save_pretrained(output_path)

    print("\nModel config:")
    for key, value in model.config.items():
        print(f"  {key}: {value}")

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
        model.push_to_hub(args.repo_id, private=args.private)
        print(f"Model available at: https://huggingface.co/{args.repo_id}")

    print("\nDone!")
    print("\nTo load this model:")
    print("  from medtokenizers import ContinuousTokenizer")
    print(f'  model = ContinuousTokenizer.from_pretrained("{args.output_dir}")')
    if args.push_to_hub:
        print("\nOr from HuggingFace Hub:")
        print(f'  model = ContinuousTokenizer.from_pretrained("{args.repo_id}")')


if __name__ == "__main__":
    main()
