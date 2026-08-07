#!/usr/bin/env python3
"""Evaluation script for medical image tokenizers.

This script provides a command-line interface for evaluating trained tokenizers
on test datasets with comprehensive metrics.

Example usage:
    # Evaluate on random synthetic data
    python scripts/evaluate.py \\
        --model_path ./my-tokenizer \\
        --num_samples 100 \\
        --batch_size 8 \\
        --save_path ./eval_results.json

    # Evaluate on custom dataset
    python scripts/evaluate.py \\
        --model_path ./my-tokenizer \\
        --data_path ./test_data.npz \\
        --batch_size 8 \\
        --compute_lpips \\
        --save_samples
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medtokenizers.evaluation import SimpleDataset, TokenizerEvaluator
from medtokenizers.inference import load_tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate medical image tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained tokenizer model"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to test data (.npz or .pt file with 'images' key). "
        "If not provided, will generate random data.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100). "
        "Only used if data_path is not provided.",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="Spatial size for generated data (default: 128 128 128). "
        "Only used if data_path is not provided.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1). "
        "Only used if data_path is not provided.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (cuda/cpu). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--data_range",
        type=float,
        default=1.0,
        help="Maximum pixel value in images (default: 1.0)",
    )
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Compute LPIPS metric (only for 2D images, requires lpips package)",
    )

    # Output arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="./eval_results.json",
        help="Path to save evaluation results (default: ./eval_results.json)",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save reconstruction samples as .npz file",
    )

    return parser.parse_args()


def load_data(args, model_config):
    """Load or generate test data.

    Args:
        args: Command line arguments
        model_config: Model configuration dictionary

    Returns:
        DataLoader for test data
    """
    if args.data_path is not None:
        # Load data from file
        print(f"Loading data from {args.data_path}...")
        data_path = Path(args.data_path)

        if data_path.suffix == ".npz":
            data = np.load(data_path)
            if "images" in data:
                images = data["images"]
            else:
                # Try to get the first array in the file
                images = data[list(data.keys())[0]]
        elif data_path.suffix == ".pt":
            data = torch.load(data_path, weights_only=True)
            if isinstance(data, dict):
                images = data["images"]
            else:
                images = data
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        print(f"Loaded {len(images)} samples with shape {images.shape[1:]}")

    else:
        # Generate random synthetic data
        print(f"Generating {args.num_samples} random samples...")

        # Determine dimensionality from model config
        dim = model_config.get("dim", 3)
        in_channels = args.in_channels

        if dim == 2:
            if len(args.spatial_size) >= 2:
                spatial_size = args.spatial_size[:2]
            else:
                spatial_size = [128, 128]
            shape = (args.num_samples, in_channels, *spatial_size)
        else:  # 3D
            if len(args.spatial_size) >= 3:
                spatial_size = args.spatial_size[:3]
            else:
                spatial_size = [128, 128, 128]
            shape = (args.num_samples, in_channels, *spatial_size)

        # Generate random data normalized to [0, 1]
        images = np.random.randn(*shape).astype(np.float32)
        images = (images - images.min()) / (images.max() - images.min())

        print(f"Generated data with shape {shape}")

    # Create dataset and dataloader
    dataset = SimpleDataset(images)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device in [None, "cuda"] else False,
    )

    return data_loader


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("MEDICAL IMAGE TOKENIZER EVALUATION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_tokenizer(args.model_path, device=args.device)
    print(f"✓ Loaded {model.config['name']}")
    print(f"  Type: {'Discrete' if hasattr(model, 'quantizer') else 'Continuous'}")
    print(f"  Dimension: {model.config['dim']}D")
    print(f"  Spatial compression: {model.config['spatial_compression']}x")

    # Load or generate data
    data_loader = load_data(args, model.config)

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = TokenizerEvaluator(
        model=model,
        device=args.device,
        data_range=args.data_range,
        compute_lpips=args.compute_lpips,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(
        data_loader=data_loader,
        num_samples=None if args.data_path else args.num_samples,
        save_reconstructions=args.save_samples,
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    print(f"\nSaving results to {args.save_path}...")
    TokenizerEvaluator.save_results(
        results, save_path=args.save_path, save_samples=args.save_samples
    )

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
