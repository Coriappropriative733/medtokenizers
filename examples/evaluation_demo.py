"""Demonstration of evaluation functionality for medical image tokenizers.

This example shows how to:
1. Load a trained tokenizer
2. Create test data
3. Evaluate with comprehensive metrics
4. Save and visualize results
"""

import torch
from torch.utils.data import DataLoader

from medtokenizers import (
    TokenizerEvaluator,
    compute_psnr,
    compute_ssim,
    load_tokenizer,
)
from medtokenizers.evaluation import SimpleDataset


def example_basic_metrics():
    """Example: Compute basic metrics manually."""
    print("=" * 60)
    print("Example 1: Basic Metrics")
    print("=" * 60)

    # Create synthetic data
    original = torch.randn(4, 1, 128, 128, 128)
    reconstruction = original + 0.1 * torch.randn_like(original)

    # Compute metrics
    psnr = compute_psnr(reconstruction, original, data_range=1.0)
    ssim = compute_ssim(reconstruction, original, data_range=1.0)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print()


def example_evaluator_continuous():
    """Example: Evaluate a continuous tokenizer."""
    print("=" * 60)
    print("Example 2: Continuous Tokenizer Evaluation")
    print("=" * 60)

    # Create a small continuous tokenizer for demonstration
    from medtokenizers import ContinuousTokenizer

    model = ContinuousTokenizer(
        dim=3,
        in_channels=1,
        out_channels=1,
        z_channels=64,
        latent_channels=4,
        channels=32,
        channels_mult=(1, 2),
        num_res_blocks=1,
        spatial_compression=8,
        formulation="VAE",
    )
    model.eval()

    print(f"Model: {model.config['name']}")
    print(f"Spatial compression: {model.config['spatial_compression']}x")

    # Create test data
    test_images = torch.randn(32, 1, 128, 128, 128)
    dataset = SimpleDataset(test_images)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create evaluator
    evaluator = TokenizerEvaluator(
        model=model,
        device="cpu",  # Use 'cuda' if available
        data_range=1.0,
        compute_lpips=False,  # Set to True if you have lpips installed
    )

    # Run evaluation
    print("\nEvaluating...")
    results = evaluator.evaluate(loader, num_samples=32)

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(results, "./continuous_eval_results.json")
    print()


def example_evaluator_discrete():
    """Example: Evaluate a discrete tokenizer."""
    print("=" * 60)
    print("Example 3: Discrete Tokenizer Evaluation")
    print("=" * 60)

    # Create a small discrete tokenizer for demonstration
    from medtokenizers import DiscreteTokenizer

    model = DiscreteTokenizer(
        dim=3,
        in_channels=1,
        out_channels=1,
        z_channels=64,
        embedding_dim=6,
        channels=32,
        channels_mult=(1, 2),
        num_res_blocks=1,
        spatial_compression=8,
        quantizer="FSQ",
        levels=[8, 8, 8],
    )
    model.eval()

    print(f"Model: {model.config['name']}")
    print(f"Quantizer: {model.config['quantizer']}")
    print(f"Spatial compression: {model.config['spatial_compression']}x")

    # Create test data
    test_images = torch.randn(32, 1, 128, 128, 128)
    dataset = SimpleDataset(test_images)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create evaluator
    evaluator = TokenizerEvaluator(
        model=model,
        device="cpu",  # Use 'cuda' if available
        data_range=1.0,
    )

    # Run evaluation
    print("\nEvaluating...")
    results = evaluator.evaluate(loader, num_samples=32)

    # Print results (includes discrete metrics)
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(results, "./discrete_eval_results.json")
    print()


def example_with_saved_model():
    """Example: Evaluate a saved model."""
    print("=" * 60)
    print("Example 4: Evaluate Saved Model")
    print("=" * 60)

    # Save a model first
    from medtokenizers import ContinuousTokenizer

    model = ContinuousTokenizer(
        dim=3,
        in_channels=1,
        out_channels=1,
        z_channels=64,
        latent_channels=4,
        channels=32,
        spatial_compression=8,
        formulation="VAE",
    )
    model.save_pretrained("./demo_model")
    print("✓ Model saved to ./demo_model")

    # Load the model
    loaded_model = load_tokenizer("./demo_model", device="cpu")
    print("✓ Model loaded")

    # Create test data
    test_images = torch.randn(16, 1, 128, 128, 128)
    dataset = SimpleDataset(test_images)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Evaluate
    evaluator = TokenizerEvaluator(loaded_model, device="cpu")
    results = evaluator.evaluate(loader)

    # Print and save
    evaluator.print_results(results)
    evaluator.save_results(results, "./saved_model_eval.json")

    # Load results back
    loaded_results = TokenizerEvaluator.load_results("./saved_model_eval.json")
    print(f"\n✓ Results loaded. PSNR: {loaded_results['avg_metrics'].psnr:.2f} dB")
    print()


def example_compare_models():
    """Example: Compare multiple models."""
    print("=" * 60)
    print("Example 5: Compare Multiple Models")
    print("=" * 60)

    from medtokenizers import ContinuousTokenizer

    # Create models with different compression ratios
    models = {
        "4x": ContinuousTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=64,
            latent_channels=4,
            channels=32,
            spatial_compression=4,
            formulation="VAE",
        ),
        "8x": ContinuousTokenizer(
            dim=3,
            in_channels=1,
            out_channels=1,
            z_channels=64,
            latent_channels=4,
            channels=32,
            spatial_compression=8,
            formulation="VAE",
        ),
    }

    # Create test data
    test_images = torch.randn(16, 1, 128, 128, 128)
    dataset = SimpleDataset(test_images)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Evaluate each model
    results_dict = {}
    for name, model in models.items():
        model.eval()
        evaluator = TokenizerEvaluator(model, device="cpu")
        results = evaluator.evaluate(loader)
        results_dict[name] = results

    # Compare results
    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Model':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'Compression':<12}")
    print("-" * 60)
    for name, results in results_dict.items():
        metrics = results["avg_metrics"]
        print(
            f"{name:<10} {metrics.psnr:<12.2f} {metrics.ssim:<12.4f} "
            f"{metrics.compression_ratio:<12.1f}x"
        )
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_metrics()
    example_evaluator_continuous()
    example_evaluator_discrete()
    example_with_saved_model()
    example_compare_models()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
