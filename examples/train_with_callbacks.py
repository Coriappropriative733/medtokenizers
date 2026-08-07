"""Example: Training a medical tokenizer with the new extensible trainer architecture.

This demonstrates:
- Using the Trainer class with callbacks
- Custom loss functions
- Early stopping and checkpointing
- Learning rate scheduling
- Clean, extensible architecture
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from medtokenizers.networks import DiscreteTokenizer
from medtokenizers.training import (
    Checkpoint,
    Combined,
    EarlyStopping,
    Logger,
    LRScheduler,
    Trainer,
)


def create_dummy_dataset(num_samples=1000, img_size=64):
    """Create a dummy medical image dataset for demonstration."""
    # Simulate 2D medical images
    images = torch.randn(num_samples, 1, img_size, img_size)
    return TensorDataset(images)


def main():
    print("=" * 80)
    print("Medical Tokenizer Training - Clean Architecture Demo")
    print("=" * 80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create model
    print("\n[1/5] Creating DiscreteTokenizer...")
    model = DiscreteTokenizer(
        dim=2,
        z_channels=4,
        embedding_dim=256,
        in_channels=1,
        out_channels=1,
        channels=64,
        channels_mult=[1, 2],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=64,
        spatial_compression=4,
        quantizer="RESFSQ",
        levels=[8, 8, 8],
        num_codebooks=2,
    )
    print(f"   Model parameters: {model.num_parameters():,}")

    # Create datasets
    print("\n[2/5] Creating datasets...")
    train_dataset = create_dummy_dataset(num_samples=800, img_size=64)
    val_dataset = create_dummy_dataset(num_samples=200, img_size=64)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create optimizer
    print("\n[3/5] Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Create loss function
    loss_fn = Combined(
        reconstruction_weight=1.0,
        perceptual_weight=0.1,
        quantization_weight=1.0,
        reconstruction_type="l1",
    )

    # Create callbacks
    print("\n[4/5] Creating callbacks...")
    callbacks = [
        Logger(print_every=10),
        EarlyStopping(patience=10, min_delta=0.001, monitor="val_loss"),
        Checkpoint(
            filepath="checkpoints/model_best.pt",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=True,
        ),
        LRScheduler(scheduler),
    ]

    # Create trainer
    print("\n[5/5] Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        mixed_precision=torch.cuda.is_available(),  # Use AMP if CUDA available
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    trainer.fit(
        train_loader=train_loader,
        epochs=50,
        val_loader=val_loader,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Save final model
    final_checkpoint_path = "checkpoints/model_final.pt"
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")

    # Demonstrate inference
    print("\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        # Get a batch
        sample_batch = next(iter(val_loader))[0].to(device)
        print(f"\nInput shape: {sample_batch.shape}")

        # Encode
        indices, codes, loss = model.encode(sample_batch)
        print(f"Encoded indices shape: {indices.shape}")
        print(f"Encoded codes shape: {codes.shape}")

        # Decode
        reconstruction = model.decode(codes)
        print(f"Reconstruction shape: {reconstruction.shape}")

        # Decode from indices
        reconstruction_from_indices = model.detokenize(indices)
        print(f"Reconstruction from indices shape: {reconstruction_from_indices.shape}")

        # Compute reconstruction error
        recon_error = torch.mean((sample_batch - reconstruction) ** 2).item()
        print(f"\nMean squared reconstruction error: {recon_error:.6f}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
