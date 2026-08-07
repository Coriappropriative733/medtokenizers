# Training Scripts

This directory contains scripts for training medical image tokenizers on a
directory of volumetric scans. The loaders discover `.nii.gz` files recursively,
so any NIfTI dataset works; the data path is fully configurable via `--data_dir`.

## Quick Start

### Training from Scratch

To train a model from scratch using your local dataset directory:

```bash
python scripts/train.py \
    --name my-tokenizer-run \
    --data_dir /path/to/dataset \
    --type FSQ \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4
```

### Finetuning from MAISI Pretrained Weights

All finetuning runs through the unified `scripts/finetune.py`, which loads the
NVIDIA MAISI pretrained VAE weights and adapts them to the requested tokenizer
type (direct weight loading for VAE; encoder/decoder transfer for the discrete
quantizers):

```bash
# Finetune a VAE (native MAISI architecture)
python scripts/finetune.py \
    --type VAE \
    --data_dir /path/to/dataset \
    --maisi_weights weights/models/autoencoder_v2.pt \
    --epochs 20

# Finetune a discrete tokenizer (transfers encoder/decoder weights)
python scripts/finetune.py \
    --type FSQ \
    --data_dir /path/to/dataset \
    --maisi_weights weights/models/autoencoder_v2.pt \
    --epochs 20
```

`--type` accepts `AE`, `VAE`, `VQ`, `LFQ`, `FSQ`, and `RESFSQ`. The staged warmup
schedule and MAISI-aligned defaults follow the recipe described under
[Finetuning Strategy](#finetuning-strategy) below.

### Key Arguments

**Common to both train.py and finetune.py:**
- `--data_dir`: Path to your dataset directory (any directory of NIfTI volumes)
- `--name`: Name for this training run (used for WandB and checkpoint directories)
- `--type`: Tokenizer type (`AE`, `VAE`, `VQ`, `LFQ`, `FSQ`, `RESFSQ`)
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--epochs`: Number of training epochs

**Finetuning-specific:**
- `--maisi_weights`: Path to NVIDIA MAISI pretrained weights (autoencoder_v2.pt)
- `--warmup_epochs`: Epochs for reconstruction-only warmup (default: 5)
- `--pretrained_lr_mult`: LR multiplier for pretrained encoder/decoder (default: 0.1)

**Training from scratch only:**
- `--twod`: Train on 2D slices instead of 3D volumes

### Example: Quick Test Run

Test that everything works with a small run:

```bash
python scripts/train.py \
    --name test-run \
    --data_dir /path/to/dataset \
    --type FSQ \
    --batch_size 2 \
    --epochs 2 \
    --test_run \
    --twod
```

### Example: Full 3D Training

Train a full 3D tokenizer:

```bash
python scripts/train.py \
    --name fsq-3d \
    --data_dir /path/to/dataset \
    --type FSQ \
    --batch_size 2 \
    --epochs 200 \
    --lr 1e-4 \
    --resolution 192 \
    --spatial_compression 8
```

## Finetuning Strategy

The finetuning script uses a staged training approach (20 epochs total):

1. **Warmup (Epochs 1-5)**: Reconstruction loss only (L1)
   - Allows the randomly initialized quantizer to stabilize
   - Uses full learning rate for new components, 0.1x for pretrained encoder/decoder

2. **Full Training (Epochs 6-20)**: Reconstruction + Perceptual losses
   - Gradually introduces VGG perceptual and Gram style losses
   - Laplacian pyramid loss for multi-scale reconstruction

### Architecture Alignment

The tokenizers are aligned with NVIDIA MAISI architecture:
- `spatial_compression=4` (4x downsampling per axis)
- `z_channels=4` (latent channels)
- No attention layers (MAISI uses pure convolutional encoder/decoder)
- MAISI-style decoder with `[2, 2, 0]` residual blocks per stage

This allows direct weight transfer from MAISI pretrained checkpoints.

### Adversarial Training (Future)

The codebase includes `VQGANLoss` and `VAEGANLoss` classes with full discriminator support:
- Multi-scale discriminator (`MultiScale`)
- Hinge adversarial loss
- R1 gradient penalty

These are not currently wired into the finetuning scripts. The current approach uses `Combined` loss (reconstruction + perceptual) which provides good results without the complexity of GAN training. For adversarial training, you would need to:

1. Replace `Combined` loss with `VQGANLoss` or `VAEGANLoss`
2. Add a separate discriminator optimizer
3. Modify the training loop to alternate generator/discriminator steps

See `medtokenizers.training.losses.compound` for the GAN loss implementations.

## Converting MAISI Weights

To convert NVIDIA MAISI weights to a HuggingFace-compatible format:

```bash
# Convert and save locally
python scripts/convert_maisi_to_hf.py \
    --maisi_weights weights/models/autoencoder_v2.pt \
    --output_dir ./maisi-vae

# Convert and push to HuggingFace Hub
python scripts/convert_maisi_to_hf.py \
    --maisi_weights weights/models/autoencoder_v2.pt \
    --output_dir ./maisi-vae \
    --push_to_hub \
    --repo_id username/maisi-vae-3d
```

After conversion, load the model:

```python
from medtokenizers import ContinuousTokenizer

model = ContinuousTokenizer.from_pretrained("./maisi-vae")
# Or from HuggingFace Hub:
model = ContinuousTokenizer.from_pretrained("username/maisi-vae-3d")
```

## Data Loading

The scripts:
- Find all `.nii.gz` files recursively under `--data_dir`
- Split into train/val (90/10) with reproducible shuffle
- Apply appropriate transforms (normalization, resizing, augmentation)
- Fail fast on corrupted or missing files
