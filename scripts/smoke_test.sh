#!/bin/bash
set -e

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

if [ -z "$MAISI_WEIGHTS" ]; then
    echo "Error: MAISI_WEIGHTS environment variable must be set"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR environment variable must be set"
    exit 1
fi

cd "$(dirname "$0")/.." && uv run --extra cloud python scripts/finetune.py \
    --allow_tf32 \
    --name "smoke_test_vae" \
    --type VAE \
    --maisi_weights "$MAISI_WEIGHTS" \
    --separate_quant_conv \
    --epochs 5 \
    --warmup_epochs 1 \
    --stage2_warmup_epochs 1 \
    --steps_per_epoch 2 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --pretrained_lr_mult 0.1 \
    --val_interval 1 \
    --max_val_batches 5 \
    --num_workers 4 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --logdir ./checkpoints \
    --seed 42 \
    \
    --spatial_compression 4 \
    --resolution 256 \
    --channels 64 \
    --channels_mult 1 2 4 \
    --num_res_blocks 2 \
    --attn_resolutions \
    --dropout 0.0 \
    --crop_size 96 \
    --crops_per_volume 4 \
    --random_crop_size \
    --anisotropic_crops \
    --reslice_prob 0.0 \
    \
    --latent_channels 4 \
    --z_channels 4 \
    \
    --l1_weight 1.0 \
    --quant_weight 0.0 \
    --vgg_weight 0.5 \
    --gram_weight 0.1 \
    --laplacian_weight 0.1 \
    \
    --decoder_blocks_per_stage 2 2 0 \
    \
    --data_dir "$DATA_DIR"
