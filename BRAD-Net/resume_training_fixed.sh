#!/bin/bash


# Resume training after contrastive-learning NaN fix

# Fixes:
# 1. Contrastive threshold: 10 -> 3 (allow small feature maps)
# 2. Numerically stable loss (avoid log(0) NaNs)
# 3. Temperature: 0.07 -> 0.5 (more stable gradients)
# 4. Sampling: boundary >= 3, region >= 5
# 5. Loss clamp to [0, 2]

# Expected:
# - Contrast Loss: 0.05-0.20 (working normally)
# - mDice: 0.913 -> 0.920-0.925


echo "=========================================================================="
echo "Resume training from best epoch (contrastive learning fixed)"
echo "=========================================================================="
echo ""
echo "Fixes:"
echo "1. Contrastive threshold: 10 -> 3"
echo "2. Numerically stable loss (no NaN)"
echo "3. Temperature: 0.07 -> 0.5"
echo "4. Sampling strategy optimized"
echo "5. Loss range protection"
echo ""
echo "Expected:"
echo "- Contrast Loss: 0.05-0.20 "
echo "- mDice gain: +0.7-1.2%"
echo "- No NaN errors"
echo ""
echo "=========================================================================="

# Config
BEST_MODEL="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TRAIN_DATA="/root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges"
TEST_DATA="/root/autodl-tmp/TestDataset/TestDataset"
SAVE_PATH="./checkpoint/innovative_dual_stream_cfanet_fixed"

# Check model file
if [ ! -f "$BEST_MODEL" ]; then
    echo "Error: best model file not found"
    echo "Path: $BEST_MODEL"
    exit 1
fi

echo "Found best model: $BEST_MODEL"
echo ""
echo "Starting resume training (fixed version)..."
echo ""

# Resume training
python train_optimized_cfanet.py \
    --epoch 15 \
    --lr 5e-5 \
    --batchsize 8 \
    --trainsize 352 \
    --clip_grad 2.0 \
    --save_epoch 3 \
    --train_path "$TRAIN_DATA" \
    --val_path "$TEST_DATA" \
    --save_path "$SAVE_PATH" \
    --channel 64 \
    --mamba_dim 96 \
    --decoder_type innovative \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --weight_bce 1.0 \
    --weight_dice 1.0 \
    --weight_boundary 0.5 \
    --weight_contrastive 0.2 \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth \
    --resume "$BEST_MODEL" \
    --use_cosine_lr true \
    --warmup_epochs 3 \
    --use_amp false \
    --freeze_resnet false \
    --resnet_lr_scale 0.1 \
    --multi_scale true \
    --use_tensorboard true \
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir" \
    --early_stopping_patience 10

echo ""
echo "=========================================================================="
echo "Training finished"
echo ""
echo "Please check:"
echo "1. Contrast Loss should be in 0.05-0.20"
echo "2. No NaN warnings"
echo "3. mDice steadily improving"
echo "=========================================================================="
