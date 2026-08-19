#!/bin/bash

# ============================================================================
# Test with TTA - typically +2-3% performance (no retraining required)
# ============================================================================

echo "=================================================="
echo "CFANet TTA Test - Quick Performance Boost"
echo "=================================================="
echo ""
echo "Config:"
echo "- TTA scales: 0.75x, 1.0x, 1.25x"
echo "- Horizontal flip: on"
echo "- Augmentations: 6x (3 scales x 2 flips)"
echo "- Expected gain: Dice +2-3%"
echo ""
echo "Note: TTA is ~6x slower at test time, but usually improves accuracy."
echo "=================================================="
echo ""


# Set paths (modify for your environment)
MODEL_PATH="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TEST_ROOT="/root/autodl-tmp/TestDataset/TestDataset/"
SAVE_ROOT="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/innovative_tta/"

# TTA config
DECODER_TYPE="innovative" # innovative, ultralight, simplified, original
USE_TTA=True
TTA_SCALES="0.75,1.0,1.25" # multi-scale
TTA_FLIP=True # horizontal flip

# Test datasets
DATASETS="CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB"

# Start testing
python test_optimized_cfanet.py \
    --pth_path "$MODEL_PATH" \
    --test_root "$TEST_ROOT" \
    --save_root "$SAVE_ROOT" \
    --decoder_type "$DECODER_TYPE" \
    --testsize 352 \
    --threshold 0.5 \
    --save_results True \
    --save_intermediate False \
    --datasets "$DATASETS" \
    --use_tta "$USE_TTA" \
    --tta_scales "$TTA_SCALES" \
    --tta_flip "$TTA_FLIP" \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --channel 64 \
    --mamba_dim 96

echo ""
echo "=================================================="
echo "TTA testing finished!"
echo "=================================================="
