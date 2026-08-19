#!/bin/bash

# ============================================================================
# Innovative decoder test script for AutoDL
# ============================================================================

echo "================================================================================================"
echo "Testing optimized innovative decoder (Query + contrastive + dual-stream boundary)"
echo "================================================================================================"

# Environment
export CUDA_VISIBLE_DEVICES=0

# Test config
MODEL_PATH="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TEST_ROOT="/root/autodl-tmp/TestDataset/TestDataset/"
SAVE_ROOT="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/innovative_dual_stream/"
DATASETS="CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB"

# Check model file
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: model file not found: $MODEL_PATH"
    echo "Please check whether training finished and the path is correct"
    exit 1
fi

echo "Test config:"
echo "- Model path: $MODEL_PATH"
echo "- Decoder type: innovative (Query + contrastive + dual-stream boundary)"
echo "- Test datasets: $DATASETS"
echo "- Save results to: $SAVE_ROOT"
echo "================================================================================================"
echo ""

# Run test
python test_optimized_cfanet.py \
    --pth_path "$MODEL_PATH" \
    --test_root "$TEST_ROOT" \
    --save_root "$SAVE_ROOT" \
    --datasets "$DATASETS" \
    --decoder_type innovative \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --testsize 352 \
    --threshold 0.5 \
    --save_results True \
    --save_intermediate False \
    --channel 64 \
    --mamba_dim 96

echo ""
echo "================================================================================================"
echo "Testing finished!"
echo "Results saved to: $SAVE_ROOT"
echo "================================================================================================"
