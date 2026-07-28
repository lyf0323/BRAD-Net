#!/bin/bash

# ============================================================================
# 使用TTA增强测试 - 提升2-3%性能 (无需重新训练!)
# ============================================================================

echo "=================================================="
echo "CFANet TTA增强测试 - 快速提升性能"
echo "=================================================="
echo ""
echo "配置："
echo "• TTA多尺度: 0.75x, 1.0x, 1.25x"
echo "• 水平翻转: 开启"
echo "• 增强次数: 6x (3尺度 × 2翻转)"
echo "• 预期提升: Dice +2-3%"
echo ""
echo "注意: TTA会让测试速度变慢约6倍，但能显著提升精度!"
echo "=================================================="
echo ""

# 设置路径 (请根据你的实际路径修改)
MODEL_PATH="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TEST_ROOT="/root/autodl-tmp/TestDataset/TestDataset/"
SAVE_ROOT="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/innovative_tta/"

# TTA配置
DECODER_TYPE="innovative" # innovative, ultralight, simplified, original
USE_TTA=True
TTA_SCALES="0.75,1.0,1.25" # 多尺度
TTA_FLIP=True # 水平翻转

# 测试数据集
DATASETS="CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB"

# 开始测试
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
echo "TTA测试完成!"
echo "=================================================="

