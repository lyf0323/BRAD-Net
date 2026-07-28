#!/bin/bash

# ============================================================================
# 创新解码器测试脚本 - AutoDL平台
# ============================================================================

echo "================================================================================================"
echo "开始测试优化版创新解码器（Innovative Decoder with Dual-Stream Boundary）"
echo "================================================================================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 测试配置
MODEL_PATH="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TEST_ROOT="/root/autodl-tmp/TestDataset/TestDataset/"
SAVE_ROOT="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/innovative_dual_stream/"
DATASETS="CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请检查训练是否完成或路径是否正确"
    exit 1
fi

echo "测试配置:"
echo "• 模型路径: $MODEL_PATH"
echo "• 解码器类型: innovative（Query + 对比学习 + 双流边界）"
echo "• 测试数据集: $DATASETS"
echo "• 结果保存: $SAVE_ROOT"
echo "================================================================================================"
echo ""

# 运行测试
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
echo "测试完成！"
echo "结果保存在: $SAVE_ROOT"
echo "================================================================================================"

