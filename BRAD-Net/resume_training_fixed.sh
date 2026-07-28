#!/bin/bash


# 修复后的恢复训练脚本 - 对比学习NaN问题已修复

# 修复内容：
# 1. 对比学习阈值: 10 → 3（允许小特征图）
# 2. 数值稳定的损失函数（避免log(0)导致NaN）
# 3. 温度参数: 0.07 → 0.5（更稳定的梯度）
# 4. 采样策略优化（boundary≥3, region≥5）
# 5. Loss范围限制（clamp到[0, 2]）

# 预期效果：
# - Contrast Loss: 0.05-0.20（正常工作）
# - mDice: 0.913 → 0.920-0.925


echo "=========================================================================="
echo "恢复训练：从最佳Epoch继续（对比学习已修复）"
echo "=========================================================================="
echo ""
echo "修复内容："
echo "1. 对比学习阈值: 10 → 3"
echo "2. 数值稳定损失函数（无NaN）"
echo "3. 温度参数: 0.07 → 0.5"
echo "4. 采样策略优化"
echo "5. Loss范围保护"
echo ""
echo "预期效果："
echo "- Contrast Loss: 0.05-0.20 "
echo "- mDice提升: +0.7-1.2%"
echo "- 无NaN错误"
echo ""
echo "=========================================================================="

# 配置参数
BEST_MODEL="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TRAIN_DATA="/root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges"
TEST_DATA="/root/autodl-tmp/TestDataset/TestDataset"
SAVE_PATH="./checkpoint/innovative_dual_stream_cfanet_fixed"

# 检查模型文件
if [ ! -f "$BEST_MODEL" ]; then
    echo "错误：找不到最佳模型文件"
    echo "路径: $BEST_MODEL"
    exit 1
fi

echo "找到最佳模型: $BEST_MODEL"
echo ""
echo "开始恢复训练（修复版本）..."
echo ""

# 恢复训练命令
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
echo "训练完成"
echo ""
echo "请检查："
echo "1. Contrast Loss 应该在 0.05-0.20"
echo "2. 无 NaN 警告"
echo "3. mDice 稳步提升"
echo "=========================================================================="
