#!/bin/bash


# 恢复训练脚本 - 从最佳epoch继续训练

# 功能：
# 1. 从最佳模型检查点恢复训练
# 2. 使用优化后的对比学习损失（降低阈值，提高温度）
# 3. 继续训练剩余的epochs

# 修改内容：
# - 对比学习阈值: 10 → 3（允许小特征图上的对比学习）
# - 采样策略: boundary从512→256, 最小从10→3
# - 温度参数: 0.07 → 0.5（更稳定的梯度）


echo "=========================================================================="
echo "恢复训练：从最佳Epoch继续"
echo "=========================================================================="
echo ""
echo "关键修改："
echo "对比学习阈值: 10 → 3（允许小特征图）"
echo "边界采样上限: 512 → 256（减少内存）"
echo "温度参数: 0.07 → 0.5（更稳定梯度）"
echo "最小样本数: boundary≥3, region≥5"
echo ""
echo "=========================================================================="

# 配置参数
BEST_MODEL="/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet/OptimizedCFANet_best.pth"
TRAIN_DATA="/root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges"
TEST_DATA="/root/autodl-tmp/TestDataset/TestDataset"
SAVE_PATH="./checkpoint/innovative_dual_stream_cfanet_resume"

# 检查模型文件是否存在
if [ ! -f "$BEST_MODEL" ]; then
    echo "错误：找不到最佳模型文件"
    echo "路径: $BEST_MODEL"
    exit 1
fi

echo "找到最佳模型: $BEST_MODEL"
echo ""
echo "开始恢复训练..."
echo ""

# 恢复训练命令
python train_optimized_cfanet.py \
    --epoch 40 \
    --lr 1e-4 \
    --batchsize 8 \
    --trainsize 352 \
    --clip_grad 2.0 \
    --save_epoch 5 \
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
    --warmup_epochs 2 \
    --use_amp false \
    --freeze_resnet false \
    --resnet_lr_scale 0.1 \
    --multi_scale true \
    --use_tensorboard true \
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir" \
    --early_stopping_patience 15

echo ""
echo "=========================================================================="
echo "训练完成"
echo "=========================================================================="

