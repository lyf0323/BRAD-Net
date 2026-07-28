#!/bin/bash

# 优化版CFANet训练脚本 - AutoDL平台

# 使用方法:
# chmod +x run_train_autodl.sh
# ./run_train_autodl.sh

# 或直接运行命令:
# bash run_train_autodl.sh


echo "========================================================================"
echo "优化版CFANet训练 - Query引导 + 对比学习"
echo "========================================================================"

# ============================================================================
# 配置1: 推荐配置（Query 100+25 + 对比学习）
# ============================================================================

python train_optimized_cfanet.py \
    --decoder_type innovative \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --weight_bce 1.0 \
    --weight_dice 1.0 \
    --weight_boundary 0.5 \
    --weight_contrastive 0.2 \
    --epoch 40 \
    --lr 1e-4 \
    --batchsize 8 \
    --trainsize 352 \
    --channel 64 \
    --mamba_dim 96 \
    --clip_grad 0.5 \
    --save_epoch 5 \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth \
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir" \
    --use_cosine_lr true \
    --warmup_epochs 5 \
    --use_tensorboard true \
    --early_stopping_patience 20 \
    --multi_scale true \
    --freeze_resnet false \
    --resnet_lr_scale 0.1

echo ""
echo "训练完成！"
echo "模型保存路径: ./checkpoint/innovative_cfanet/"
echo "Tensorboard: tensorboard --logdir=./checkpoint/innovative_cfanet/logs --port=6006"

