# AutoDL训练指南 - 优化版CFANet

## 快速开始

### Step 1: 上传代码到AutoDL

```bash
# 在AutoDL终端中
cd /root/autodl-tmp
git clone your_repo # 或上传代码

cd CFANet-main-improve
```

### Step 2: 确认数据集路径

```bash
# 确认训练集路径
ls /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/
# 应该看到: images/ masks/ edges/

# 确认测试集路径
ls /root/autodl-tmp/TestDataset/TestDataset/
# 应该看到: CVC-300/ CVC-ClinicDB/ Kvasir/ ETIS-LaribPolypDB/ CVC-ColonDB/
```

### Step 3: 开始训练

```bash
# 使用推荐配置（一行命令）
python train_optimized_cfanet.py \
    --decoder_type innovative \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth
```

**其余参数使用脚本默认值。**


## 推荐配置说明

### 默认配置（已优化）

```python
decoder_type='innovative' # 优化版创新解码器
num_region_queries=100 # 区域queries（完整配置）
num_boundary_queries=25 # 边界queries（完整配置）
weight_bce=1.0 # BCE损失权重
weight_dice=1.0 # Dice损失权重
weight_boundary=0.5 # 边界损失权重
weight_contrastive=0.2 # 对比学习权重
epoch=40 # 训练轮数
lr=1e-4 # 学习率
batchsize=8 # batch大小
trainsize=352 # 图像尺寸
use_cosine_lr=true # Cosine学习率
warmup_epochs=5 # Warmup轮数
```

### 为什么这样配置？

1. **decoder_type=innovative**
   - 简化MSCA（3分支，避免过拟合）
   - 完整Query（100+25，新创新）
   - 完整对比学习（3层，核心创新）

2. **weight_contrastive=0.2**
   - 不能太高（>0.3会不稳定）
   - 不能太低（<0.1效果不明显）
   - 0.2是经验最优值

3. **batchsize=8**
   - 平衡显存和性能
   - RTX 3090可用
   - 如果OOM，降到4


## 完整训练命令

### 推荐配置（复制即用）

```bash
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
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir,ETIS-LaribPolypDB,CVC-ColonDB" \
    --use_cosine_lr true \
    --warmup_epochs 5 \
    --use_tensorboard true \
    --early_stopping_patience 20 \
    --multi_scale true \
    --freeze_resnet false \
    --resnet_lr_scale 0.1
```


## 显存优化方案

### 如果遇到OOM（显存不足）

#### 方案1: 减小batch size（推荐）

```bash
python train_optimized_cfanet.py \
    --decoder_type innovative \
    --batchsize 4 \
    --epoch 60 \
    --lr 5e-5 \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_bs4/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth
```

#### 方案2: 使用混合精度（节省30%显存）

```bash
python train_optimized_cfanet.py \
    --decoder_type innovative \
    --use_amp true \
    --batchsize 12 \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_amp/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth
```

#### 方案3: 使用超轻量版（最后选择）

```bash
python train_optimized_cfanet.py \
    --decoder_type ultralight \
    --batchsize 8 \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/ultralight/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth
```


## 后台训练（推荐）

### 使用nohup

```bash
nohup python -u train_optimized_cfanet.py \
    --decoder_type innovative \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth \
    > train_innovative.log 2>&1 &

# 查看训练进度
tail -f train_innovative.log

# 查看进程
ps aux | grep python
```

### 使用screen（更方便）

```bash
# 1. 创建screen会话
screen -S cfanet

# 2. 在screen中运行训练
python train_optimized_cfanet.py \
    --decoder_type innovative \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth

# 3. 分离screen（训练继续运行）
# 按 Ctrl+A，然后按 D

# 4. 重新连接screen
screen -r cfanet

# 5. 查看所有screen
screen -ls

# 6. 结束screen
screen -X -S cfanet quit
```


## 监控训练

### Tensorboard（实时可视化）

```bash
# 启动Tensorboard
tensorboard --logdir=./checkpoint/innovative_cfanet/logs --port=6006 --bind_all

# AutoDL中访问
# 浏览器打开: http://your_instance_id.autodl.com:6006
```

### 查看日志

```bash
# 实时查看训练日志
tail -f checkpoint/innovative_cfanet/train.log

# 搜索最佳结果
grep "Best Model Saved" checkpoint/innovative_cfanet/train.log

# 查看最近20行
tail -20 checkpoint/innovative_cfanet/train.log
```

### GPU监控

```bash
# 实时监控（每秒刷新）
watch -n 1 nvidia-smi

# 查看GPU使用率
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# 查看显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```


## 预期训练时间

| 配置 | 数据集大小 | 预期时间/epoch | 总时间(40 epochs) |
|------|-----------|---------------|------------------|
| innovative, bs=8 | ~1500张 | ~3分钟 | ~2小时 |
| innovative, bs=4 | ~1500张 | ~5分钟 | ~3.5小时 |
| ultralight, bs=8 | ~1500张 | ~2分钟 | ~1.5小时 |


## 训练完成后

### 查看结果

```bash
# 查看所有保存的模型
ls -lh checkpoint/innovative_cfanet/*.pth

# 查看最佳模型信息
grep "Best Model Saved" checkpoint/innovative_cfanet/train.log | tail -1
```

### 使用最佳模型测试

```bash
python MyTest.py \
    --pth_path ./checkpoint/innovative_cfanet/OptimizedCFANet_best.pth \
    --decoder_type innovative
```


## 重要提示

### 1. Query配置很关键
- 推荐: 100个区域 + 25个边界
- 不要轻易简化（这是新创新，未验证）
- 如果显存不够，先减batch size，不要减query数量

### 2. 对比学习权重
- 起始值: 0.2
- 如果训练不稳定 → 降到0.1
- 如果边界不清晰 → 增到0.3

### 3. 消融实验建议
```bash
# 实验顺序
1. Baseline (original)
2. +简化解码器 (simplified)
3. +Query无对比 (innovative, weight_contrastive=0)
4. +完整版 (innovative, weight_contrastive=0.2) ← 主推荐
```


## 常见问题

**Q: 显存不够怎么办？**
```bash
# 优先: 减小batch size
--batchsize 4

# 次选: 混合精度
--use_amp true --batchsize 12

# 最后: 超轻量版
--decoder_type ultralight
```

**Q: 训练很慢怎么办？**
```bash
# 冻结ResNet
--freeze_resnet true

# 取消多尺度
--multi_scale false
```

**Q: 想快速验证效果？**
```bash
# 减少epoch
--epoch 20

# 减少validation频率
--save_epoch 10
```


```bash
python train_optimized_cfanet.py \
    --decoder_type innovative \
    --train_path /root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth
```
