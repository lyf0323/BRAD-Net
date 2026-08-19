@echo off
chcp 65001 >nul
REM ################################################################################
REM Resume training from the best epoch (Windows)
REM
REM Features:
REM 1. Resume from the best checkpoint
REM 2. Use the optimized contrastive loss (lower threshold, higher temperature)
REM 3. Continue remaining epochs
REM
REM Changes:
REM - Contrastive threshold: 10 -> 3 (allow contrastive learning on small maps)
REM - Sampling: boundary 512 -> 256, min 10 -> 3
REM - Temperature: 0.07 -> 0.5 (more stable gradients)
REM
REM
REM ################################################################################

echo ==========================================================================
echo Resume training from best epoch
echo ==========================================================================
echo.
echo Key changes:
echo    Contrastive threshold: 10 -^> 3 (allow small feature maps)
echo    Boundary sample cap: 512 -^> 256 (less memory)
echo    Temperature: 0.07 -^> 0.5 (more stable gradients)
echo    Min samples: boundary^>=3, region^>=5
echo.
echo ==========================================================================

REM Config (modify for your local paths)
set BEST_MODEL=D:\CFANet-main\CFANet-main-improve\checkpoint\innovative_dual_stream_cfanet\OptimizedCFANet_best.pth
set TRAIN_DATA=D:\CFANet-main\TrainDatasetEdges
set TEST_DATA=D:\CFANet-main\TestDataset
set SAVE_PATH=./checkpoint/innovative_dual_stream_cfanet_resume

REM Check model file
if not exist "%BEST_MODEL%" (
    echo Error: best model file not found
    echo    Path: %BEST_MODEL%
    pause
    exit /b 1
)

echo Found best model: %BEST_MODEL%
echo.
echo Starting resume training...
echo.

REM Resume training
python train_optimized_cfanet.py ^
    --epoch 40 ^
    --lr 1e-4 ^
    --batchsize 8 ^
    --trainsize 352 ^
    --clip_grad 2.0 ^
    --save_epoch 5 ^
    --train_path "%TRAIN_DATA%" ^
    --val_path "%TEST_DATA%" ^
    --save_path "%SAVE_PATH%" ^
    --channel 64 ^
    --mamba_dim 96 ^
    --decoder_type innovative ^
    --num_region_queries 100 ^
    --num_boundary_queries 25 ^
    --weight_bce 1.0 ^
    --weight_dice 1.0 ^
    --weight_boundary 0.5 ^
    --weight_contrastive 0.2 ^
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth ^
    --resume "%BEST_MODEL%" ^
    --use_cosine_lr true ^
    --warmup_epochs 2 ^
    --use_amp false ^
    --freeze_resnet false ^
    --resnet_lr_scale 0.1 ^
    --multi_scale true ^
    --use_tensorboard true ^
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir" ^
    --early_stopping_patience 15

echo.
echo ==========================================================================
echo Training finished
echo ==========================================================================
pause
