@echo off
REM ============================================================================
REM Test with TTA - typically +2-3%% performance (no retraining) - Windows
REM ============================================================================

echo ==================================================
echo CFANet TTA Test - Quick Performance Boost
echo ==================================================
echo.
echo Config:
echo - TTA scales: 0.75x, 1.0x, 1.25x
echo - Horizontal flip: on
echo - Augmentations: 6x (3 scales x 2 flips)
echo - Expected gain: Dice +2-3%%
echo.
echo Note: TTA is ~6x slower at test time, but usually improves accuracy.
echo ==================================================
echo.

REM Set paths (modify for your local paths)
set MODEL_PATH=D:\CFANet-main\CFANet-main-improve\checkpoint\innovative_dual_stream_cfanet\OptimizedCFANet_best.pth
set TEST_ROOT=D:\CFANet-main\CFANet-main-improve\TestDataset\TestDataset\
set SAVE_ROOT=D:\CFANet-main\CFANet-main-improve\results\innovative_tta\

REM TTA config
set DECODER_TYPE=innovative
set USE_TTA=True
set TTA_SCALES=0.75,1.0,1.25
set TTA_FLIP=True

REM Test datasets
set DATASETS=CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB

REM Start testing
python test_optimized_cfanet.py ^
    --pth_path "%MODEL_PATH%" ^
    --test_root "%TEST_ROOT%" ^
    --save_root "%SAVE_ROOT%" ^
    --decoder_type %DECODER_TYPE% ^
    --testsize 352 ^
    --threshold 0.5 ^
    --save_results True ^
    --save_intermediate False ^
    --datasets "%DATASETS%" ^
    --use_tta %USE_TTA% ^
    --tta_scales "%TTA_SCALES%" ^
    --tta_flip %TTA_FLIP% ^
    --num_region_queries 100 ^
    --num_boundary_queries 25 ^
    --channel 64 ^
    --mamba_dim 96

echo.
echo ==================================================
echo TTA testing finished!
echo ==================================================
pause
