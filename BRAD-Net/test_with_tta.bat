@echo off
REM ============================================================================
REM 使用TTA增强测试 - 提升2-3%%性能 (无需重新训练!) - Windows版本
REM ============================================================================

echo ==================================================
echo CFANet TTA增强测试 - 快速提升性能
echo ==================================================
echo.
echo 配置：
echo • TTA多尺度: 0.75x, 1.0x, 1.25x
echo • 水平翻转: 开启
echo • 增强次数: 6x (3尺度 × 2翻转)
echo • 预期提升: Dice +2-3%%
echo.
echo 注意: TTA会让测试速度变慢约6倍，但能显著提升精度!
echo ==================================================
echo.

REM 设置路径 (请根据你的实际路径修改)
set MODEL_PATH=D:\CFANet-main\CFANet-main-improve\checkpoint\innovative_dual_stream_cfanet\OptimizedCFANet_best.pth
set TEST_ROOT=D:\CFANet-main\CFANet-main-improve\TestDataset\TestDataset\
set SAVE_ROOT=D:\CFANet-main\CFANet-main-improve\results\innovative_tta\

REM TTA配置
set DECODER_TYPE=innovative
set USE_TTA=True
set TTA_SCALES=0.75,1.0,1.25
set TTA_FLIP=True

REM 测试数据集
set DATASETS=CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB

REM 开始测试
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
echo TTA测试完成!
echo ==================================================
pause

