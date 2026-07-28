"""
优化双分支CFANet测试脚本 - AutoDL平台适配版

主要功能:
1. 支持渐进式解码器和原始解码器
2. 多数据集批量测试
3. 完整评估指标（Dice, IoU, MAE, Precision, Recall, F1）
4. 边界评估
5. 可视化结果保存
6. AutoDL数据集路径自适应
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
from datetime import datetime

# 模型导入
from lib.BRAD_Net import create_optimized_dual_branch_cfanet

# 数据加载器
from utils.dataloader import test_dataset as TestDataLoader

# 评估函数
try:
    from utils.eva_funcs import eval_Smeasure, eval_mae, numpy2tensor
except:
    print("Warning: eva_funcs not found, using simple metrics only")
    eval_Smeasure = None
    eval_mae = None


# ============================================================================
# 评估指标计算
# ============================================================================

class Evaluator:
    """完整评估指标计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.mae_sum = 0.0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.count = 0

        # mDice计算所需的累积变量
        self.total_intersection = 0.0
        self.total_pred_sum = 0.0
        self.total_gt_sum = 0.0

    def update(self, pred, gt, threshold=0.5):
        """
        更新统计
        pred: 预测图 (numpy array, 0-1)
        gt: 真值图 (numpy array, 0-1)
        """
        # MAE
        mae = np.mean(np.abs(pred - gt))
        self.mae_sum += mae

        # 二值化
        pred_binary = (pred > threshold).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)

        # 交集和并集
        intersection = np.sum(pred_binary * gt_binary)
        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)
        union = pred_sum + gt_sum - intersection

        # 累积mDice计算所需的数据
        self.total_intersection += intersection
        self.total_pred_sum += pred_sum
        self.total_gt_sum += gt_sum

        # Dice (平均Dice)
        dice = (2.0 * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)
        self.dice_sum += dice

        # IoU
        iou = (intersection + 1e-8) / (union + 1e-8)
        self.iou_sum += iou

        # Precision
        precision = (intersection + 1e-8) / (pred_sum + 1e-8)
        self.precision_sum += precision

        # Recall
        recall = (intersection + 1e-8) / (gt_sum + 1e-8)
        self.recall_sum += recall

        # F1
        f1 = (2.0 * precision * recall + 1e-8) / (precision + recall + 1e-8)
        self.f1_sum += f1

        self.count += 1

    def get_metrics(self):
        """获取平均指标"""
        if self.count == 0:
            return {}

        # 计算mDice（全局Dice系数）
        mDice = (2.0 * self.total_intersection + 1e-8) / \
                (self.total_pred_sum + self.total_gt_sum + 1e-8)

        return {
            'MAE': self.mae_sum / self.count,
            'Dice': self.dice_sum / self.count,
            'mDice': mDice,
            'IoU': self.iou_sum / self.count,
            'Precision': self.precision_sum / self.count,
            'Recall': self.recall_sum / self.count,
            'F1': self.f1_sum / self.count
        }


def calculate_boundary_metrics(pred, gt, threshold=0.5):
    """
    计算边界评估指标
    pred: 预测图 (numpy array, 0-1)
    gt: 真值图 (numpy array, 0-1)
    """
    # 二值化
    pred_binary = (pred > threshold).astype(np.uint8) * 255
    gt_binary = (gt > 0.5).astype(np.uint8) * 255

    # 提取边界（形态学梯度）
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    gt_boundary = cv2.morphologyEx(gt_binary, cv2.MORPH_GRADIENT, kernel)

    # 归一化
    pred_boundary = pred_boundary.astype(np.float32) / 255.0
    gt_boundary = gt_boundary.astype(np.float32) / 255.0

    # 计算指标
    intersection = np.sum(pred_boundary * gt_boundary)
    pred_sum = np.sum(pred_boundary)
    gt_sum = np.sum(gt_boundary)

    boundary_dice = (2.0 * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)
    boundary_precision = (intersection + 1e-8) / (pred_sum + 1e-8)
    boundary_recall = (intersection + 1e-8) / (gt_sum + 1e-8)
    boundary_f1 = (2.0 * boundary_precision * boundary_recall + 1e-8) / \
                  (boundary_precision + boundary_recall + 1e-8)

    return {
        'Boundary_Dice': boundary_dice,
        'Boundary_Precision': boundary_precision,
        'Boundary_Recall': boundary_recall,
        'Boundary_F1': boundary_f1
    }


# ============================================================================
# 测试函数
# ============================================================================

def test_with_tta(model, image, opt):
    """
    Test-Time Augmentation (TTA)
    使用多尺度 + 水平翻转来提升测试性能

    Args:
        model: 模型
        image: 输入图像 [1, 3, H, W]
        opt: 参数

    Returns:
        final_pred: 融合后的预测结果
        edge_map: 边界图（如果有）
    """
    predictions = []
    edge_maps = []
    original_size = image.shape[2:]

    # TTA配置
    if opt.use_tta:
        scales = opt.tta_scales # 例如 [0.75, 1.0, 1.25]
        use_flip = opt.tta_flip
    else:
        scales = [1.0]
        use_flip = False

    for scale in scales:
        # 缩放图像
        if scale != 1.0:
            h, w = int(original_size[0] * scale), int(original_size[1] * scale)
            # 确保是32的倍数
            h = int(np.round(h / 32) * 32)
            w = int(np.round(w / 32) * 32)
            img_scaled = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        else:
            img_scaled = image

        # 原图预测
        if opt.decoder_type in ['innovative', 'ultralight']:
            outputs = model(img_scaled, return_contrast_outputs=False)
            if isinstance(outputs, dict):
                pred = outputs['pred_final']
                edge = outputs.get('edge_map', None)
            else:
                _, _, pred, _ = outputs
                edge = None
        elif opt.decoder_type == 'simplified':
            outputs = model(img_scaled, return_intermediates=False)
            if isinstance(outputs, dict):
                pred = outputs['pred_final']
            else:
                _, _, _, pred = outputs
            edge = None
        else:
            _, _, _, pred = model(img_scaled)
            edge = None

        # 恢复到原始尺寸
        pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
        predictions.append(pred)

        if edge is not None:
            edge = F.interpolate(edge, size=original_size, mode='bilinear', align_corners=False)
            edge_maps.append(edge)

        # 水平翻转增强
        if use_flip:
            img_flipped = torch.flip(img_scaled, dims=[3])

            if opt.decoder_type in ['innovative', 'ultralight']:
                outputs_flip = model(img_flipped, return_contrast_outputs=False)
                if isinstance(outputs_flip, dict):
                    pred_flip = outputs_flip['pred_final']
                    edge_flip = outputs_flip.get('edge_map', None)
                else:
                    _, _, pred_flip, _ = outputs_flip
                    edge_flip = None
            elif opt.decoder_type == 'simplified':
                outputs_flip = model(img_flipped, return_intermediates=False)
                if isinstance(outputs_flip, dict):
                    pred_flip = outputs_flip['pred_final']
                else:
                    _, _, _, pred_flip = outputs_flip
                edge_flip = None
            else:
                _, _, _, pred_flip = model(img_flipped)
                edge_flip = None

            # 翻转回来
            pred_flip = torch.flip(pred_flip, dims=[3])
            pred_flip = F.interpolate(pred_flip, size=original_size, mode='bilinear', align_corners=False)
            predictions.append(pred_flip)

            if edge_flip is not None:
                edge_flip = torch.flip(edge_flip, dims=[3])
                edge_flip = F.interpolate(edge_flip, size=original_size, mode='bilinear', align_corners=False)
                edge_maps.append(edge_flip)

    # 融合所有预测
    final_pred = torch.stack(predictions).mean(dim=0)
    final_edge = torch.stack(edge_maps).mean(dim=0) if len(edge_maps) > 0 else None

    return final_pred, final_edge


def test_dataset(model, test_loader, save_path, opt):
    """
    测试单个数据集
    """
    model.eval()

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)

    # 评估器
    evaluator = Evaluator()
    boundary_metrics = {'Boundary_Dice': 0, 'Boundary_Precision': 0,
                       'Boundary_Recall': 0, 'Boundary_F1': 0}

    tta_info = ""
    if opt.use_tta:
        tta_info = f" (TTA: scales={opt.tta_scales}, flip={opt.tta_flip})"

    print(f"\n{'='*80}")
    print(f"测试数据集: {test_loader.size} 张图像{tta_info}")
    print(f"保存路径: {save_path}")
    print(f"{'='*80}\n")

    with torch.no_grad():
        pbar = tqdm(range(test_loader.size), desc="Testing")

        for i in pbar:
            # 加载数据
            image, gt, name = test_loader.load_data()

            # 真值处理
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            # 移到GPU
            image = image.cuda()

            # 推理（使用TTA或普通推理）
            if opt.use_tta:
                # 使用TTA
                res, edge_map = test_with_tta(model, image, opt)
            else:
                # 普通推理（原始方法）
                if opt.decoder_type in ['innovative', 'ultralight']:
                    # 创新解码器：支持对比学习输出
                    outputs = model(image, return_contrast_outputs=True)
                    res = outputs['pred_final']
                    edge_map = outputs.get('edge_map', None)

                    # 可选：保存中间结果和边界图
                    if opt.save_intermediate:
                        # 保存边界图
                        if 'edge_map' in outputs and outputs['edge_map'] is not None:
                            edge_map_vis = outputs['edge_map']
                            edge_map_vis = F.interpolate(edge_map_vis, size=gt.shape,
                                               mode='bilinear', align_corners=False)
                            edge_map_vis = edge_map_vis.sigmoid().data.cpu().numpy().squeeze()
                            edge_map_vis = (edge_map_vis - edge_map_vis.min()) / (edge_map_vis.max() - edge_map_vis.min() + 1e-8)
                            edge_save_path = save_path.replace('/final/', '/edge/')
                            os.makedirs(edge_save_path, exist_ok=True)
                            cv2.imwrite(edge_save_path + name, edge_map_vis * 255)

                        # 保存各阶段预测
                        for stage in ['pred1', 'pred2', 'pred3', 'pred4']:
                            if stage in outputs:
                                stage_res = outputs[stage]
                                stage_res = F.interpolate(stage_res, size=gt.shape,
                                                         mode='bilinear', align_corners=False)
                                stage_res = stage_res.sigmoid().data.cpu().numpy().squeeze()
                                stage_res = (stage_res - stage_res.min()) / (stage_res.max() - stage_res.min() + 1e-8)

                                stage_save_path = save_path.replace('/final/', f'/{stage}/')
                                os.makedirs(stage_save_path, exist_ok=True)
                                cv2.imwrite(stage_save_path + name, stage_res * 255)
                elif opt.decoder_type == 'simplified':
                    # 简化解码器：支持中间结果
                    outputs = model(image, return_intermediates=True)
                    res = outputs['pred_final']
                    edge_map = None

                    if opt.save_intermediate:
                        for stage in ['pred1', 'pred2', 'pred3', 'pred4']:
                            stage_res = outputs[stage]
                            stage_res = F.interpolate(stage_res, size=gt.shape,
                                                     mode='bilinear', align_corners=False)
                            stage_res = stage_res.sigmoid().data.cpu().numpy().squeeze()
                            stage_res = (stage_res - stage_res.min()) / (stage_res.max() - stage_res.min() + 1e-8)

                            stage_save_path = save_path.replace('/final/', f'/{stage}/')
                            os.makedirs(stage_save_path, exist_ok=True)
                            cv2.imwrite(stage_save_path + name, stage_res * 255)
                else:
                    # 原始解码器
                    _, _, _, res = model(image)
                    edge_map = None

            # 后处理
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # 保存预测结果
            if opt.save_results:
                # 保存灰度图（0-255）
                cv2.imwrite(save_path + name, res * 255)

                # 如果需要保存二值化结果
                if opt.save_binary:
                    binary_res = (res > opt.threshold).astype(np.uint8) * 255
                    binary_save_path = save_path.replace('/final/', '/binary/')
                    os.makedirs(binary_save_path, exist_ok=True)
                    cv2.imwrite(binary_save_path + name, binary_res)

            # 更新评估指标
            evaluator.update(res, gt, threshold=opt.threshold)

            # 边界指标（每10张图计算一次以节省时间）
            if i % 10 == 0 or i == test_loader.size - 1:
                boundary = calculate_boundary_metrics(res, gt, threshold=opt.threshold)
                for key in boundary_metrics:
                    boundary_metrics[key] += boundary[key]

            # 更新进度条
            metrics = evaluator.get_metrics()
            pbar.set_postfix({
                'mDice': f"{metrics['mDice']:.4f}",
                'Dice': f"{metrics['Dice']:.4f}",
                'MAE': f"{metrics['MAE']:.4f}"
            })

    # 计算最终指标
    final_metrics = evaluator.get_metrics()

    # 边界指标平均
    boundary_count = (test_loader.size // 10) + 1
    for key in boundary_metrics:
        boundary_metrics[key] /= boundary_count

    # 合并指标
    final_metrics.update(boundary_metrics)

    return final_metrics


# ============================================================================
# 主测试流程
# ============================================================================

def main():
    # 参数解析
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--testsize', type=int, default=352, help='测试图像尺寸')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')

    # 模型参数
    parser.add_argument('--pth_path', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet_fixed_singlesscaleOptimizedCFANet_best0.9599.pth',
                       help='模型权重路径')
    parser.add_argument('--channel', type=int, default=64, help='解码器基础通道数')
    parser.add_argument('--mamba_dim', type=int, default=96, help='Mamba嵌入维度')
    parser.add_argument('--decoder_type', type=str, default='innovative',
                       choices=['innovative', 'ultralight', 'simplified', 'original'],
                       help='解码器类型: innovative(推荐), ultralight, simplified, original')
    parser.add_argument('--num_region_queries', type=int, default=100,
                       help='区域查询数量（仅用于innovative/ultralight）')
    parser.add_argument('--num_boundary_queries', type=int, default=25,
                       help='边界查询数量（仅用于innovative/ultralight）')

    # 数据路径（AutoDL适配）
    parser.add_argument('--test_root', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/TestDataset/TestDataset/',
                       help='测试数据集根路径')
    parser.add_argument('--save_root', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/best_model_test/',
                       help='结果保存根路径')

    # 测试选项
    parser.add_argument('--save_results', type=bool, default=True,
                       help='保存预测结果')
    parser.add_argument('--save_binary', type=bool, default=True,
                       help='保存二值化结果（0或255）')
    parser.add_argument('--save_intermediate', type=bool, default=False,
                       help='保存中间阶段结果（仅渐进式解码器）')
    parser.add_argument('--datasets', type=str,
                       default='CVC-300',
                       help='测试数据集列表（逗号分隔）')

    # TTA增强选项
    parser.add_argument('--use_tta', type=bool, default=False,
                       help='使用Test-Time Augmentation (TTA)进行测试增强')
    parser.add_argument('--tta_scales', type=str, default='0.75,1.0,1.25',
                       help='TTA多尺度列表（逗号分隔），推荐: 0.75,1.0,1.25')
    parser.add_argument('--tta_flip', type=bool, default=True,
                       help='TTA是否使用水平翻转')

    opt = parser.parse_args()

    # 解析TTA scales
    if opt.use_tta:
        opt.tta_scales = [float(s.strip()) for s in opt.tta_scales.split(',')]
    else:
        opt.tta_scales = [1.0]

    print("=" * 80)
    print("优化双分支CFANet测试程序")
    print("=" * 80)
    print(f"配置信息:")
    print(f"• 模型路径: {opt.pth_path}")
    decoder_type_names = {
        'innovative': '创新解码器（Query + 对比学习 + 双流边界）',
        'ultralight': '超轻量解码器',
        'simplified': '简化渐进式解码器',
        'original': '原始CFANet解码器'
    }
    print(f"• 解码器类型: {decoder_type_names.get(opt.decoder_type, opt.decoder_type)}")
    if opt.decoder_type in ['innovative', 'ultralight']:
        print(f"• Query配置: {opt.num_region_queries}区域 + {opt.num_boundary_queries}边界")
    print(f"• 测试尺寸: {opt.testsize}")
    print(f"• 二值化阈值: {opt.threshold}")
    print(f"• 保存结果: {opt.save_results}")
    print(f"• 保存二值化结果: {opt.save_binary}")
    print(f"• 保存中间结果: {opt.save_intermediate}")

    # TTA信息
    if opt.use_tta:
        print(f"• TTA增强: 开启")
        print(f"- 多尺度: {opt.tta_scales}")
        print(f"- 水平翻转: {opt.tta_flip}")
        num_augments = len(opt.tta_scales) * (2 if opt.tta_flip else 1)
        print(f"- 增强次数: {num_augments}x (每张图预测{num_augments}次后融合)")
    else:
        print(f"• TTA增强: 关闭 (使用 --use_tta True 开启)")

    print("=" * 80)

    # 创建模型
    print("\n创建模型...")
    model = create_optimized_dual_branch_cfanet(
        channel=opt.channel,
        mamba_dim=opt.mamba_dim,
        auto_download_weights=False,
        decoder_type=opt.decoder_type,
        num_region_queries=opt.num_region_queries,
        num_boundary_queries=opt.num_boundary_queries
    ).cuda()

    # 加载权重
    print(f"加载模型权重: {opt.pth_path}")
    if not os.path.exists(opt.pth_path):
        print(f"错误: 模型文件不存在: {opt.pth_path}")
        return

    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()

    # 统计参数
    params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成: {params/1e6:.2f}M 参数")

    # 测试数据集列表
    dataset_list = opt.datasets.split(',')

    # 存储所有数据集的结果
    all_results = {}

    # 遍历测试数据集
    for dataset_name in dataset_list:
        print(f"\n{'='*80}")
        print(f"测试数据集: {dataset_name}")
        print(f"{'='*80}")

        # 数据路径
        data_path = os.path.join(opt.test_root, dataset_name)
        if not os.path.exists(data_path):
            print(f"警告: 数据集不存在: {data_path}")
            continue

        image_root = os.path.join(data_path, 'images/')
        gt_root = os.path.join(data_path, 'masks/')

        # 结果保存路径
        save_path = os.path.join(opt.save_root, dataset_name, 'final/')

        # 加载测试集
        try:
            test_loader = TestDataLoader(image_root, gt_root, opt.testsize)
        except Exception as e:
            print(f"错误: 无法加载数据集: {e}")
            continue

        # 测试
        try:
            metrics = test_dataset(model, test_loader, save_path, opt)
            all_results[dataset_name] = metrics

            # 打印结果
            print(f"\n{'='*80}")
            print(f"{dataset_name} 测试结果:")
            print(f"{'='*80}")
            print(f"• mDice: {metrics['mDice']:.4f} (全局Dice)")
            print(f"• Dice: {metrics['Dice']:.4f} (平均Dice)")
            print(f"• IoU: {metrics['IoU']:.4f}")
            print(f"• MAE: {metrics['MAE']:.4f}")
            print(f"• Precision: {metrics['Precision']:.4f}")
            print(f"• Recall: {metrics['Recall']:.4f}")
            print(f"• F1: {metrics['F1']:.4f}")
            print(f"\n边界指标:")
            print(f"• Boundary Dice: {metrics['Boundary_Dice']:.4f}")
            print(f"• Boundary F1: {metrics['Boundary_F1']:.4f}")
            print(f"{'='*80}")

        except Exception as e:
            print(f"错误: 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 总结所有数据集的平均结果
    if all_results:
        print(f"\n{'='*80}")
        print("所有数据集平均结果")
        print(f"{'='*80}")

        # 计算平均
        avg_metrics = {}
        for key in all_results[list(all_results.keys())[0]].keys():
            avg_metrics[key] = np.mean([result[key] for result in all_results.values()])

        print(f"\n平均指标:")
        print(f"• mDice: {avg_metrics['mDice']:.4f} (全局Dice平均)")
        print(f"• avgDice: {avg_metrics['Dice']:.4f} (平均Dice)")
        print(f"• mIoU: {avg_metrics['IoU']:.4f}")
        print(f"• mMAE: {avg_metrics['MAE']:.4f}")
        print(f"• mPrecision: {avg_metrics['Precision']:.4f}")
        print(f"• mRecall: {avg_metrics['Recall']:.4f}")
        print(f"• mF1: {avg_metrics['F1']:.4f}")
        print(f"\n平均边界指标:")
        print(f"• mBoundary Dice: {avg_metrics['Boundary_Dice']:.4f}")
        print(f"• mBoundary F1: {avg_metrics['Boundary_F1']:.4f}")
        print(f"{'='*80}")

        # 保存结果到文件
        results_file = os.path.join(opt.save_root, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"测试时间: {datetime.now()}\n")
            f.write(f"模型: {opt.pth_path}\n")
            f.write("=" * 80 + "\n\n")

            # 写入各数据集结果
            for dataset_name, metrics in all_results.items():
                f.write(f"{dataset_name}:\n")
                for key, value in metrics.items():
                    f.write(f" {key}: {value:.4f}\n")
                f.write("\n")

            # 写入平均结果
            f.write("=" * 80 + "\n")
            f.write("平均结果:\n")
            for key, value in avg_metrics.items():
                f.write(f" {key}: {value:.4f}\n")

        print(f"\n结果已保存到: {results_file}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

