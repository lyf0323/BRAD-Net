
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
from datetime import datetime

# Model imports
from lib.BRAD_Net import create_optimized_dual_branch_cfanet

# Data loaders
from utils.dataloader import test_dataset as TestDataLoader

# Evaluation functions
try:
    from utils.eva_funcs import eval_Smeasure, eval_mae, numpy2tensor
except:
    print("Warning: eva_funcs not found, using simple metrics only")
    eval_Smeasure = None
    eval_mae = None


# ============================================================================
# Evaluation Metric Computation
# ============================================================================

class Evaluator:
    """Full evaluation metric calculator"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset statistics"""
        self.mae_sum = 0.0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.count = 0

        # Accumulated variables required for mDice computation
        self.total_intersection = 0.0
        self.total_pred_sum = 0.0
        self.total_gt_sum = 0.0

    def update(self, pred, gt, threshold=0.5):
        """
        Update statistics
        pred: prediction map (numpy array, 0-1)
        gt: ground-truth map (numpy array, 0-1)
        """
        # MAE
        mae = np.mean(np.abs(pred - gt))
        self.mae_sum += mae

        # Binarization
        pred_binary = (pred > threshold).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)

        # Intersection and union
        intersection = np.sum(pred_binary * gt_binary)
        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)
        union = pred_sum + gt_sum - intersection

        # Accumulate data required for mDice computation
        self.total_intersection += intersection
        self.total_pred_sum += pred_sum
        self.total_gt_sum += gt_sum

        # Dice (mean Dice)
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
        """Get average metrics"""
        if self.count == 0:
            return {}

        # Compute mDice (global Dice coefficient)
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
    Compute boundary evaluation metrics
    pred: prediction map (numpy array, 0-1)
    gt: ground-truth map (numpy array, 0-1)
    """
    # Binarization
    pred_binary = (pred > threshold).astype(np.uint8) * 255
    gt_binary = (gt > 0.5).astype(np.uint8) * 255

    # Extract boundaries (morphological gradient)
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    gt_boundary = cv2.morphologyEx(gt_binary, cv2.MORPH_GRADIENT, kernel)

    # Normalize
    pred_boundary = pred_boundary.astype(np.float32) / 255.0
    gt_boundary = gt_boundary.astype(np.float32) / 255.0

    # Compute metrics
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
# Test Functions
# ============================================================================

def test_with_tta(model, image, opt):
    """
    Test-Time Augmentation (TTA)
    Use multi-scale + horizontal flip to improve test performance

    Args:
        model: model
        image: input image [1, 3, H, W]
        opt: options

    Returns:
        final_pred: fused prediction result
        edge_map: edge map (if available)
    """
    predictions = []
    edge_maps = []
    original_size = image.shape[2:]

    # TTA configuration
    if opt.use_tta:
        scales = opt.tta_scales # e.g. [0.75, 1.0, 1.25]
        use_flip = opt.tta_flip
    else:
        scales = [1.0]
        use_flip = False

    for scale in scales:
        # Scale image
        if scale != 1.0:
            h, w = int(original_size[0] * scale), int(original_size[1] * scale)
            # Ensure multiples of 32
            h = int(np.round(h / 32) * 32)
            w = int(np.round(w / 32) * 32)
            img_scaled = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        else:
            img_scaled = image

        # Original-image prediction
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

        # Restore to original size
        pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
        predictions.append(pred)

        if edge is not None:
            edge = F.interpolate(edge, size=original_size, mode='bilinear', align_corners=False)
            edge_maps.append(edge)

        # Horizontal flip augmentation
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

            # Flip back
            pred_flip = torch.flip(pred_flip, dims=[3])
            pred_flip = F.interpolate(pred_flip, size=original_size, mode='bilinear', align_corners=False)
            predictions.append(pred_flip)

            if edge_flip is not None:
                edge_flip = torch.flip(edge_flip, dims=[3])
                edge_flip = F.interpolate(edge_flip, size=original_size, mode='bilinear', align_corners=False)
                edge_maps.append(edge_flip)

    # Fuse all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    final_edge = torch.stack(edge_maps).mean(dim=0) if len(edge_maps) > 0 else None

    return final_pred, final_edge


def test_dataset(model, test_loader, save_path, opt):
    """
    Test a single dataset
    """
    model.eval()

    # Create save path
    os.makedirs(save_path, exist_ok=True)

    # Evaluator
    evaluator = Evaluator()
    boundary_metrics = {'Boundary_Dice': 0, 'Boundary_Precision': 0,
                       'Boundary_Recall': 0, 'Boundary_F1': 0}

    tta_info = ""
    if opt.use_tta:
        tta_info = f" (TTA: scales={opt.tta_scales}, flip={opt.tta_flip})"

    print(f"\n{'='*80}")
    print(f"Testing dataset: {test_loader.size} images{tta_info}")
    print(f"Save path: {save_path}")
    print(f"{'='*80}\n")

    with torch.no_grad():
        pbar = tqdm(range(test_loader.size), desc="Testing")

        for i in pbar:
            # Load data
            image, gt, name = test_loader.load_data()

            # Ground-truth processing
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            # Move to GPU
            image = image.cuda()

            # Inference (TTA or standard)
            if opt.use_tta:
                # Use TTA
                res, edge_map = test_with_tta(model, image, opt)
            else:
                # Standard inference (original method)
                if opt.decoder_type in ['innovative', 'ultralight']:
                    # Innovative decoder: supports contrastive learning outputs
                    outputs = model(image, return_contrast_outputs=True)
                    res = outputs['pred_final']
                    edge_map = outputs.get('edge_map', None)

                    # Optional: save intermediate results and edge maps
                    if opt.save_intermediate:
                        # Save edge map
                        if 'edge_map' in outputs and outputs['edge_map'] is not None:
                            edge_map_vis = outputs['edge_map']
                            edge_map_vis = F.interpolate(edge_map_vis, size=gt.shape,
                                               mode='bilinear', align_corners=False)
                            edge_map_vis = edge_map_vis.sigmoid().data.cpu().numpy().squeeze()
                            edge_map_vis = (edge_map_vis - edge_map_vis.min()) / (edge_map_vis.max() - edge_map_vis.min() + 1e-8)
                            edge_save_path = save_path.replace('/final/', '/edge/')
                            os.makedirs(edge_save_path, exist_ok=True)
                            cv2.imwrite(edge_save_path + name, edge_map_vis * 255)

                        # Save stage-wise predictions
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
                    # Simplified decoder: supports intermediate results
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
                    # Original decoder
                    _, _, _, res = model(image)
                    edge_map = None

            # Post-processing
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # Save prediction results
            if opt.save_results:
                # Save grayscale map (0-255)
                cv2.imwrite(save_path + name, res * 255)

                # Optionally save binarized results
                if opt.save_binary:
                    binary_res = (res > opt.threshold).astype(np.uint8) * 255
                    binary_save_path = save_path.replace('/final/', '/binary/')
                    os.makedirs(binary_save_path, exist_ok=True)
                    cv2.imwrite(binary_save_path + name, binary_res)

            # Update evaluation metrics
            evaluator.update(res, gt, threshold=opt.threshold)

            # Boundary metrics (compute every 10 images to save time)
            if i % 10 == 0 or i == test_loader.size - 1:
                boundary = calculate_boundary_metrics(res, gt, threshold=opt.threshold)
                for key in boundary_metrics:
                    boundary_metrics[key] += boundary[key]

            # Update progress bar
            metrics = evaluator.get_metrics()
            pbar.set_postfix({
                'mDice': f"{metrics['mDice']:.4f}",
                'Dice': f"{metrics['Dice']:.4f}",
                'MAE': f"{metrics['MAE']:.4f}"
            })

    # Compute final metrics
    final_metrics = evaluator.get_metrics()

    # Average boundary metrics
    boundary_count = (test_loader.size // 10) + 1
    for key in boundary_metrics:
        boundary_metrics[key] /= boundary_count

    # Merge metrics
    final_metrics.update(boundary_metrics)

    return final_metrics


# ============================================================================
# Main Test Flow
# ============================================================================

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--testsize', type=int, default=352, help='test image size')
    parser.add_argument('--threshold', type=float, default=0.5, help='binarization threshold')

    # Model parameters
    parser.add_argument('--pth_path', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/checkpoint/innovative_dual_stream_cfanet_fixed_singlesscaleOptimizedCFANet_best0.9599.pth',
                       help='model weight path')
    parser.add_argument('--channel', type=int, default=64, help='decoder base channels')
    parser.add_argument('--mamba_dim', type=int, default=96, help='Mamba embedding dimension')
    parser.add_argument('--decoder_type', type=str, default='innovative',
                       choices=['innovative', 'ultralight', 'simplified', 'original'],
                       help='decoder type: innovative(recommended), ultralight, simplified, original')
    parser.add_argument('--num_region_queries', type=int, default=100,
                       help='number of region queries (for innovative/ultralight only)')
    parser.add_argument('--num_boundary_queries', type=int, default=25,
                       help='number of boundary queries (for innovative/ultralight only)')

    # Data paths (AutoDL adapted)
    parser.add_argument('--test_root', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/TestDataset/TestDataset/',
                       help='test dataset root path')
    parser.add_argument('--save_root', type=str,
                       default='/root/autodl-tmp/CFANet-improved/CFANet-main-improve/results/best_model_test/',
                       help='result save root path')

    # Test options
    parser.add_argument('--save_results', type=bool, default=True,
                       help='save prediction results')
    parser.add_argument('--save_binary', type=bool, default=True,
                       help='save binarized results (0 or 255)')
    parser.add_argument('--save_intermediate', type=bool, default=False,
                       help='save intermediate stage results (progressive decoder only)')
    parser.add_argument('--datasets', type=str,
                       default='CVC-300',
                       help='test dataset list (comma-separated)')

    # TTA options
    parser.add_argument('--use_tta', type=bool, default=False,
                       help='use Test-Time Augmentation (TTA) for test enhancement')
    parser.add_argument('--tta_scales', type=str, default='0.75,1.0,1.25',
                       help='TTA multi-scale list (comma-separated), recommended: 0.75,1.0,1.25')
    parser.add_argument('--tta_flip', type=bool, default=True,
                       help='whether TTA uses horizontal flip')

    opt = parser.parse_args()

    # Parse TTA scales
    if opt.use_tta:
        opt.tta_scales = [float(s.strip()) for s in opt.tta_scales.split(',')]
    else:
        opt.tta_scales = [1.0]

    print("=" * 80)
    print("Optimized Dual-Branch CFANet Test Program")
    print("=" * 80)
    print(f"Configuration:")
    print(f"• Model path: {opt.pth_path}")
    decoder_type_names = {
        'innovative': 'Innovative decoder (Query + Contrastive + Dual-stream Boundary)',
        'ultralight': 'Ultralight decoder',
        'simplified': 'Simplified progressive decoder',
        'original': 'Original CFANet decoder'
    }
    print(f"• Decoder type: {decoder_type_names.get(opt.decoder_type, opt.decoder_type)}")
    if opt.decoder_type in ['innovative', 'ultralight']:
        print(f"• Query config: {opt.num_region_queries} region + {opt.num_boundary_queries} boundary")
    print(f"• Test size: {opt.testsize}")
    print(f"• Binarization threshold: {opt.threshold}")
    print(f"• Save results: {opt.save_results}")
    print(f"• Save binary results: {opt.save_binary}")
    print(f"• Save intermediate results: {opt.save_intermediate}")

    # TTA info
    if opt.use_tta:
        print(f"• TTA: Enabled")
        print(f"- Multi-scale: {opt.tta_scales}")
        print(f"- Horizontal flip: {opt.tta_flip}")
        num_augments = len(opt.tta_scales) * (2 if opt.tta_flip else 1)
        print(f"- Augmentations: {num_augments}x (predict {num_augments} times per image then fuse)")
    else:
        print(f"• TTA: Disabled (enable with --use_tta True)")

    print("=" * 80)

    # Create model
    print("\nCreating model...")
    model = create_optimized_dual_branch_cfanet(
        channel=opt.channel,
        mamba_dim=opt.mamba_dim,
        auto_download_weights=False,
        decoder_type=opt.decoder_type,
        num_region_queries=opt.num_region_queries,
        num_boundary_queries=opt.num_boundary_queries
    ).cuda()

    # Load weights
    print(f"Loading model weights: {opt.pth_path}")
    if not os.path.exists(opt.pth_path):
        print(f"Error: model file not found: {opt.pth_path}")
        return

    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params/1e6:.2f}M parameters")

    # Test dataset list
    dataset_list = opt.datasets.split(',')

    # Store results for all datasets
    all_results = {}

    # Iterate over test datasets
    for dataset_name in dataset_list:
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*80}")

        # Data path
        data_path = os.path.join(opt.test_root, dataset_name)
        if not os.path.exists(data_path):
            print(f"Warning: dataset not found: {data_path}")
            continue

        image_root = os.path.join(data_path, 'images/')
        gt_root = os.path.join(data_path, 'masks/')

        # Result save path
        save_path = os.path.join(opt.save_root, dataset_name, 'final/')

        # Load test set
        try:
            test_loader = TestDataLoader(image_root, gt_root, opt.testsize)
        except Exception as e:
            print(f"Error: failed to load dataset: {e}")
            continue

        # Test
        try:
            metrics = test_dataset(model, test_loader, save_path, opt)
            all_results[dataset_name] = metrics

            # Print results
            print(f"\n{'='*80}")
            print(f"{dataset_name} test results:")
            print(f"{'='*80}")
            print(f"• mDice: {metrics['mDice']:.4f} (global Dice)")
            print(f"• Dice: {metrics['Dice']:.4f} (mean Dice)")
            print(f"• IoU: {metrics['IoU']:.4f}")
            print(f"• MAE: {metrics['MAE']:.4f}")
            print(f"• Precision: {metrics['Precision']:.4f}")
            print(f"• Recall: {metrics['Recall']:.4f}")
            print(f"• F1: {metrics['F1']:.4f}")
            print(f"\nBoundary metrics:")
            print(f"• Boundary Dice: {metrics['Boundary_Dice']:.4f}")
            print(f"• Boundary F1: {metrics['Boundary_F1']:.4f}")
            print(f"{'='*80}")

        except Exception as e:
            print(f"Error: test failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summarize average results across all datasets
    if all_results:
        print(f"\n{'='*80}")
        print("Average results across all datasets")
        print(f"{'='*80}")

        # Compute averages
        avg_metrics = {}
        for key in all_results[list(all_results.keys())[0]].keys():
            avg_metrics[key] = np.mean([result[key] for result in all_results.values()])

        print(f"\nAverage metrics:")
        print(f"• mDice: {avg_metrics['mDice']:.4f} (mean of global Dice)")
        print(f"• avgDice: {avg_metrics['Dice']:.4f} (mean Dice)")
        print(f"• mIoU: {avg_metrics['IoU']:.4f}")
        print(f"• mMAE: {avg_metrics['MAE']:.4f}")
        print(f"• mPrecision: {avg_metrics['Precision']:.4f}")
        print(f"• mRecall: {avg_metrics['Recall']:.4f}")
        print(f"• mF1: {avg_metrics['F1']:.4f}")
        print(f"\nAverage boundary metrics:")
        print(f"• mBoundary Dice: {avg_metrics['Boundary_Dice']:.4f}")
        print(f"• mBoundary F1: {avg_metrics['Boundary_F1']:.4f}")
        print(f"{'='*80}")

        # Save results to file
        results_file = os.path.join(opt.save_root, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Test time: {datetime.now()}\n")
            f.write(f"Model: {opt.pth_path}\n")
            f.write("=" * 80 + "\n\n")

            # Write per-dataset results
            for dataset_name, metrics in all_results.items():
                f.write(f"{dataset_name}:\n")
                for key, value in metrics.items():
                    f.write(f" {key}: {value:.4f}\n")
                f.write("\n")

            # Write average results
            f.write("=" * 80 + "\n")
            f.write("Average results:\n")
            for key, value in avg_metrics.items():
                f.write(f" {key}: {value:.4f}\n")

        print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
