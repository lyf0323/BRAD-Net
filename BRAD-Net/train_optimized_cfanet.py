"""
Optimized Dual-Branch CFANet Training Script - AutoDL Platform Adapted

Main Features:
1. Progressive Boundary-Guided Refinement Decoder Support
2. Deep Supervision Training (6 outputs: edge + pred1-4 + pred_final)
3. Uncertainty Weighted Loss
4. Boundary Consistency Loss
5. Three-Stage Training Strategy (Optional)
6. Pretrained Weight Auto-Loading
7. AutoDL Dataset Path Adaptation
8. Focal Tversky Loss for Boundary Optimization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import logging
import platform
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Model imports
from lib.BRAD_Net import (
    create_innovative_cfanet,
    create_ultralight_cfanet,
    create_optimized_dual_branch_cfanet,
    CombinedSegmentationLoss
)

# Data loaders
from utils.dataloader import get_loader, test_dataset
from utils.trainer import adjust_lr

# Global variables
best_mae = 1.0
best_epoch = 0
best_dice = 0.0


# ============================================================================
# Utility Functions
# ============================================================================

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ============================================================================
# Loss Function Definitions
# ============================================================================

class StructureLoss(nn.Module):
    """Structure Loss: Weighted BCE + Weighted IoU (Numerically Stable Version)"""
    def __init__(self, eps=1e-7):
        super(StructureLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + self.eps)

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + self.eps)

        loss = (wbce + wiou).mean()
        loss = torch.clamp(loss, min=0.0, max=10.0)
        return loss


class DiceLoss(nn.Module):
    """Dice Loss (Numerically Stable Version)"""
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        loss = 1 - dice.mean()
        loss = torch.clamp(loss, min=0.0, max=1.0)
        return loss


class IoULoss(nn.Module):
    """IoU Loss (Numerically Stable Version)"""
    def __init__(self, smooth=1.0, eps=1e-7):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth + self.eps)
        loss = 1 - iou.mean()
        loss = torch.clamp(loss, min=0.0, max=1.0)
        return loss


class BoundaryConsistencyLoss(nn.Module):
    """Boundary Consistency Loss (Numerically Stable Version)"""
    def __init__(self, eps=1e-7):
        super(BoundaryConsistencyLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, edge_gt):
        edge_gt = edge_gt.float()
        if edge_gt.numel() > 0 and edge_gt.max() > 1.0:
            edge_gt = edge_gt / 255.0

        pred_sigmoid = torch.sigmoid(pred)

        sobel_h = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_w = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

        grad_h = F.conv2d(pred_sigmoid, sobel_h, padding=1)
        grad_w = F.conv2d(pred_sigmoid, sobel_w, padding=1)

        pred_boundary = torch.sqrt(grad_h**2 + grad_w**2 + self.eps)

        loss = F.mse_loss(pred_boundary, edge_gt)
        loss = torch.clamp(loss, min=0.0, max=1.0)
        return loss


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty Weighted Loss"""
    def __init__(self, base_loss):
        super(UncertaintyWeightedLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred, target, uncertainty):
        if isinstance(self.base_loss, StructureLoss):
            base = self.base_loss(pred, target)
            return base
        else:
            loss_map = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

            if uncertainty is not None:
                if uncertainty.shape[2:] != loss_map.shape[2:]:
                    uncertainty = F.interpolate(uncertainty, size=loss_map.shape[2:],
                                               mode='bilinear', align_corners=False)
                weighted_loss = loss_map * (1.0 + 2.0 * uncertainty)
            else:
                weighted_loss = loss_map

            return weighted_loss.mean()


# ============================================================================
# Specialized Boundary Loss Functions
# ============================================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss - For Class Imbalance

    Paper: "Tversky loss function for image segmentation using 3D fully convolutional deep networks"

    Parameters:
        alpha: False Positive weight (recommend 0.3)
        beta: False Negative weight (recommend 0.7)
        When alpha=beta=0.5, degenerates to Dice Loss

    Boundary detection config: alpha=0.3, beta=0.7 (focus on recall, reduce boundary omission)
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))

        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth + self.eps)

        loss = 1 - tversky_index.mean()
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Best Choice for Boundary Detection

    Paper: "A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation"

    Advantages:
    1. Handles class imbalance (via Tversky)
    2. Enhances hard sample learning (via Focal modulation)
    3. Particularly effective for boundary and small target detection

    Parameters:
        alpha: FP weight, recommend 0.3 (boundary detection)
        beta: FN weight, recommend 0.7 (boundary detection)
        gamma: focal modulation parameter, recommend 1.0-2.0
               - larger gamma = more focus on hard samples
               - recommend 1.5 for boundary detection
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0, eps=1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))

        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth + self.eps)

        focal_tversky = torch.pow((1 - tversky_index), self.gamma)

        loss = focal_tversky.mean()
        loss = torch.clamp(loss, min=0.0, max=2.0)

        return loss


class BoundaryIoULoss(nn.Module):
    """
    Boundary IoU Loss - Focus on Boundary Regions Only

    Idea: Boundaries are key to segmentation, should receive more attention
    Extract boundary regions via morphological operations, calculate IoU only in boundary areas

    Parameters:
        boundary_width: boundary width in pixels, recommend 3-7
    """
    def __init__(self, boundary_width=5, smooth=1.0, eps=1e-7):
        super(BoundaryIoULoss, self).__init__()
        self.boundary_width = boundary_width
        self.smooth = smooth
        self.eps = eps

    def get_boundary_region(self, mask):
        """Extract boundary region: dilation - erosion = boundary band"""
        B, C, H, W = mask.shape

        kernel_size = self.boundary_width
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size**2)

        dilated = F.conv2d(mask, kernel, padding=kernel_size//2, groups=1)
        dilated = (dilated > 0.5).float()

        eroded = F.conv2d(1 - mask, kernel, padding=kernel_size//2, groups=1)
        eroded = (eroded > 0.5).float()
        eroded = 1 - eroded

        boundary = (dilated - eroded).clamp(0, 1)

        return boundary

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        boundary_mask = self.get_boundary_region(target)

        pred_boundary = pred * boundary_mask
        target_boundary = target * boundary_mask

        intersection = (pred_boundary * target_boundary).sum(dim=(2, 3))
        union = pred_boundary.sum(dim=(2, 3)) + target_boundary.sum(dim=(2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth + self.eps)

        loss = 1 - iou.mean()
        loss = torch.clamp(loss, min=0.0, max=1.0)

        return loss


class EnhancedBoundaryLoss(nn.Module):
    """
    Enhanced Boundary Loss - Combination of Multiple Losses

    Combination Strategy:
    1. BCE: Basic pixel-level supervision (30%)
    2. Focal Tversky: Handle imbalance + hard samples (40%) - Core
    3. Boundary IoU: Enhance boundary region accuracy (30%)

    Use Case: Tasks requiring high-precision boundary detection
    """
    def __init__(self,
                 focal_alpha=0.3,
                 focal_beta=0.7,
                 focal_gamma=1.5,
                 boundary_width=5):
        super(EnhancedBoundaryLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = FocalTverskyLoss(alpha=focal_alpha, beta=focal_beta, gamma=focal_gamma)
        self.boundary_iou = BoundaryIoULoss(boundary_width=boundary_width)

        self.w_bce = 0.3
        self.w_focal_tversky = 0.4
        self.w_boundary_iou = 0.3

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_focal_tversky = self.focal_tversky(pred, target)
        loss_boundary_iou = self.boundary_iou(pred, target)

        total_loss = (
            self.w_bce * loss_bce +
            self.w_focal_tversky * loss_focal_tversky +
            self.w_boundary_iou * loss_boundary_iou
        )

        return total_loss


class SimplifiedProgressiveLoss(nn.Module):
    """
    Simplified Progressive Loss (Recommended)

    Core Improvements:
    - Remove uncertainty weighting (simplify training)
    - Use clean loss combination: BCE + Dice + Structure
    - Use Focal Tversky Loss for boundary optimization
    - Balanced weight allocation (reference original CFANet)

    Advantages:
    - More stable training
    - Faster convergence
    - More accurate boundary detection (via Focal Tversky Loss)
    - Easier to tune
    """
    def __init__(self, edge_loss_type='focal_tversky'):
        super(SimplifiedProgressiveLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.structure = StructureLoss()
        self.dice = DiceLoss()

        if edge_loss_type == 'focal_tversky':
            self.edge_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
            print("Boundary Loss: Focal Tversky Loss (alpha=0.3, beta=0.7, gamma=1.5)")
        elif edge_loss_type == 'tversky':
            self.edge_loss = TverskyLoss(alpha=0.3, beta=0.7)
            print("Boundary Loss: Tversky Loss (alpha=0.3, beta=0.7)")
        elif edge_loss_type == 'enhanced':
            self.edge_loss = EnhancedBoundaryLoss(focal_alpha=0.3, focal_beta=0.7, focal_gamma=1.5, boundary_width=5)
            print("Boundary Loss: Enhanced Boundary Loss (combined)")
        else:
            self.edge_loss = nn.BCEWithLogitsLoss()
            print("Boundary Loss: BCE Loss (original)")

        self.edge_loss_type = edge_loss_type

    def forward(self, outputs, gt_mask, gt_edge):
        total_loss = 0.0
        losses = {}

        output_size = outputs['edge'].shape[2:]
        if gt_edge.shape[2:] != output_size:
            gt_edge = F.interpolate(gt_edge, size=output_size,
                                   mode='bilinear', align_corners=False)
        if gt_mask.shape[2:] != output_size:
            gt_mask = F.interpolate(gt_mask, size=output_size,
                                   mode='bilinear', align_corners=False)

        loss_edge = self.edge_loss(outputs['edge'], gt_edge) * 1.0
        losses['edge'] = loss_edge.item()
        total_loss += loss_edge

        stage_weights = [0.5, 0.6, 0.7, 0.8]

        for i, (pred_key, weight) in enumerate(zip(['pred1', 'pred2', 'pred3', 'pred4'],
                                                     stage_weights)):
            pred = outputs[pred_key]

            bce = self.bce(pred, gt_mask)
            dice = self.dice(pred, gt_mask)

            stage_loss = (bce + dice) * weight
            losses[f'stage{i+1}'] = stage_loss.item()
            total_loss += stage_loss

        final_bce = self.bce(outputs['pred_final'], gt_mask)
        final_dice = self.dice(outputs['pred_final'], gt_mask)
        final_struct = self.structure(outputs['pred_final'], gt_mask)

        final_loss = (final_bce + final_dice + final_struct) * 2.0
        losses['final'] = final_loss.item()
        total_loss += final_loss

        losses['total'] = total_loss.item()
        return total_loss, losses


class ProgressiveRefinementLoss(nn.Module):
    """
    Progressive Refinement Decoder Complete Loss Function (Original - Keep for Comparison)

    Update: Integrate Focal Tversky Loss for boundary optimization
    """
    def __init__(self, edge_loss_type='focal_tversky'):
        super(ProgressiveRefinementLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.structure_loss = StructureLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryConsistencyLoss()

        if edge_loss_type == 'focal_tversky':
            self.edge_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        elif edge_loss_type == 'tversky':
            self.edge_loss = TverskyLoss(alpha=0.3, beta=0.7)
        elif edge_loss_type == 'enhanced':
            self.edge_loss = EnhancedBoundaryLoss(focal_alpha=0.3, focal_beta=0.7, focal_gamma=1.5, boundary_width=5)
        else:
            self.edge_loss = nn.BCEWithLogitsLoss()

        self.edge_loss_type = edge_loss_type

    def forward(self, outputs, gt_mask, gt_edge, use_uncertainty=True):
        total_loss = 0.0
        losses = {}

        output_size = outputs['edge'].shape[2:]
        if gt_edge.shape[2:] != output_size:
            gt_edge = F.interpolate(gt_edge, size=output_size, mode='bilinear', align_corners=False)
        if gt_mask.shape[2:] != output_size:
            gt_mask = F.interpolate(gt_mask, size=output_size, mode='bilinear', align_corners=False)

        edge_loss = self.edge_loss(outputs['edge'], gt_edge)
        losses['edge'] = edge_loss.item()
        total_loss += edge_loss * 0.8

        stage_weights = [0.4, 0.5, 0.6, 0.7]

        for i, (pred_key, weight) in enumerate(zip(['pred1', 'pred2', 'pred3', 'pred4'],
                                                     stage_weights)):
            pred = outputs[pred_key]

            bce = self.bce_loss(pred, gt_mask)
            dice = self.dice_loss(pred, gt_mask)
            struct = self.structure_loss(pred, gt_mask)

            if use_uncertainty and i < 3 and 'uncertainties' in outputs:
                if len(outputs['uncertainties']) > i:
                    uncertainty = outputs['uncertainties'][i]
                    uncertainty = F.interpolate(uncertainty, size=pred.shape[2:],
                                               mode='bilinear', align_corners=False)
                    loss_map = F.binary_cross_entropy_with_logits(pred, gt_mask, reduction='none')
                    weighted_bce = (loss_map * (1.0 + uncertainty)).mean()
                    bce = weighted_bce

            stage_loss = (bce + dice + 0.5 * struct) * weight
            losses[f'stage{i+1}'] = stage_loss.item()
            total_loss += stage_loss

        final_bce = self.bce_loss(outputs['pred_final'], gt_mask)
        final_dice = self.dice_loss(outputs['pred_final'], gt_mask)
        final_struct = self.structure_loss(outputs['pred_final'], gt_mask)
        final_iou = self.iou_loss(outputs['pred_final'], gt_mask)

        final_loss = (final_bce + final_dice + final_struct + 0.5 * final_iou) * 1.5
        losses['final'] = final_loss.item()
        total_loss += final_loss

        losses['total'] = total_loss.item()
        return total_loss, losses


# ============================================================================
# Training and Validation Functions
# ============================================================================

def train_one_epoch(train_loader, model, optimizer, epoch, opt, loss_func, total_step, scaler=None):
    """Train for one epoch with innovative decoder (Query + Contrastive)"""
    model.train()

    size_rates = [0.75, 1.0, 1.25] if opt.multi_scale else [1.0]

    epoch_losses = {'total': [], 'bce': [], 'dice': [], 'boundary': [], 'contrastive': []}

    pbar = tqdm(enumerate(train_loader), total=total_step,
                desc=f'Epoch {epoch}/{opt.epoch}')

    use_amp = opt.use_amp and scaler is not None
    use_contrast = opt.decoder_type in ['innovative', 'ultralight']

    for step, data_pack in pbar:
        images, gts, egs = data_pack

        if torch.isnan(images).any() or torch.isnan(gts).any() or torch.isnan(egs).any():
            print(f"Warning: NaN detected in input data at step {step}, skipping")
            logging.warning(f'NaN detected in input data at step {step}, skipping')
            continue

        images = images.cuda()
        gts = gts.cuda()
        egs = egs.cuda()

        images = torch.clamp(images, min=-10, max=10)

        optimizer.zero_grad()

        accumulated_loss = 0.0
        for idx, rate in enumerate(size_rates):
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1.0:
                images_scaled = F.interpolate(images, size=(trainsize, trainsize),
                                             mode='bilinear', align_corners=True)
                gts_scaled = F.interpolate(gts, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True)
                egs_scaled = F.interpolate(egs, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True)
            else:
                images_scaled = images
                gts_scaled = gts
                egs_scaled = egs

            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    if use_contrast:
                        # Innovative decoder (Query + contrastive learning)
                        outputs = model(images_scaled, return_contrast_outputs=True)
                        loss_total, losses = loss_func(
                            outputs, gts_scaled,
                            contrast_outputs=outputs['contrast_outputs'],
                            boundary_gt=egs_scaled
                        )
                    else:
                        # Original or simplified decoder
                        edge_out, sal_out1, sal_out2, sal_out3 = model(images_scaled)
                        structure_loss = StructureLoss().cuda()
                        loss_edge = F.binary_cross_entropy_with_logits(edge_out, egs_scaled)
                        loss_sal1 = structure_loss(sal_out1, gts_scaled)
                        loss_sal2 = structure_loss(sal_out2, gts_scaled)
                        loss_sal3 = structure_loss(sal_out3, gts_scaled)
                        loss_total = loss_edge + loss_sal1 + loss_sal2 + loss_sal3
                        losses = {
                            'total': loss_total.item(),
                            'bce': loss_sal3.item(),
                            'dice': 0,
                            'boundary': loss_edge.item(),
                            'contrastive': 0
                        }
            else:
                if use_contrast:
                    # Innovative decoder (Query + contrastive learning)
                    outputs = model(images_scaled, return_contrast_outputs=True)
                    loss_total, losses = loss_func(
                        outputs, gts_scaled,
                        contrast_outputs=outputs['contrast_outputs'],
                        boundary_gt=egs_scaled
                    )
                else:
                    # Original or simplified decoder
                    edge_out, sal_out1, sal_out2, sal_out3 = model(images_scaled)
                    structure_loss = StructureLoss().cuda()
                    loss_edge = F.binary_cross_entropy_with_logits(edge_out, egs_scaled)
                    loss_sal1 = structure_loss(sal_out1, gts_scaled)
                    loss_sal2 = structure_loss(sal_out2, gts_scaled)
                    loss_sal3 = structure_loss(sal_out3, gts_scaled)
                    loss_total = loss_edge + loss_sal1 + loss_sal2 + loss_sal3
                    losses = {
                        'total': loss_total.item(),
                        'bce': loss_sal3.item(),
                        'dice': 0,
                        'boundary': loss_edge.item(),
                        'contrastive': 0
                    }

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"Warning: Epoch {epoch}, Step {step}, Rate {rate:.2f} - NaN/Inf loss")
                print(f"Loss details: total={loss_total.item()}, edge={losses.get('edge', 0):.4f}, final={losses.get('final', 0):.4f}")
                logging.warning(f'NaN/Inf loss detected at epoch {epoch}, step {step}, rate {rate}')
                optimizer.zero_grad()
                continue

            if use_amp:
                scaler.scale(loss_total / len(size_rates)).backward()
            else:
                (loss_total / len(size_rates)).backward()
            accumulated_loss += loss_total.item()

        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item()**2
        total_grad_norm = total_grad_norm**0.5

        if total_grad_norm > 100:
            print(f"Warning: Large gradient norm = {total_grad_norm:.2f} (Epoch {epoch}, Step {step})")
            logging.warning(f'Large gradient norm {total_grad_norm:.2f} at epoch {epoch}, step {step}')

        if opt.clip_grad:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        avg_loss = accumulated_loss / len(size_rates)
        epoch_losses['total'].append(avg_loss)
        epoch_losses['bce'].append(losses.get('bce', 0))
        epoch_losses['dice'].append(losses.get('dice', 0))
        epoch_losses['boundary'].append(losses.get('boundary', 0))
        epoch_losses['contrastive'].append(losses.get('contrastive', 0))

        if step % 10 == 0:
            if use_contrast:
                pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'BCE': f"{losses.get('bce', 0):.4f}",
                    'Dice': f"{losses.get('dice', 0):.4f}",
                    'Contrast': f"{losses.get('contrastive', 0):.4f}"
                })
            else:
                pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'BCE': f"{losses.get('bce', 0):.4f}",
                    'Boundary': f"{losses.get('boundary', 0):.4f}"
                })

    avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

    return avg_losses


def validate_single_dataset(val_loader, model, opt):
    """Validate on a single dataset"""
    model.eval()
    mae_sum = 0.0
    dice_sum = 0.0
    count = 0

    use_contrast = opt.decoder_type in ['innovative', 'ultralight']

    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name = val_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            if use_contrast:
                # Innovative decoder (Query + contrastive learning)
                outputs = model(image, return_contrast_outputs=False)
                _, _, res, _ = outputs # edge_map, pred4, pred_final, pred_final
            else:
                # Original or simplified decoder
                _, _, _, res = model(image)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae = np.mean(np.abs(res - gt))
            mae_sum += mae

            intersection = np.sum(res * gt)
            dice = (2.0 * intersection) / (np.sum(res) + np.sum(gt) + 1e-8)
            dice_sum += dice

            count += 1

    avg_mae = mae_sum / count
    avg_dice = dice_sum / count

    return avg_mae, avg_dice


def validate_multi_datasets(model, opt, epoch):
    """Multi-dataset validation"""
    global best_mae, best_epoch, best_dice

    dataset_names = [d.strip() for d in opt.val_datasets.split(',')]
    all_results = {}

    print(f"\n{'='*60}")
    print(f"Validation - Epoch {epoch} - Multi-Dataset")
    print(f"{'='*60}")

    total_mae = 0.0
    total_dice = 0.0

    for dataset_name in dataset_names:
        val_path = f'/root/autodl-tmp/TestDataset/TestDataset/{dataset_name}/'
        if not os.path.exists(val_path):
            val_path = f'./TestDataset/TestDataset/{dataset_name}/'
        if not os.path.exists(val_path):
            print(f"Warning: Dataset {dataset_name} not found, skipping")
            continue

        val_image_root = f'{val_path}/images/'
        val_gt_root = f'{val_path}/masks/'
        val_loader = test_dataset(val_image_root, val_gt_root, opt.trainsize)

        avg_mae, avg_dice = validate_single_dataset(val_loader, model, opt)

        all_results[dataset_name] = {'mae': avg_mae, 'dice': avg_dice}
        total_mae += avg_mae
        total_dice += avg_dice

        print(f"{dataset_name:20s} -> MAE: {avg_mae:.4f}, Dice: {avg_dice:.4f}")
        logging.info(f'[Val-{dataset_name}] Epoch: {epoch}, MAE: {avg_mae:.4f}, Dice: {avg_dice:.4f}')

    num_datasets = len(all_results)
    if num_datasets > 0:
        avg_mae = total_mae / num_datasets
        avg_dice = total_dice / num_datasets

        print(f"{'='*60}")
        print(f"Average Performance -> MAE: {avg_mae:.4f}, Dice: {avg_dice:.4f}")
        print(f"{'='*60}\n")

        is_best = False
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_epoch = epoch
            best_dice = avg_dice
            is_best = True

        return is_best, all_results

    return False, {}


# ============================================================================
# Main Training Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training image size')
    parser.add_argument('--clip_grad', type=float, default=0.5, help='gradient clipping threshold')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='learning rate decay epoch')
    parser.add_argument('--save_epoch', type=int, default=5, help='model save interval')

    parser.add_argument('--train_path', type=str,
                       default='/root/autodl-tmp/TrainDatasetEdges/TrainDatasetEdges/',
                       help='training dataset path')
    parser.add_argument('--val_path', type=str,
                       default='/root/autodl-tmp/TestDataset/TestDataset/CVC-300/',
                       help='validation dataset path')
    parser.add_argument('--save_path', type=str,
                       default='./checkpoint/optimized_cfanet/',
                       help='model save path')

    parser.add_argument('--channel', type=int, default=64, help='decoder base channels')
    parser.add_argument('--mamba_dim', type=int, default=96, help='Mamba embedding dimension')
    parser.add_argument('--decoder_type', type=str, default='innovative',
                       choices=['innovative', 'ultralight', 'simplified', 'original'],
                       help='decoder type: innovative(recommended, Query+contrastive), ultralight(tiny datasets), simplified(no Query), original(baseline)')
    parser.add_argument('--num_region_queries', type=int, default=100,
                       help='number of region queries (for innovative/ultralight decoder)')
    parser.add_argument('--num_boundary_queries', type=int, default=25,
                       help='number of boundary queries (for innovative/ultralight decoder)')
    parser.add_argument('--use_uncertainty', type=str2bool, default=True,
                       help='use uncertainty weighting')
    parser.add_argument('--multi_scale', type=str2bool, default=True,
                       help='multi-scale training')

    # Contrastive learning loss weights
    parser.add_argument('--weight_bce', type=float, default=1.0, help='BCE loss weight')
    parser.add_argument('--weight_dice', type=float, default=1.0, help='Dice loss weight')
    parser.add_argument('--weight_boundary', type=float, default=0.5, help='boundary loss weight')
    parser.add_argument('--weight_contrastive', type=float, default=0.2,
                       help='contrastive learning loss weight (0.2 for innovative, 0 to disable)')

    parser.add_argument('--res2net_path', type=str,
                       default='./lib/res2net50_v1b_26w_4s-3cf99910.pth',
                       help='Res2Net pretrained weight path')
    parser.add_argument('--vit_path', type=str, default=None,
                       help='ViT pretrained weight path (optional)')
    parser.add_argument('--auto_download_vit', type=str2bool, default=False,
                       help='auto download ViT weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='resume training from checkpoint')

    parser.add_argument('--use_cosine_lr', type=str2bool, default=True,
                       help='use cosine annealing learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='learning rate warmup epochs')
    parser.add_argument('--use_amp', type=str2bool, default=False,
                       help='use mixed precision training (save memory)')
    parser.add_argument('--use_ema', type=str2bool, default=False,
                       help='use EMA model (improve generalization)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='early stopping patience (stop after N epochs without improvement)')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                       help='use Tensorboard to log training process')
    parser.add_argument('--val_datasets', type=str,
                       default='CVC-300',
                       help='validation datasets (comma-separated, e.g. CVC-300,Kvasir)')

    parser.add_argument('--edge_loss_type', type=str, default='focal_tversky',
                       choices=['bce', 'tversky', 'focal_tversky', 'enhanced'],
                       help='boundary loss type: bce(original), tversky, focal_tversky(recommended), enhanced(combined)')

    parser.add_argument('--freeze_resnet', type=str2bool, default=False,
                       help='freeze ResNet branch (only train Mamba+decoder)')
    parser.add_argument('--resnet_lr_scale', type=float, default=0.1,
                       help='ResNet learning rate scale factor (0.0-1.0, used when not frozen)')

    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=opt.save_path + '/train.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
    logging.info("=" * 80)
    logging.info(f"Training Started at {datetime.now()}")
    logging.info(f"Configuration: {opt}")

    print("=" * 80)
    print("Optimized Dual-Branch CFANet Training - Query-Guided + Contrastive Learning")
    print("=" * 80)
    print(f"Configuration:")
    print(f"Decoder Type: {opt.decoder_type.upper()}")
    decoder_desc = {
        'innovative': 'Query(100+25) + Contrastive Learning(3 layers) + Simplified MSCA',
        'ultralight': 'Query(50+12) + Contrastive Learning(2 layers) (resource-constrained)',
        'simplified': 'Simplified MSCA + Lightweight BGRM (no-Query baseline)',
        'original': 'Original CFANet (baseline)'
    }
    print(f"{decoder_desc.get(opt.decoder_type, 'Unknown')}")
    if opt.decoder_type in ['innovative', 'ultralight']:
        print(f"Query Config: {opt.num_region_queries} region + {opt.num_boundary_queries} boundary")
        print(f"Contrastive Learning Weight: {opt.weight_contrastive}")
    print(f"Batch Size: {opt.batchsize}")
    print(f"Learning Rate: {opt.lr}")
    print(f"Training Epochs: {opt.epoch}")
    print(f"ResNet Strategy: {'Frozen (Use Pretrained)' if opt.freeze_resnet else f'Fine-tune ({opt.resnet_lr_scale}x LR)'}")
    print(f"Multi-scale Training: {opt.multi_scale}")
    print(f"Save Path: {opt.save_path}")
    print(f"Query Adaptive Aggregation + Contrastive Boundary-Region Decoupling + Simplified MSCA")
    print("=" * 80)

    print("\nLoading Dataset...")
    image_root = f'{opt.train_path}/images/'
    gt_root = f'{opt.train_path}/masks/'
    edge_root = f'{opt.train_path}/edges/'

    num_workers = 0 if platform.system() == 'Windows' else 4

    train_loader = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=num_workers)
    total_step = len(train_loader)

    print(f"Training set loaded: {total_step} batches")
    print(f"Validation datasets configured: {opt.val_datasets}")

    print("\nCreating Model...")

    # Create model based on decoder_type
    if opt.decoder_type == 'innovative':
        print(f"Creating optimized innovative decoder (Query {opt.num_region_queries}+{opt.num_boundary_queries} + Contrastive)")
        model = create_innovative_cfanet(
            channel=opt.channel,
            mamba_dim=opt.mamba_dim,
            num_region_queries=opt.num_region_queries,
            num_boundary_queries=opt.num_boundary_queries
        ).cuda()
    elif opt.decoder_type == 'ultralight':
        print(f"Creating ultralight decoder (Query {opt.num_region_queries}+{opt.num_boundary_queries})")
        model = create_ultralight_cfanet(
            channel=opt.channel,
            mamba_dim=opt.mamba_dim
        ).cuda()
    else:
        print(f"Creating {opt.decoder_type} decoder")
        model = create_optimized_dual_branch_cfanet(
            channel=opt.channel,
            mamba_dim=opt.mamba_dim,
            decoder_type=opt.decoder_type,
            num_region_queries=opt.num_region_queries,
            num_boundary_queries=opt.num_boundary_queries
        ).cuda()

    if opt.resume:
        print(f"Loading Checkpoint: {opt.resume}")
        model.load_state_dict(torch.load(opt.resume))
    else:
        print("Loading Pretrained Weights...")
        model.load_pretrained_weights(
            res2net_path=opt.res2net_path,
            vit_path=opt.vit_path
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Created: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable params")

    if opt.freeze_resnet:
        model.freeze_resnet_branch()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ResNet Branch Frozen, Only Training Mamba Branch and Decoder")
        print(f"Trainable Params: {trainable_after_freeze/1e6:.2f}M ({trainable_after_freeze/total_params*100:.1f}%)")
    elif opt.resnet_lr_scale < 1.0:
        print(f"ResNet Branch Training with {opt.resnet_lr_scale}x Learning Rate")

    # Create optimizer
    if not opt.freeze_resnet and opt.resnet_lr_scale < 1.0:
        # Differentiated learning rates
        if opt.decoder_type in ['innovative', 'ultralight', 'simplified']:
            optimizer = torch.optim.Adam([
                {'params': model.resnet_encoder.parameters(), 'lr': opt.lr * opt.resnet_lr_scale},
                {'params': model.mamba_encoder.parameters(), 'lr': opt.lr},
                {'params': model.feature_fusion.parameters(), 'lr': opt.lr},
                {'params': model.decoder.parameters(), 'lr': opt.lr},
            ])
        else:
            decoder_params = []
            for name, param in model.named_parameters():
                if 'resnet_encoder' not in name and 'mamba_encoder' not in name and 'feature_fusion' not in name:
                    decoder_params.append(param)
            optimizer = torch.optim.Adam([
                {'params': model.resnet_encoder.parameters(), 'lr': opt.lr * opt.resnet_lr_scale},
                {'params': model.mamba_encoder.parameters(), 'lr': opt.lr},
                {'params': model.feature_fusion.parameters(), 'lr': opt.lr},
                {'params': decoder_params, 'lr': opt.lr},
            ])
        print(f"Optimizer: Adam (lr={opt.lr}, ResNet_lr={opt.lr * opt.resnet_lr_scale})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        print(f"Optimizer: Adam (lr={opt.lr})")

    if opt.use_cosine_lr:
        print(f"Learning Rate Schedule: Manual Cosine Annealing (Warmup->Stable->Cosine)")
        print(f"Fix: Full manual control, avoid scheduler NaN Bug")
        print(f"Epoch 1-{opt.warmup_epochs}: Warmup (0 -> {opt.lr})")
        print(f"Epoch {opt.warmup_epochs+1}-{opt.warmup_epochs+10}: Stable ({opt.lr})")
        print(f"Epoch {opt.warmup_epochs+11}+: Cosine Annealing ({opt.lr} -> 1e-6)")
    else:
        print(f"Learning Rate Schedule: Step Decay (decay_rate={opt.decay_rate}, decay_epoch={opt.decay_epoch})")

    scaler = None
    if opt.use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print(f"Mixed Precision Training: Enabled (Save ~30% Memory)")

    print("\nInitializing Loss Functions...")
    if opt.decoder_type in ['innovative', 'ultralight']:
        # Use combined contrastive learning loss
        loss_func = CombinedSegmentationLoss(
            weight_bce=opt.weight_bce,
            weight_dice=opt.weight_dice,
            weight_boundary=opt.weight_boundary,
            weight_contrastive=opt.weight_contrastive
        ).cuda()
        print(f"Loss Function: Combined Segmentation Loss (Query + Contrastive)")
        print(f"BCE: {opt.weight_bce}, Dice: {opt.weight_dice}, "
              f"Boundary: {opt.weight_boundary}, Contrastive: {opt.weight_contrastive}")
    else:
        # Use original loss
        loss_func = StructureLoss().cuda()
        print(f"Loss Function: Structure Loss (Original CFANet)")

    writer = None
    if opt.use_tensorboard:
        writer = SummaryWriter(opt.save_path + '/logs')
        print(f"Tensorboard: {opt.save_path}/logs")
        print(f"Start: tensorboard --logdir={opt.save_path}/logs --port=6006")

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")

    for epoch in range(1, opt.epoch + 1):
        if opt.use_cosine_lr:
            import math
            if epoch <= opt.warmup_epochs:
                warmup_lr = opt.lr * epoch / opt.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                cur_lr = warmup_lr
            elif epoch <= opt.warmup_epochs + 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr
                cur_lr = opt.lr
            else:
                cosine_epoch = epoch - (opt.warmup_epochs + 10)
                T_max = opt.epoch - (opt.warmup_epochs + 10)
                eta_min = 1e-6
                cos_lr = eta_min + (opt.lr - eta_min) * (1 + math.cos(math.pi * cosine_epoch / T_max)) / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cos_lr
                cur_lr = cos_lr
        else:
            adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)
            cur_lr = optimizer.param_groups[0]['lr']

        avg_losses = train_one_epoch(train_loader, model, optimizer, epoch,
                                      opt, loss_func, total_step, scaler)

        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"Error: Parameter {name} contains NaN!")
                logging.error(f'NaN detected in parameter: {name}')
                has_nan = True
                break
            if torch.isinf(param).any():
                print(f"Error: Parameter {name} contains Inf!")
                logging.error(f'Inf detected in parameter: {name}')
                has_nan = True
                break

        if has_nan:
            print(f"Training Terminated: Model parameters contain NaN/Inf")
            print(f"Suggestion: Restart training from previous checkpoint (Epoch {epoch-1})")
            break

        if np.isnan(avg_losses["total"]) or np.isinf(avg_losses["total"]):
            print(f"Error: Epoch {epoch} average loss is NaN/Inf = {avg_losses['total']}")
            logging.error(f'Average loss is NaN/Inf at epoch {epoch}')
            print(f"Suggestion: Lower learning rate or restart from previous checkpoint")
            break

        logging.info(f'Epoch [{epoch}/{opt.epoch}], LR: {cur_lr:.6f}, '
                    f'Loss: {avg_losses["total"]:.4f}, '
                    f'BCE: {avg_losses.get("bce", 0):.4f}, '
                    f'Dice: {avg_losses.get("dice", 0):.4f}, '
                    f'Boundary: {avg_losses.get("boundary", 0):.4f}, '
                    f'Contrastive: {avg_losses.get("contrastive", 0):.4f}')

        print(f'\n[Epoch {epoch}/{opt.epoch}] LR: {cur_lr:.6f}, '
              f'Avg Loss: {avg_losses["total"]:.4f} '
              f'(BCE:{avg_losses.get("bce", 0):.3f}, Dice:{avg_losses.get("dice", 0):.3f}, '
              f'Contrast:{avg_losses.get("contrastive", 0):.3f})')

        is_best, val_results = validate_multi_datasets(model, opt, epoch)

        if writer is not None:
            writer.add_scalar('Train/loss_total', avg_losses['total'], epoch)
            writer.add_scalar('Train/loss_bce', avg_losses.get('bce', 0), epoch)
            writer.add_scalar('Train/loss_dice', avg_losses.get('dice', 0), epoch)
            writer.add_scalar('Train/loss_boundary', avg_losses.get('boundary', 0), epoch)
            writer.add_scalar('Train/loss_contrastive', avg_losses.get('contrastive', 0), epoch)
            writer.add_scalar('Train/learning_rate', cur_lr, epoch)

            for dataset_name, metrics in val_results.items():
                writer.add_scalar(f'Val/{dataset_name}_MAE', metrics['mae'], epoch)
                writer.add_scalar(f'Val/{dataset_name}_Dice', metrics['dice'], epoch)

        if epoch - best_epoch >= opt.early_stopping_patience:
            print(f"\nEarly Stopping Triggered: {opt.early_stopping_patience} epochs without improvement")
            print(f"Best Epoch: {best_epoch}, MAE: {best_mae:.4f}, Dice: {best_dice:.4f}")
            logging.info(f'Early stopping at epoch {epoch}. Best: Epoch {best_epoch}, MAE: {best_mae:.4f}')
            break

        if epoch % opt.save_epoch == 0:
            save_file = opt.save_path + f'OptimizedCFANet_epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_file)
            print(f'Model Saved: {save_file}')

        if is_best:
            save_file = opt.save_path + 'OptimizedCFANet_best.pth'
            torch.save(model.state_dict(), save_file)
            print(f'Best Model Saved: {save_file} (MAE: {best_mae:.4f}, Dice: {best_dice:.4f})')
            logging.info(f'Best model saved at epoch {epoch}, MAE: {best_mae:.4f}, Dice: {best_dice:.4f}')

    print("\n" + "=" * 80)
    print("Training Completed!")
    print(f"Best Model: Epoch {best_epoch}, MAE: {best_mae:.4f}, Dice: {best_dice:.4f}")
    print("=" * 80)

    logging.info("=" * 80)
    logging.info(f"Training Completed! Best Epoch: {best_epoch}, MAE: {best_mae:.4f}, Dice: {best_dice:.4f}")
    logging.info("=" * 80)

    if writer is not None:
        writer.close()
        print(f"Tensorboard logs saved to: {opt.save_path}/logs")


if __name__ == '__main__':
    main()
