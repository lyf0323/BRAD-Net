# BRAD-Net

**BRAD-Net** is a medical image segmentation network designed for polyp segmentation and related tasks.

The default configuration uses a **Res2Net + Mamba dual encoder** and the **`innovative`** decoder, which combines three-level contrastive learning, progressive refinement, and dual-stream boundary prediction.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Resume Training](#resume-training)
- [Key Hyperparameters](#key-hyperparameters)
- [Notes](#notes)

---

## Project Structure

```text
BRAD-Net/
├── lib/
│   ├── BRAD_Net.py                          # Model definition (encoder + decoder)
│   ├── res2net_v1b_base.py                  # Res2Net backbone
│   └── res2net50_v1b_26w_4s-3cf99910.pth    # Res2Net pretrained weights
├── utils/
│   ├── dataloader.py                        # Train / test data loading
│   ├── trainer.py                           # Training utilities (e.g., LR schedule)
│   └── eva_funcs.py                         # Evaluation metrics (S-measure / MAE, etc.)
├── train_optimized_cfanet.py                # Training entry
├── test_optimized_cfanet.py                 # Testing entry
├── run_train_autodl.sh                      # Training example script
├── run_test_innovative.sh                   # Testing example script
├── resume_training_from_best.sh / .bat      # Resume from best checkpoint
├── test_with_tta.sh / .bat                  # Test-time augmentation
├── requirements_optimized.txt               # Dependencies
├── TrainDatasetEdges/                       # Training set (images / masks / edges)
└── TestDataset/                             # Test sets (multiple datasets)
```

---

## Architecture

### Overall Pipeline

```text
Input Image
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      │
 Res2Net-50            Vision Mamba               │
 (local CNN features)  (long-range modeling)      │
    │                      │                      │
    └──────────┬───────────┘                      │
               ▼                                  │
   Adaptive Multi-Level Fusion                    │
   (level-wise adaptive fusion)                   │
               │                                  │
               ▼                                  │
   Innovative Contrastive Decoder                 │
   (progressive + contrastive + dual-stream)      │
               │                                  │
               ▼                                  │
      Segmentation Mask + Boundary Map  ◄─────────┘
```

### Encoder: Dual Encoder

All decoder variants share the same dual-encoder backbone:

| Branch | Implementation | Role |
|--------|----------------|------|
| **CNN branch** | Res2Net-50 (`Res2Net_model`) | Extract multi-scale local textures and boundary details |
| **Mamba branch** | `OptimizedVisionMamba` | Model long-range dependencies via selective scan |
| **Feature fusion** | `AdaptiveMultiLevelFusion` | Adaptively fuse ResNet / Mamba multi-scale features |

The fusion stage produces 5 feature levels (`x0`–`x4`), which are then fed into the decoder.

### Decoder: Innovative (Default / Recommended)

| Item | Value |
|------|-------|
| Factory | `create_innovative_cfanet()` |
| Training flag | `--decoder_type innovative` |

Main components used in training and testing:

| Module | Description |
|--------|-------------|
| **Three-level contrastive learning** | `ContrastiveBoundaryRegionEncoder` on **f2 / f3 / f4** to explicitly disentangle boundary and region features |
| **Progressive refinement** | **4-stage** progressive refinement with stepwise upsampling and mask refinement |
| **Dual-stream boundary** | Shallow stream preserves spatial detail; deep stream provides semantic boundaries; fused with learnable weights |

Create the model:

```python
from lib.BRAD_Net import create_innovative_cfanet

model = create_innovative_cfanet(
    channel=64,
    mamba_dim=96,
).cuda()
```

Or via the unified interface:

```python
from lib.BRAD_Net import create_optimized_dual_branch_cfanet

model = create_optimized_dual_branch_cfanet(
    decoder_type='innovative',  # default
    channel=64,
    mamba_dim=96,
).cuda()
```

### Other Decoder Variants (Optional)

| `decoder_type` | Description |
|----------------|-------------|
| `innovative` | **Default**: 3-level contrastive + progressive refinement + dual-stream boundary |
| `ultralight` | Lightweight variant (2-level contrastive) |
| `simplified` | Progressive decoder without contrastive learning |
| `original` | Original CFANet decoder (baseline) |

Training and testing scripts in this repository default to **`innovative`**.

---

## Requirements

```bash
pip install -r requirements_optimized.txt
```

Main dependencies: PyTorch, torchvision, einops, OpenCV, Pillow, tqdm, tensorboard, etc.

A CUDA environment is recommended. The default input resolution is **352×352**.

---

## Data Preparation

### Training Set

Required layout:

```text
TrainDatasetEdges/
├── images/
├── masks/
└── edges/
```

### Test Set

```text
TestDataset/
├── CVC-300/
├── CVC-ClinicDB/
├── Kvasir/
├── CVC-ColonDB/
└── ETIS-LaribPolypDB/
```

Each subset typically contains `images/` and `masks/`.

Place the Res2Net pretrained weights at:

```text
lib/res2net50_v1b_26w_4s-3cf99910.pth
```

---

## Training

### Recommended Command (`innovative` decoder)

```bash
python train_optimized_cfanet.py \
    --decoder_type innovative \
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
    --train_path ./TrainDatasetEdges/ \
    --save_path ./checkpoint/innovative_cfanet/ \
    --res2net_path ./lib/res2net50_v1b_26w_4s-3cf99910.pth \
    --val_datasets "CVC-300,CVC-ClinicDB,Kvasir" \
    --use_cosine_lr true \
    --warmup_epochs 5 \
    --multi_scale true
```

Or run the example script:

```bash
bash run_train_autodl.sh
```

Paths inside the script are AutoDL-style examples; update them for your local setup.

### Loss Composition

| Loss | Default Weight | Role |
|------|----------------|------|
| BCE | 1.0 | Pixel-wise classification |
| Dice | 1.0 | Region overlap |
| Boundary | 0.5 | Boundary consistency |
| Contrastive | 0.2 | Boundary–region contrastive decoupling |

Checkpoints are saved as:

```text
checkpoint/.../OptimizedCFANet_best.pth
```

---

## Testing

```bash
python test_optimized_cfanet.py \
    --pth_path ./checkpoint/innovative_cfanet/OptimizedCFANet_best.pth \
    --test_root ./TestDataset/ \
    --save_root ./results/innovative/ \
    --datasets "CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB" \
    --decoder_type innovative \
    --testsize 352 \
    --threshold 0.5 \
    --channel 64 \
    --mamba_dim 96 \
    --save_results True
```

Test-time augmentation (multi-scale + flip):

```bash
bash test_with_tta.sh
# Windows:
test_with_tta.bat
```

Common metrics: Dice, IoU, MAE, Precision, Recall, F1, and optional S-measure.

---

## Resume Training

```bash
bash resume_training_from_best.sh
# Windows:
resume_training_from_best.bat
```

Update `BEST_MODEL` and dataset paths in the script before running.

---

## Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `decoder_type` | `innovative` | Decoder variant |
| `channel` | 64 | Decoder base channels |
| `mamba_dim` | 96 | Mamba embedding dimension |
| `trainsize` / `testsize` | 352 | Input resolution |
| `weight_contrastive` | 0.2 | Contrastive loss weight |

---

## Notes

This implementation builds on dual-branch encoding (Res2Net + Mamba) and contrastive / progressive decoding. Related works include:

- Res2Net / CFANet-style medical segmentation
- Mamba / selective-scan sequence modeling
- Contrastive learning: Supervised Contrastive Learning
- Boundary-aware segmentation: PraNet and related methods

Please add complete citations according to your paper and experimental setup.

Follow the licenses of this project and all pretrained weights you use.
