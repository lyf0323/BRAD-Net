# BRAD-Net: Boundary-Region Adaptive Decoupling Network for Polyp Segmentation

Official implementation of **BRAD-Net** for colonoscopy polyp segmentation in medical image analysis.

---

## Table of Contents
- [1. Method Overview](#1-method-overview)
- [2. Datasets & Download Links](#2-datasets--download-links)
- [3. Code Structure](#3-code-structure)
- [4. Environment Requirements](#4-environment-requirements)
- [5. Model Weights & Releases](#5-model-weights--releases)
- [6. Training & Evaluation Workflow](#6-training--evaluation-workflow)
- [7. Citations](#7-citations)
- [8. License](#8-license)

---

## 1. Method Overview

**BRAD-Net** is designed for robust polyp segmentation under challenging conditions such as severe scale variations, diverse morphological patterns, and low polyp–background contrast.

### Key Architectural Highlights
* **Dual Encoder**: Combines Res2Net-50 (for local multi-scale edge and texture feature extraction) and Vision Mamba (for linear-complexity long-range contextual dependencies).
* **Adaptive Multi-Level Feature Fusion**: Dynamically aligns and fuses multi-scale feature hierarchies into unified representation stages (`x0`–`x4`).
* **Boundary–Region Contrastive Decoupling Decoder**: Employs query-guided feature aggregation, multi-level boundary-region contrastive supervision, progressive 4-stage refinement, and dual-stream boundary prediction.

```text
Input Image (352 x 352)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             │
  Res2Net-50 Backbone           Vision Mamba Branch                 │
  (Local Multi-Scale)         (Long-Range Context SSM)              │
        │                             │                             │
        └──────────────┬──────────────┘                             │
                       ▼                                            │
        Adaptive Multi-Level Feature Fusion                         │
                       │                                            │
                       ▼                                            │
        Boundary-Region Decoupling Decoder                          │                       
                       │                                            │
                       ▼                                            │
        Predicted Polyp Mask & Boundary Map  ◄──────────────────────┘

```

---

## 2. Datasets & Download Links

All experiments follow standard benchmarking protocols for polyp segmentation across **four publicly available datasets**:

| Dataset | Official Source / Portal | Role in Benchmark |
| --- | --- | --- |
| **Kvasir-SEG** | [Simula Portal](https://datasets.simula.no/kvasir-seg/) | Train / In-distribution Test |
| **CVC-ClinicDB** | [Grand Challenge](https://polyp.grand-challenge.org/CVCClinicDB/) | Train / In-distribution Test |
| **CVC-300** | [EndoScene Project](https://endoscopymsa.com/) | Out-of-distribution Test |
| **ETIS-LaribPolypDB** | [Grand Challenge](https://polyp.grand-challenge.org/EtisLarib/) | Generalization Test (Challenging) |

### Directory Structure

Organize the datasets as follows:

```text
TrainDatasetEdges/
├── images/          # Colonoscopy RGB frames (.png / .jpg)
├── masks/           # Binary ground-truth polyp masks (.png)
└── edges/           # Boundary/edge ground truth for boundary loss

TestDataset/
├── CVC-300/
│   ├── images/
│   └── masks/
├── CVC-ClinicDB/
│   ├── images/
│   └── masks/
├── Kvasir/
│   ├── images/
│   └── masks/
└── ETIS-LaribPolypDB/
    ├── images/
    └── masks/

```

---

## 3. Code Structure

```text
BRAD-Net/
├── README.md
├── LICENSE
├── requirements_optimized.txt
├── lib/
│   ├── BRAD_Net.py                           # Model definition (Dual encoder + decoder)
│   ├── res2net_v1b_base.py                   # Res2Net-50 backbone definition
│   └── res2net50_v1b_26w_4s-3cf99910.pth     # Pretrained backbone weights (downloaded)
├── utils/
│   ├── dataloader.py                         # Data loaders and augmentations
│   ├── trainer.py                            # Training optimization routines
│   └── eva_funcs.py                          # Evaluation metrics (mDice, IoU, MAE, HD95)
├── train_optimized_cfanet.py                 # Training script
├── test_optimized_cfanet.py                  # Testing and evaluation script
├── run_train_autodl.sh                       # Batch training script
├── run_test_innovative.sh                    # Evaluation pipeline script
├── checkpoint/                               # Directory for saved checkpoints
└── results/                                  # Directory for exported segmentation predictions

```

---

## 4. Environment Requirements

### Hardware & Software

* **OS**: Linux / Ubuntu 20.04+ (or Windows 10/11)
* **Python**: >= 3.8
* **PyTorch**: >= 1.9.0
* **CUDA**: 11.3+ with compatible NVIDIA GPU (24GB VRAM recommended for batch size 8)

### Installation

```bash
# Clone the repository
git clone [https://github.com/lyf0323/BRAD-Net.git](https://github.com/lyf0323/BRAD-Net.git)
cd BRAD-Net

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements_optimized.txt

```

---

## 5. Model Weights & Releases

Pretrained backbone weights and trained model checkpoints are hosted on the GitHub Releases portal:
👉 **[Download Weights from GitHub Releases v1.0.0](https://www.google.com/search?q=https://github.com/lyf0323/BRAD-Net/releases/tag/v1.0.0)**

1. Download `res2net50_v1b_26w_4s-3cf99910.pth` and place it in the `lib/` directory.
2. Download `BRAD-Net_best.pth` and place it in the `checkpoint/innovative_cfanet/` directory for direct evaluation.

---

## 6. Training & Evaluation Workflow

### 6.1 Training

To train BRAD-Net from scratch using the default innovative contrastive-decoupled decoder:

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
    --train_path ./TrainDatasetEdges/ \
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

```

### 6.2 Testing & Benchmark Evaluation

To evaluate the trained checkpoint across all 4 benchmarking datasets:

```bash
python test_optimized_cfanet.py \
    --pth_path ./checkpoint/innovative_cfanet/BRAD-Net_best.pth \
    --test_root ./TestDataset/ \
    --save_root ./results/innovative/ \
    --datasets "CVC-300,CVC-ClinicDB,Kvasir,ETIS-LaribPolypDB" \
    --decoder_type innovative \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --testsize 352 \
    --threshold 0.5 \
    --channel 64 \
    --mamba_dim 96 \
    --save_results True

```

The script will compute standard segmentation metrics, including **Mean Dice (mDice)**, **Mean IoU (mIoU)**, **Mean Absolute Error (MAE)**, and boundary-aware measures.

---

## 7. Citations

If you find this work or code useful in your research, please cite:

```bibtex
@article{bradnet2026,
  title   = {BRAD-Net: Boundary-Region Adaptive Decoupling Network for Polyp Segmentation},
  author  = {Guangchao Zhou and Co-authors},
  journal = {PeerJ Computer Science},
  year    = {2026}
}

```

### Acknowledgments & Baselines

```bibtex
@article{zhou2023cfanet,
  title   = {Cross-level Feature Aggregation Network for Polyp Segmentation},
  author  = {Zhou, Tao and Zhou, Yi and He, Kelei and Gong, Chen and Yang, Jian and Fu, Huazhu and Shen, Dinggang},
  journal = {Pattern Recognition},
  volume  = {140},
  pages   = {109555},
  year    = {2023}
}

@article{gao2021res2net,
  title   = {Res2Net: A New Multi-scale Backbone Architecture},
  author  = {Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {43},
  number  = {2},
  pages   = {652--662},
  year    = {2021}
}

@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv preprint arXiv:2312.00752},
  year    = {2023}
}

```

---

## 8. License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

Third-party datasets and pretrained model backbones remain subject to their respective original licenses and terms of use.

```

```
