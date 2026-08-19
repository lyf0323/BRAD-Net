# BRAD-Net

**Boundary-aware dual-encoder network for polyp segmentation**

## Table of Contents

- [1. Title & Description](#1-title--description)
- [2. Dataset & Code Information](#2-dataset--code-information)
- [3. Requirements](#3-requirements)
- [4. Usage Instructions / Steps for Implementation](#4-usage-instructions--steps-for-implementation)
- [5. Citations](#5-citations)
- [6. License](#6-license)

---

## 1. Title & Description

**BRAD-Net** is a medical image segmentation model for **colonoscopy polyp segmentation**. It is designed to handle large scale variation and weak polyp–background boundaries.

The default architecture is:

- **Dual encoder**: Res2Net-50 extracts local multi-scale texture; Vision Mamba models long-range context with selective scan.
- **Adaptive fusion**: multi-level Res2Net and Mamba features are aligned and fused into five scales (`x0`–`x4`).
- **Innovative decoder** (default): query-guided aggregation, three-level boundary–region contrastive learning, 4-stage progressive refinement, and dual-stream boundary prediction.

Training uses a combination of BCE, Dice, boundary, and contrastive losses. Testing supports five public polyp datasets and optional test-time augmentation (TTA).

This repository provides training, testing, resume, and TTA scripts so that experiments can be reproduced from data preparation to evaluation.

---

## 2. Dataset & Code Information

### 2.1 Datasets

Experiments follow the common polyp-segmentation protocol. Training uses a combined set with **images**, **masks**, and **edge maps**. Testing uses five public benchmarks:

| Split | Dataset | Role |
|-------|---------|------|
| Train | Combined train set (from Kvasir / CVC-ClinicDB protocol) | Supervise region and boundary |
| Test | **CVC-300** | Seen-style / in-distribution test |
| Test | **CVC-ClinicDB** | Clinic colonoscopy frames |
| Test | **Kvasir** | Large-variation polyps |
| Test | **CVC-ColonDB** | Unseen / generalization |
| Test | **ETIS-LaribPolypDB** | Unseen / generalization |

Place data as follows (paths can be changed via CLI arguments):

```text
TrainDatasetEdges/
├── images/          # RGB frames (.jpg / .png)
├── masks/           # binary polyp masks (same stem as images)
└── edges/           # boundary maps for boundary supervision

TestDataset/
├── CVC-300/
│   ├── images/
│   └── masks/
├── CVC-ClinicDB/
├── Kvasir/
├── CVC-ColonDB/
└── ETIS-LaribPolypDB/
```

Download Res2Net-50 pretrained weights and put them at:

```text
lib/res2net50_v1b_26w_4s-3cf99910.pth
```

Typical source: [Res2Net official weights](https://github.com/Res2Net/Res2Net-PretrainedModels).

### 2.2 Code structure

```text
BRAD-Net/
├── README.md
├── LICENSE
├── requirements_optimized.txt
├── lib/
│   ├── BRAD_Net.py                          # Dual encoder + decoder (main model)
│   ├── res2net_v1b_base.py                  # Res2Net-50 backbone
│   └── res2net50_v1b_26w_4s-3cf99910.pth    # Pretrained CNN weights (download)
├── utils/
│   ├── dataloader.py                        # Train / test loaders and augmentation
│   ├── trainer.py                           # Learning-rate schedule helpers
│   └── eva_funcs.py                         # Metrics (e.g. S-measure, MAE)
├── train_optimized_cfanet.py                # Training entry
├── test_optimized_cfanet.py                 # Testing / evaluation entry
├── run_train_autodl.sh                      # Example training script
├── run_test_innovative.sh                   # Example testing script
├── resume_training_from_best.sh / .bat      # Resume from best checkpoint
├── resume_training_fixed.sh                 # Resume after contrastive-loss fix
├── test_with_tta.sh / .bat                  # Multi-scale + flip TTA
├── TrainDatasetEdges/                       # Training data (user-provided)
├── TestDataset/                             # Test data (user-provided)
├── checkpoint/                              # Saved weights (created at run time)
└── results/                                 # Prediction maps (created at run time)
```

### 2.3 Model overview

```text
Input (352 x 352)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
   Res2Net-50    Vision Mamba         │
   (local CNN)   (long-range SSM)     │
        │              │              │
        └──────┬───────┘              │
               ▼                      │
   Adaptive Multi-Level Fusion        │
               │                      │
               ▼                      │
   Innovative decoder                 │
   (query + contrastive + dual-stream)│
               │                      │
               ▼                      │
     Mask + boundary map  ◄───────────┘
```

Decoder options (`--decoder_type`):

| Value | Description |
|-------|-------------|
| `innovative` | **Default.** 3-level contrastive learning, query aggregation, progressive refinement, dual-stream boundary |
| `ultralight` | Lighter variant (fewer queries / contrastive levels) |
| `simplified` | Progressive decoder without contrastive learning |
| `original` | Original CFANet-style decoder (baseline) |

---

## 3. Requirements

### 3.1 Environment

- Python **3.8+**
- NVIDIA GPU with CUDA (recommended; default resolution is 352×352)
- PyTorch **1.9+** with a matching `torchvision`

Install dependencies:

```bash
pip install -r requirements_optimized.txt
```

### 3.2 Python packages (`requirements_optimized.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 1.9.0 | Deep learning |
| `torchvision` | >= 0.10.0 | Image transforms |
| `numpy` | >= 1.21.0 | Arrays |
| `einops` | >= 0.6.0 | Tensor rearrange for Mamba |
| `timm` | >= 0.6.0 | Optional ViT weights |
| `opencv-python` | >= 4.5.0 | Image I/O |
| `Pillow` | >= 8.3.0 | PIL loading / augmentation |
| `matplotlib` | >= 3.4.0 | Visualization |
| `scikit-image` | >= 0.18.0 | Image processing |
| `tensorboard` | >= 2.7.0 | Training logs |
| `tqdm` | >= 4.62.0 | Progress bars |
| `pyyaml` | >= 5.4.0 | Config parsing |
| `scikit-learn` | >= 1.0.0 | Extra metrics |

Optional: `transformers`, `wandb`.

### 3.3 Hardware notes

- Recommended batch size: **8** on a 24 GB GPU at 352×352 with the innovative decoder.
- Reduce `--batchsize` or set `--use_amp true` if GPU memory is limited.
- Mixed precision (`--use_amp`) is off by default for numerical stability of contrastive loss.

---

## 4. Usage Instructions / Steps for Implementation

Run all commands from the `BRAD-Net/` directory (the folder that contains `train_optimized_cfanet.py`).

### Step 1. Clone / enter the project

```bash
cd BRAD-Net
```

### Step 2. Create a virtual environment and install packages

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements_optimized.txt
```

### Step 3. Prepare data and pretrained weights

1. Organize `TrainDatasetEdges/` and `TestDataset/` as in [Section 2.1](#21-datasets).
2. Download Res2Net weights into `lib/res2net50_v1b_26w_4s-3cf99910.pth`.
3. Edit dataset paths in the scripts or pass them on the command line.

### Step 4. Train

Recommended configuration (innovative decoder):

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

Or run the example script (update AutoDL-style paths first):

```bash
bash run_train_autodl.sh
```

**Loss weights**

| Loss | Default | Role |
|------|---------|------|
| BCE | 1.0 | Pixel classification |
| Dice | 1.0 | Region overlap |
| Boundary | 0.5 | Edge consistency |
| Contrastive | 0.2 | Boundary vs. region decoupling |

Best checkpoint is written to:

```text
checkpoint/innovative_cfanet/OptimizedCFANet_best.pth
```

TensorBoard:

```bash
tensorboard --logdir=./checkpoint/innovative_cfanet/logs --port=6006
```

### Step 5. Test / reproduce evaluation

```bash
python test_optimized_cfanet.py \
    --pth_path ./checkpoint/innovative_cfanet/OptimizedCFANet_best.pth \
    --test_root ./TestDataset/ \
    --save_root ./results/innovative/ \
    --datasets "CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB" \
    --decoder_type innovative \
    --num_region_queries 100 \
    --num_boundary_queries 25 \
    --testsize 352 \
    --threshold 0.5 \
    --channel 64 \
    --mamba_dim 96 \
    --save_results True
```

Example script (update `MODEL_PATH` / `TEST_ROOT` first):

```bash
bash run_test_innovative.sh
```

Reported metrics typically include **mDice**, **Dice**, **IoU**, **MAE**, Precision, Recall, F1, and optional boundary scores.

### Step 6. Optional: test-time augmentation

TTA uses scales `{0.75, 1.0, 1.25}` and horizontal flip (6 views). It is slower (~6×) and does not require retraining.

```bash
bash test_with_tta.sh
# Windows
test_with_tta.bat
```

### Step 7. Optional: resume training

```bash
bash resume_training_from_best.sh
# Windows
resume_training_from_best.bat
```

Set `BEST_MODEL`, `TRAIN_DATA`, and `TEST_DATA` inside the script. Resume uses `--resume path/to/OptimizedCFANet_best.pth`.

### Key hyperparameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `--decoder_type` | `innovative` | Decoder variant |
| `--channel` | 64 | Decoder base channels |
| `--mamba_dim` | 96 | Mamba embedding size |
| `--trainsize` / `--testsize` | 352 | Input resolution |
| `--batchsize` | 16 in parser; 8 in scripts | Mini-batch size |
| `--lr` | 1e-4 | Learning rate |
| `--weight_contrastive` | 0.2 | Contrastive loss weight |
| `--num_region_queries` | 100 | Region queries |
| `--num_boundary_queries` | 25 | Boundary queries |

---

## 5. Citations

If you use this code or the related methods, please cite the following works.

**CFANet (boundary-aware polyp segmentation baseline)**

```bibtex
@article{zhou2023cfanet,
  title   = {Cross-level Feature Aggregation Network for Polyp Segmentation},
  author  = {Zhou, Tao and Zhou, Yi and He, Kelei and Gong, Chen and Yang, Jian and Fu, Huazhu and Shen, Dinggang},
  journal = {Pattern Recognition},
  volume  = {140},
  pages   = {109555},
  year    = {2023}
}
```

**Res2Net backbone**

```bibtex
@article{gao2021res2net,
  title   = {Res2Net: A New Multi-scale Backbone Architecture},
  author  = {Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {43},
  number  = {2},
  pages   = {652--662},
  year    = {2021}
}
```

**Mamba / selective state spaces**

```bibtex
@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv preprint arXiv:2312.00752},
  year    = {2023}
}
```

**PraNet (boundary-aware polyp segmentation)**

```bibtex
@inproceedings{fan2020pranet,
  title     = {PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
  author    = {Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  booktitle = {MICCAI},
  pages     = {263--273},
  year      = {2020}
}
```

**Supervised contrastive learning**

```bibtex
@inproceedings{khosla2020supcon,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

**Query-based decoding (MaskFormer)**

```bibtex
@inproceedings{cheng2021maskformer,
  title     = {Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author    = {Cheng, Bowen and Schwing, Alexander G. and Kirillov, Alexander},
  booktitle = {NeurIPS},
  year      = {2021}
}
```

**Datasets**

```bibtex
@article{bernal2015clinicdb,
  title   = {WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Experimental Assessment},
  author  = {Bernal, Jorge and S{\'a}nchez, F. Javier and Fern{\'a}ndez-Esparrach, Gloria and Gil, Debora and Rodr{\'i}guez, Cristina and Vilari{\~n}o, Fernando},
  journal = {Computerized Medical Imaging and Graphics},
  volume  = {48},
  pages   = {99--111},
  year    = {2016}
}

@inproceedings{jha2020kvasir,
  title     = {Kvasir-SEG: A Segmented Polyp Dataset},
  author    = {Jha, Debesh and Smedsrud, Pia H. and Riegler, Michael A. and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D.},
  booktitle = {MMM},
  year      = {2020}
}
```

Official CFANet code: [https://github.com/taozh2017/CFANet](https://github.com/taozh2017/CFANet).

---

## 6. License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file in this directory.

Pretrained weights (Res2Net, optional ViT) and third-party datasets remain under their original licenses. Please follow those terms when redistributing models or data.
