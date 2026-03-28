# CSR: CLIP Spectral Robustness

> **📄 Paper Status**: The associated paper is currently **under review**. Full details will be released upon acceptance.

A **training-free**, frequency-domain adversarial defense framework for CLIP vision-language models.

## Overview

CSR (CLIP Spectral Robustness) defends CLIP models against adversarial attacks by leveraging the **spectral properties** of adversarial perturbations. The core insight is that adversarial perturbations concentrate in high-frequency components, causing a measurable discrepancy between features of the original image and its low-pass filtered version.

### Key Features

- **Training-free**: No adversarial fine-tuning or additional training required
- **Model-agnostic**: Works with any CLIP variant (ViT-B/16, ViT-B/32, ViT-L/14)
- **Two-stage defense**: Spectral detection + PGD-based purification
- **Drop-in replacement**: Compatible with standard CLIP inference pipelines

### Defense Pipeline

1. **Detection** — Compare CLIP features of the original image vs. its low-pass filtered version. A low cosine similarity indicates adversarial manipulation.
2. **Purification** — PGD-based optimization that pushes the adversarial image's features toward the clean low-frequency reference while repelling from the adversarial features, with best-step tracking.

## Project Structure

```
CSR/
├── csr/                          # Core library
│   ├── config.py                 # Centralized configuration (paths & hyperparams)
│   ├── data/                     # Dataset loading
│   │   ├── dataset.py            # CLIPDataset, PairedCleanAdvDataset
│   │   └── collate.py            # Custom collate functions
│   ├── models/                   # Model management
│   │   ├── loader.py             # Load CLIP / FARE / TeCoA models
│   │   └── model_manager.py      # Unified model manager with caching
│   ├── attacks/                  # Adversarial attacks
│   │   ├── attacker.py           # CLIPAttacker wrapper
│   │   ├── pgd.py                # PGD-Linf attack
│   │   ├── cw.py                 # CW margin-loss attack
│   │   ├── apgd.py               # Auto-PGD attack
│   │   └── adaptive_pgd.py       # Adaptive PGD (frequency-aware)
│   ├── defense/                  # Defense methods
│   │   ├── csr.py                # CSR defense (core method)
│   │   ├── csr_fast.py           # FastCSR (optimized version)
│   │   └── ttc.py                # TTC baseline defense
│   ├── evaluation/               # Evaluation pipelines
│   │   ├── evaluator.py          # Zero-shot evaluator
│   │   └── benchmark.py          # Multi-model benchmark runner
│   └── utils/                    # Shared utilities
├── scripts/                      # Entry-point scripts
│   ├── generate_adv.py           # Generate adversarial samples
│   ├── evaluate.py               # Run evaluation benchmark
│   └── demo_csr.py               # Quick demo
├── configs/
│   └── default.yaml              # Default configuration
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Summu77/CSR.git
cd CSR
pip install -e .
```

## Configuration

Paths to datasets and model weights can be set via **environment variables** or by editing `configs/default.yaml`:

```bash
export CSR_DATA_ROOT="/path/to/datasets"
export CSR_FARE_WEIGHTS="/path/to/FARE/L-14_eps4.pt"
export CSR_TECOA_WEIGHTS="/path/to/TeCoA/L-14_eps4.pt"
```

## Quick Start

### 1. CSR Defense Demo

```python
from csr.defense import CSRDefense, FastCSRDefense
from csr.config import CSRConfig

# Initialize defense
defense = FastCSRDefense(
    "openai/clip-vit-large-patch14",
    config=CSRConfig(lpf_radius=40, detect_thresh=0.85, purify_steps=3),
    device="cuda"
)

# Detect adversarial samples
is_adv = defense.detect(image_tensor)  # [B] boolean

# Zero-shot prediction with automatic defense
probs, logits, attack_mask = defense.predict_zero_shot(
    image_tensor,          # [B, 3, 224, 224] in [0, 1]
    ["cat", "dog", "bird"]
)

# Drop-in CLIP replacement
features = defense.get_image_features(pixel_values=image_tensor)
```

### 2. Generate Adversarial Samples

```bash
# PGD attack
python scripts/generate_adv.py --attack pgd --model CLIP-L-14 --epsilon 4 --device cuda:0

# CW attack
python scripts/generate_adv.py --attack cw --model CLIP-L-14 --epsilon 4 --steps 30

# Auto-PGD attack
python scripts/generate_adv.py --attack apgd --model CLIP-L-14 --epsilon 4
```

### 3. Evaluate Robustness

```bash
# Evaluate with CSR defense
python scripts/evaluate.py \
    --model CLIP-L-14 \
    --defense fast_csr \
    --adv_root ./outputs/adv_samples/PGD \
    --adv_attack PGD \
    --output results/csr_eval.csv

# Evaluate baseline (no defense)
python scripts/evaluate.py --model CLIP-L-14 --defense none
```

## Supported Models

| Model | Type | Source |
|-------|------|--------|
| CLIP ViT-B/16 | HuggingFace | `openai/clip-vit-base-patch16` |
| CLIP ViT-B/32 | HuggingFace | `openai/clip-vit-base-patch32` |
| CLIP ViT-L/14 | HuggingFace / OpenCLIP | `openai/clip-vit-large-patch14` |
| FARE ViT-L/14 | OpenCLIP (adversarial fine-tuned) | Custom weights |
| TeCoA ViT-L/14 | OpenCLIP (adversarial fine-tuned) | Custom weights |

## Supported Attacks

| Attack | Method | Loss |
|--------|--------|------|
| PGD | Projected Gradient Descent | Cross-Entropy |
| CW | Carlini-Wagner (PGD variant) | Margin Loss |
| APGD | Auto-PGD | Cross-Entropy (adaptive step) |
| Adaptive PGD | Frequency-aware PGD | Cross-Entropy + HF regularization |

## Datasets

Supports 16 zero-shot classification benchmarks:

- **General**: ImageNet, CIFAR-10, CIFAR-100, STL-10, Caltech-101, Caltech-256
- **Fine-Grained**: OxfordPets, Flowers102, Food101, StanfordCars
- **Scene**: SUN397, Country211
- **Domain**: FGVCAircraft, EuroSAT, DTD, PCAM

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_type` | `gaussian` | LPF type: `gaussian` / `butterworth` / `ideal` |
| `lpf_radius` | `40` | Low-pass filter cutoff radius |
| `detect_thresh` | `0.85` | Cosine similarity threshold for detection |
| `purify_steps` | `3` | Number of PGD purification iterations |
| `purify_eps` | `4/255` | Purification perturbation budget |
| `purify_alpha` | `2/255` | Purification step size |

## Citation

> Paper is currently under review. Citation information will be available upon publication.

## License

MIT
