# Skin Cancer Detection Model

A deep learning system for classifying skin lesions into 7 diagnostic categories using a fine-tuned Vision Transformer (ViT), trained on the HAM10000 dermatology dataset.

---

## Overview

This project fine-tunes a **Vision Transformer (ViT-Base)** pretrained on ImageNet-21k to detect and classify skin cancer from dermoscopy images. The model distinguishes between 7 types of skin lesions and achieves **83% accuracy** on the test set.

---

## Classes

| Label | Description |
|-------|-------------|
| `akiec` | Actinic Keratoses & Intraepithelial Carcinoma |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

---

## Model Architecture

- **Base Model:** `google/vit-base-patch16-224-in21k` (Vision Transformer)
- **Framework:** PyTorch + HuggingFace Transformers
- **Input Size:** 224×224 RGB images
- **Output:** 7-class softmax classification

### Training Strategy

A two-stage fine-tuning approach was used:

**Stage 1 – Head Training**
- All ViT layers frozen except the classifier head
- Trained for up to 10 epochs with early stopping (patience = 5)
- Learning rate: `1e-3` with cosine scheduler and 100 warmup steps

**Stage 2 – Full Fine-tuning**
- All layers unfrozen
- Trained for up to 15 epochs with early stopping (patience = 5)
- Learning rate: `1e-5` with cosine scheduler

### Handling Class Imbalance
- Computed balanced class weights using `sklearn`
- Weights clipped at 5.0 to avoid training instability
- Applied as weights in `CrossEntropyLoss`

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **83%** |
| Loss Function | Weighted Cross-Entropy |
| Optimizer | AdamW |

---

## Dataset

**HAM10000** (Human Against Machine with 10000 training images)
- Source: [ISIC Archive](https://www.isic-archive.com/)
- 10,015 dermoscopy images across 7 lesion types
- Split by unique `lesion_id` to prevent data leakage between train/val/test sets

**Split ratio:**
- Train: 70%
- Validation: 15%
- Test: 15%

---

## Getting Started

### Requirements

```bash
pip install torch torchvision transformers scikit-learn pandas pillow seaborn matplotlib
```

### Dataset Setup

1. Download the HAM10000 dataset from the [ISIC Archive](https://www.isic-archive.com/)
2. Place images in the following structure:

```
skin_cancer_project/
├── ham10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
```

### Training

Open the notebook and run all cells in order:

```
Untitled0.ipynb
```

The notebook will:
1. Load and split the dataset
2. Initialize the ViT model
3. Run Stage 1 training (head only)
4. Run Stage 2 fine-tuning (full model)
5. Evaluate on the test set and generate a confusion matrix

### Loading the Trained Model

```python
from transformers import ViTForImageClassification, AutoImageProcessor
import torch

processor = AutoImageProcessor.from_pretrained('path/to/saved/model')
model = ViTForImageClassification.from_pretrained('path/to/saved/model')
model.eval()
```

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- Google Colab (training environment)

---

## License

This project is for educational purposes. The HAM10000 dataset is subject to its own license from the ISIC Archive.
