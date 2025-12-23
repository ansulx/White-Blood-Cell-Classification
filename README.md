

gdown 1nbRi3vuo414PigZGpNa6746t3Sy1EdZW -O dataset.zip
apt-get update && apt-get install -y unzip
unzip dataset.zip
pip install -r requirements.txt

# WBC-Bench-2026 Competition Solution

A comprehensive solution for the **WBCBench 2026** Kaggle competition focused on White Blood Cell (WBC) image classification.

**Competition Links:**
- 🏆 [Kaggle Competition Page](https://www.kaggle.com/competitions/wbc-bench-2026)
- 📋 [WBCBench 2026 Website](https://www.kaggle.com/competitions/wbc-bench-2026) (check competition overview for official website)
- 📝 [Participation Form](https://www.kaggle.com/competitions/wbc-bench-2026) (see competition overview)

## 🚀 **NEW TO THIS? START HERE!**

1. **Read this first**: [`STEP_BY_STEP_START_HERE.md`](STEP_BY_STEP_START_HERE.md) - Complete day-by-day action plan
2. **Full guide**: [`COMPLETE_GUIDE.md`](COMPLETE_GUIDE.md) - Everything you need to know
3. **Quick reference**: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) - Command cheat sheet

## 📚 Documentation

- **[STEP_BY_STEP_START_HERE.md](STEP_BY_STEP_START_HERE.md)** - Beginner-friendly day-by-day guide
- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** - Comprehensive strategy and techniques
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference

## 🎯 Competition Overview

**WBCBench 2026** is an ISBI 2026 Grand Challenge designed to benchmark the robustness of machine learning models for hematological image analysis. This competition evaluates algorithms for automated white blood cell (WBC) classification under:
- **Severe class imbalance** - Rare classes must be accurately identified
- **Fine-grained morphological variation** - Subtle differences between similar cell types
- **Simulated domain shift** - Controlled variations in scanner characteristics and imaging settings (noise, blur, color perturbations)

### Competition Goals

1. **Comparability & Reproducibility**: Standardized splits and an open evaluator with a fixed submission format
2. **Rare-Class Reliability**: Prioritize macro-F1 and require class-wise reporting to highlight minority performance
3. **Robustness**: Models must perform well under realistic variations in imaging conditions

### Cell Type Classes

This competition involves classifying white blood cell images into 13 different classes:
- **BA** (Basophil)
- **BL** (Blast cell) - *Clinically critical rare class*
- **BNE** (Band-form neutrophil)
- **EO** (Eosinophil)
- **LY** (Lymphocyte)
- **MMY** (Metamyelocyte)
- **MO** (Monocyte)
- **MY** (Myelocyte)
- **PC** (Plasma cell)
- **PLY** (Prolymphocyte)
- **PMY** (Promyelocyte)
- **SNE** (Segmented neutrophil)
- **VLY** (Variant/atypical lymphocyte)

## ⚠️ **CRITICAL COMPETITION DETAILS**

### Evaluation Metric

**Primary Metric: Macro-Averaged F1 Score**

The competition uses **macro-averaged F1 score** as the primary evaluation metric. This metric treats each class equally, making it suitable for highly imbalanced datasets and ensuring that rare-cell performance is appropriately measured.

**Formula:**
- For each class `c`, compute precision `P_c`, recall `R_c`, and F1 score `F1_c`
- Macro-averaged F1 = `(1/C) × Σ F1_c` where `C` is the total number of classes

**Tie-Breaking Criteria** (in order):
1. Balanced Accuracy
2. Macro-Averaged Precision
3. Macro-Averaged Specificity
4. Inference Time (faster models ranked higher)

### Key Challenges

1. **Severe Class Imbalance**: Rare classes (especially BL - Blast cells) must be accurately identified despite limited training examples
2. **Domain Shift**: Test images contain controlled variations (noise, blur, color perturbations) simulating real-world scanner variability
3. **Patient-Level Separation**: Strict patient-level separation between train/test splits - no data leakage allowed
4. **Fine-Grained Classification**: Subtle morphological differences between similar cell types (e.g., different neutrophil stages)

### Dataset Structure & Phases

**Phase 1** (15% of patients - 74 patients, 8,288 images):
- Pristine training set with high-quality images
- Located in `phase1/` directory
- Labels in `phase1_label.csv`

**Phase 2** (85% of patients with image degradation):
- **Training** (≈45% of patients - 222 patients, 24,897 images): `phase2/train/`
- **Evaluation** (≈10% of patients - 49 patients, 5,350 images): `phase2/eval/` 
  - ⚠️ **May be used for training** (validation set)
- **Test** (≈30% of patients - 148 patients, 16,477 images): `phase2/test/`
  - ⚠️ **Used for leaderboard ranking** - no labels provided

**Important Notes:**
- Patient-level separation is maintained between all splits
- Phase 2 images contain simulated domain shift (noise, blur, color variations)
- Labels use consistent abbreviations across all CSV files

### Submission Requirements

**Format:**
- CSV file with columns: `ID, labels`
- One row per test image
- Labels must use the exact abbreviations listed above

**Code Submission (Top 5 Teams):**
- Top-ranking participants must submit:
  - Trained models
  - Complete inference code or containerized environment
  - Sufficient documentation to reproduce final predictions
- Teams whose results cannot be reproduced may be disqualified

**Competition Rules:**
- Maximum 10 submissions per day
- Maximum team size: 5 members
- External data/models allowed if publicly available and reasonably accessible
- Code sharing must be public (on Kaggle forums) if shared during competition

## 📁 Dataset Structure

```
data/
├── phase1/
│   └── images/                 # Phase 1 training images (8,288 images, 74 patients)
├── phase2/
│   ├── train/                  # Phase 2 training images (24,897 images, 222 patients)
│   ├── eval/                   # Phase 2 eval images (5,350 images, 49 patients) - may use for training
│   └── test/                   # Phase 2 test images (16,477 images, 148 patients) - for leaderboard
├── phase1_label.csv            # Labels for Phase 1 training images
├── phase2_train.csv            # Labels for Phase 2 'train' images
├── phase2_eval.csv             # Labels for Phase 2 'eval' images (validation, but may use for training)
└── phase2_test.csv             # Template for Test set submission (includes IDs, no labels)
```

**Dataset Summary:**
- **Total Images**: ~55,000+ images across all phases
- **Total Patients**: ~493 patients (strict patient-level separation)
- **Image Characteristics**: Single-site microscopic blood smear acquisitions with standardized staining
- **Domain Shift**: Phase 2 images contain controlled noise, blur, and color perturbations simulating real-world scanner variability

## 📁 Project Structure

This is a **research-grade, modular codebase** with clear separation of concerns:

```
wbc-bench-2026/
├── src/              # Core package (models, data, config)
├── scripts/          # Executable scripts
├── utils/            # Utility functions
├── docs/guides/      # Documentation
└── outputs/          # Generated outputs (models, predictions)
```

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed structure.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: Install as package for easier imports
pip install -e .
```

### 2. Explore the Dataset

```bash
python scripts/explore_data.py
```

This will analyze the dataset distribution, class imbalance, and image statistics.

### 3. Train Models

```bash
python scripts/train.py
```

This will:
- Combine Phase 1 and Phase 2 training data
- Perform 5-fold cross-validation
- Train EfficientNet-B4 models with advanced data augmentation
- Handle class imbalance with weighted loss
- Save best models for each fold

### 4. Make Predictions

```bash
python scripts/inference.py
```

This will:
- Load all trained models
- Create ensemble predictions
- Apply Test Time Augmentation (TTA)
- Generate predictions for test and eval sets
- Save results to `outputs/predictions/`

## 🏗️ Solution Architecture

### Models
- **Primary**: EfficientNet-B4 (pretrained on ImageNet)
- **Ensemble**: Multiple EfficientNet variants + ConvNeXt
- **Loss**: Focal Loss + Class Weights for handling imbalance

### Data Augmentation
- Horizontal/Vertical flips
- Random rotations
- Shift/Scale/Rotate
- Brightness/Contrast adjustments
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian noise and blur
- Coarse dropout

### Training Strategy
- **5-Fold Stratified Cross-Validation**: Ensures robust validation
- **Class Balancing**: Weighted sampling + class weights in loss
- **Learning Rate**: Cosine annealing schedule
- **Early Stopping**: Prevents overfitting
- **Image Size**: 384x384 for optimal performance

### Inference Strategy
- **Ensemble**: Average predictions from multiple models
- **Test Time Augmentation (TTA)**: 5 different augmentations
- **Soft Voting**: Average probabilities before final prediction

## 📊 Key Features

1. **Handles Class Imbalance**: Uses weighted loss and sampling
2. **Robust Validation**: 5-fold stratified cross-validation
3. **Advanced Augmentation**: Albumentations library for medical imaging
4. **Model Ensembling**: Combines multiple models for better accuracy
5. **TTA**: Test Time Augmentation for improved predictions
6. **Efficient Training**: Uses mixed precision and optimized data loading

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Image size
- Batch size
- Learning rate

## 📈 Expected Performance

With this solution, you should achieve:
- High accuracy on validation set (>95%)
- Robust generalization to test set
- Competitive leaderboard position

## 🔧 Tips for Improvement

1. **Try Different Models**: Experiment with ConvNeXt, Swin Transformers
2. **Increase Image Size**: Try 512x512 or 640x640 for better accuracy
3. **More Augmentation**: Add more aggressive augmentations
4. **Pseudo-labeling**: Use confident predictions to expand training set
5. **External Data**: If allowed, use additional WBC datasets
6. **Hyperparameter Tuning**: Use Optuna or similar tools

## 📝 Submission Format

**Kaggle Competition**: [WBCBench 2026](https://www.kaggle.com/competitions/wbc-bench-2026)

The predictions must be submitted in CSV format with exactly two columns:
```csv
ID,labels
01447013.jpg,SNE
01447014.jpg,LY
01447015.jpg,BL
...
```

**Requirements:**
- Column headers must be exactly: `ID,labels` (case-sensitive)
- `ID` column: Image filename (e.g., `01447013.jpg`)
- `labels` column: One of the 13 class abbreviations (BA, BL, BNE, EO, LY, MMY, MO, MY, PC, PLY, PMY, SNE, VLY)
- One row per test image
- No header row needed if submitting via Kaggle interface

## 🎓 Best Practices Used

- Stratified K-Fold for proper validation
- Class weights for imbalanced data
- Advanced data augmentation
- Model ensembling
- Test Time Augmentation
- Early stopping
- Learning rate scheduling
- Mixed precision training (if GPU available)

## 📚 References

- EfficientNet: https://arxiv.org/abs/1905.11946
- Focal Loss: https://arxiv.org/abs/1708.02002
- Albumentations: https://albumentations.ai/

## 🤝 Contributing

Feel free to experiment and improve upon this solution!

## 📄 License

This solution is provided for educational purposes in the context of the Kaggle competition.

