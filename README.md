# WBC-Bench-2026 Competition Solution

A comprehensive, research-grade solution for the **WBCBench 2026** Kaggle competition focused on White Blood Cell (WBC) image classification. This solution implements state-of-the-art models, advanced training strategies, and automated workflows to achieve top leaderboard performance.

**Competition Links:**
- 🏆 [Kaggle Competition Page](https://www.kaggle.com/competitions/wbc-bench-2026)
- 📋 [WBCBench 2026 Website](https://www.kaggle.com/competitions/wbc-bench-2026)

---

## 🚀 **ONE-COMMAND WORKFLOW**

```bash`
python scripts/train.py
```

**That's it!** This single command runs the complete automated pipeline from training to final submission.

### What Happens Automatically:

1. **Session 1**: Trains all 3 ensemble models (2-4 hours)
2. **Session 2**: Generates pseudo-labels and retrains best models (1-2 hours)
3. **Session 3**: Optimizes ensemble weights and generates final submission (30-60 minutes)

**Final Output**: `outputs/predictions/final_submission.csv` - Ready for Kaggle submission!

---

## 📋 Table of Contents

1. [Competition Overview](#competition-overview)
2. [Dataset Structure](#dataset-structure)
3. [Project Architecture](#project-architecture)
4. [Complete Workflow (Sessions 1-3)](#complete-workflow-sessions-1-3)
5. [Model Architectures](#model-architectures)
6. [Scripts Documentation](#scripts-documentation)
7. [Configuration Parameters](#configuration-parameters)
8. [Installation & Setup](#installation--setup)
9. [Expected Performance](#expected-performance)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Competition Overview

**WBCBench 2026** is an ISBI 2026 Grand Challenge designed to benchmark the robustness of machine learning models for hematological image analysis. This competition evaluates algorithms for automated white blood cell (WBC) classification under:

- **Severe class imbalance** - Rare classes must be accurately identified
- **Fine-grained morphological variation** - Subtle differences between similar cell types
- **Simulated domain shift** - Controlled variations in scanner characteristics and imaging settings (noise, blur, color perturbations)

### Evaluation Metric

**Primary Metric: Macro-Averaged F1 Score**

The competition uses **macro-averaged F1 score** as the primary evaluation metric. This metric treats each class equally, making it suitable for highly imbalanced datasets and ensuring that rare-cell performance is appropriately measured.

**Formula:**
- For each class `c`, compute precision `P_c`, recall `R_c`, and F1 score `F1_c`
- Macro-averaged F1 = `(1/C) × Σ F1_c` where `C` is the total number of classes (13)

**Tie-Breaking Criteria** (in order):
1. Balanced Accuracy
2. Macro-Averaged Precision
3. Macro-Averaged Specificity
4. Inference Time (faster models ranked higher)

### Cell Type Classes (13 Classes)

- **BA** (Basophil)
- **BL** (Blast cell) - *Clinically critical rare class - special monitoring implemented*
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

### Key Challenges

1. **Severe Class Imbalance**: Rare classes (especially BL - Blast cells) must be accurately identified despite limited training examples
2. **Domain Shift**: Test images contain controlled variations (noise, blur, color perturbations) simulating real-world scanner variability
3. **Patient-Level Separation**: Strict patient-level separation between train/test splits - no data leakage allowed
4. **Fine-Grained Classification**: Subtle morphological differences between similar cell types (e.g., different neutrophil stages)

---

## 📁 Dataset Structure

### Phase 1 (15% of patients - 74 patients, 8,288 images)
- Pristine training set with high-quality images
- Located in `phase1/` directory
- Labels in `phase1_label.csv`

### Phase 2 (85% of patients with image degradation)
- **Training** (≈45% of patients - 222 patients, 24,897 images): `phase2/train/`
- **Evaluation** (≈10% of patients - 49 patients, 5,350 images): `phase2/eval/` 
  - ⚠️ **May be used for training** (validation set) - **INCLUDED in training by default**
- **Test** (≈30% of patients - 148 patients, 16,477 images): `phase2/test/`
  - ⚠️ **Used for leaderboard ranking** - no labels provided

**Important Notes:**
- Patient-level separation is maintained between all splits
- Phase 2 images contain simulated domain shift (noise, blur, color variations)
- Labels use consistent abbreviations across all CSV files
- **Total Images**: ~55,000+ images across all phases
- **Total Patients**: ~493 patients (strict patient-level separation)

### Directory Structure

```
project_root/
├── phase1/                    # Phase 1 images (8,288 images)
├── phase2/
│   ├── train/                 # Phase 2 training images (24,897 images)
│   ├── eval/                  # Phase 2 eval images (5,350 images)
│   └── test/                  # Phase 2 test images (16,477 images)
├── phase1_label.csv           # Phase 1 labels
├── phase2_train.csv           # Phase 2 train labels
├── phase2_eval.csv            # Phase 2 eval labels
├── phase2_test.csv            # Phase 2 test IDs (no labels)
├── src/                       # Core package
├── scripts/                   # Executable scripts
└── outputs/                   # Generated outputs
    ├── models/                # Trained model checkpoints
    ├── predictions/           # Prediction CSVs
    └── logs/                  # Training metrics and plots
```

---

## 🏗️ Project Architecture

This is a **research-grade, modular codebase** with clear separation of concerns:

```
White-Blood-Cell-Classification/
├── src/                          # Core package (models, data, config)
│   ├── config.py                 # Central configuration (ALL parameters)
│   ├── metrics.py                # Evaluation metrics (macro F1, etc.)
│   ├── data/
│   │   ├── dataset.py           # WBCDataset class, transforms, augmentation
│   │   └── __init__.py
│   └── models/
│       ├── architectures.py      # Model factory (get_model function)
│       ├── losses.py             # Loss functions (Focal, CE with weights)
│       └── __init__.py
├── scripts/                       # Executable scripts
│   ├── train.py                  # MAIN SCRIPT - Complete automated workflow
│   ├── inference.py              # Inference utilities (library functions)
│   ├── pseudo_labeling.py        # Pseudo-label generation (Session 2)
│   ├── ensemble_optimizer.py     # Ensemble weight optimization (Session 3)
│   ├── final_submission.py       # Final submission generation (Session 3)
│   ├── validate_submission.py    # Submission format validation
│   ├── session3_summary.py       # Summary report generator
│   ├── hyperparameter_opt.py     # Optional: Optuna-based hyperparameter tuning
│   └── verify_eval_performance.py # Optional: CV vs Eval performance comparison
├── outputs/                      # Generated outputs (gitignored)
│   ├── models/                   # Model checkpoints (*_best.pth)
│   ├── predictions/              # Prediction CSVs
│   └── logs/                     # Metrics, plots, summaries
│       ├── metrics/              # CSV and JSON metrics per fold
│       └── plots/                # Training curves, LR schedules
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation (optional)
└── README.md                     # This file
```

---

## 🔄 Complete Workflow (Sessions 1-3)

### **SESSION 1: Foundation Models Training** (2-4 hours)

**What Happens:**
1. Loads all available training data:
   - Phase 1 training images (8,288)
   - Phase 2 training images (24,897)
   - **Phase 2 eval images (5,350)** - Included for maximum training data
   - **Total: ~38,535 training samples**

2. Trains 3 ensemble models sequentially (automatically):
   - **ConvNeXt V2 Large** (LR=5e-5, Batch=64, Drop Path=0.3)
   - **Swin V2 Large** (LR=4e-5, Batch=48, Drop Path=0.2)
   - **MaxViT XLarge** (LR=3e-5, Batch=32, Drop Path=0.2)

3. For each model:
   - **5-fold Stratified Cross-Validation**
   - Domain adaptation augmentation enabled
   - Class-aware augmentation for rare classes
   - BL class performance monitoring (warnings if F1 < 0.5)
   - Weighted loss with class weights
   - Label smoothing (0.1)
   - Mixup & CutMix augmentation
   - Learning rate warmup + cosine annealing
   - Early stopping (patience=10)
   - Mixed precision training (AMP)
   - PyTorch 2.0+ compilation (torch.compile)

4. **Output:**
   - 15 model files (3 models × 5 folds)
   - Training metrics (CSV, JSON) per fold
   - Training plots (loss curves, accuracy, LR schedule)
   - CV performance summary

**Key Features:**
- **Auto-detection**: Automatically uses pseudo-labels if `merged_train_with_pseudo.csv` exists
- **Model-specific settings**: Each model gets optimized hyperparameters automatically
- **BL monitoring**: Special tracking for critical rare class (Blast cells)
- **Phase 2 Eval inclusion**: Adds ~5,350 additional samples (10% more data)

---

### **SESSION 2: Pseudo-Labeling & Retraining** (1-2 hours)

**What Happens (if `RUN_PSEUDO_LABELING = True`):**

1. **Pseudo-Label Generation:**
   - Uses all trained models from Session 1 for ensemble predictions
   - Generates predictions on eval set with TTA (15 transforms)
   - Filters by confidence threshold (default: 0.95 - very high)
   - Ensures minimum samples per class (default: 10)
   - Saves to `outputs/predictions/pseudo_labels_thresh0.95.csv`

2. **Merge with Training Data:**
   - Combines original training data with high-confidence pseudo-labels
   - Saves merged dataset to `outputs/predictions/merged_train_with_pseudo.csv`
   - This file is automatically detected in future training runs

3. **Retraining Best Models:**
   - Automatically identifies top 2 best models (by macro F1)
   - Retrains them with expanded dataset (original + pseudo-labels)
   - Uses same 5-fold CV strategy
   - Applies model-specific hyperparameters

4. **Output:**
   - Additional 10 model files (2 models × 5 folds, retrained)
   - **Total: 25 model files** (15 from Session 1 + 10 from Session 2)
   - Improved performance with pseudo-labels

**Key Features:**
- **Automatic**: No manual intervention needed
- **High confidence**: Only 0.95+ confidence predictions used
- **Class balance**: Ensures minimum samples per class
- **Best model selection**: Automatically picks top 2 performers

---

### **SESSION 3: Final Ensemble Optimization & Submission** (30-60 minutes)

**What Happens (if `RUN_FINAL_SUBMISSION = True`):**

1. **Ensemble Weight Optimization:**
   - Tests 4 ensemble methods on eval set:
     - **Weighted Ensemble**: Weights by individual model macro F1
     - **Equal Weight**: Simple averaging
     - **Classy Ensemble**: Per-class weighted ensemble (best for rare classes)
     - **Grid Search**: Exhaustive weight search (optional)
   - Selects best method based on eval macro F1
   - Saves results to `outputs/logs/ensemble_optimization_results.json`

2. **Final Submission Generation:**
   - Generates predictions on eval set (for validation)
   - Computes eval metrics (macro F1, accuracy)
   - Generates final predictions on test set
   - Uses optimized ensemble weights
   - Applies TTA (15 transforms) if enabled
   - Saves to `outputs/predictions/final_submission.csv`

3. **Submission Validation:**
   - Validates format (columns, class labels, duplicates)
   - Checks against test CSV (ID matching)
   - Reports warnings/errors
   - Ensures submission is ready for Kaggle

4. **Output:**
   - `outputs/predictions/final_submission.csv` - **Ready for Kaggle!**
   - `outputs/predictions/final_eval_predictions.csv` - Eval predictions
   - `outputs/logs/final_submission_metrics.json` - Performance metrics
   - `outputs/logs/ensemble_optimization_results.json` - Optimization results

**Key Features:**
- **Automatic optimization**: Tests all methods and picks best
- **Comprehensive validation**: Catches format errors before submission
- **Eval metrics**: Reports expected performance on eval set
- **Edge case handling**: Handles missing models, OOM, format errors

---

## 🤖 Model Architectures

### Supported Models

The solution supports multiple state-of-the-art architectures via `timm` library:

#### **ConvNeXt V2 (RECOMMENDED - Best CNN)**
- `convnextv2_base` - Base variant
- `convnextv2_large` - **BEST CNN** - Used in ensemble
- **Features**: Modern CNN, FCMAE pretraining, excellent for medical imaging
- **Drop Path Rate**: 0.3 (default for Large)

#### **Swin Transformer V2 (Best Transformer)**
- `swinv2_base_window8_256` - Base variant
- `swinv2_large_window12to16_192to256` - **BEST TRANSFORMER** - Used in ensemble
- **Features**: Vision transformer, global feature learning, optimized window size for 512px images
- **Drop Path Rate**: 0.2 (default)

#### **MaxViT (Best Multi-Axis)**
- `maxvit_base_tf_384` - Base variant
- `maxvit_large_tf_384` - Large variant
- `maxvit_xlarge_tf_384` - **BEST MULTI-AXIS** - Used in ensemble
- **Features**: Multi-axis vision transformer, combines CNN and transformer benefits
- **Drop Path Rate**: 0.2 (default)

#### **Other Supported Models**
- **ConvNeXt**: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
- **EfficientNet**: `efficientnet_b0` through `efficientnet_b7`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **Swin Transformer**: `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224`, etc.

### Ensemble Strategy

**Current Ensemble (3 Models):**
1. **ConvNeXt V2 Large** - Strongest CNN architecture
2. **Swin V2 Large** - Best transformer with optimized window size
3. **MaxViT XLarge** - Best multi-axis transformer

**Why These Models:**
- **Diversity**: Different architectures (CNN, Transformer, Multi-Axis)
- **Performance**: Best available variants of each architecture
- **Complementary**: Each model captures different features
- **Optimized**: Window sizes and hyperparameters tuned for 512px images

---

## 📜 Scripts Documentation

### **`scripts/train.py`** - Main Training Script

**Purpose**: Complete automated workflow orchestrator

**What It Does:**
- Orchestrates Sessions 1, 2, and 3
- Auto-trains all ensemble models
- Generates pseudo-labels
- Retrains best models
- Optimizes ensemble weights
- Generates final submission

**Key Functions:**
- `main()` - Main orchestrator function
- `train_all_ensemble_models()` - Trains all 3 ensemble models sequentially
- `train_single_model()` - Trains one model with 5-fold CV
- `train_fold()` - Trains a single fold
- `train_epoch()` - Training loop with mixup/cutmix
- `validate()` - Validation with metrics computation
- `save_training_results()` - Saves metrics, plots, summaries

**Usage:**
```bash
python scripts/train.py
```

**Configuration Flags:**
- `TRAIN_ALL_ENSEMBLE = True` - Enable auto-train
- `RUN_PSEUDO_LABELING = True` - Enable Session 2
- `RUN_FINAL_SUBMISSION = True` - Enable Session 3

---

### **`scripts/inference.py`** - Inference Utilities Library

**Purpose**: Provides inference functions (used as library, not run directly)

**Key Functions:**
- `load_model()` - Loads trained model checkpoint
- `predict_single_model()` - Single model prediction
- `predict_ensemble()` - Equal-weight ensemble prediction
- `predict_ensemble_classy()` - Per-class weighted ensemble (Classy Ensemble)
- `predict_ensemble_optimized()` - Optimized weight ensemble
- `predict_test_set()` - Generate test set predictions
- `predict_eval_set()` - Generate eval set predictions

**Usage:**
- Called automatically by Session 3
- Can be run standalone for quick inference: `python scripts/inference.py`

---

### **`scripts/pseudo_labeling.py`** - Pseudo-Label Generation

**Purpose**: Generates high-confidence pseudo-labels from eval set

**Key Functions:**
- `generate_pseudo_labels()` - Main pseudo-label generation
- `merge_pseudo_labels_with_training()` - Merges with training data

**Parameters:**
- `confidence_threshold = 0.95` - Minimum confidence to accept
- `min_samples_per_class = 10` - Minimum samples per class

**Output:**
- `outputs/predictions/pseudo_labels_thresh0.95.csv`
- `outputs/predictions/merged_train_with_pseudo.csv`

**Usage:**
- Called automatically by Session 2
- Can be run standalone if needed

---

### **`scripts/ensemble_optimizer.py`** - Ensemble Weight Optimization

**Purpose**: Optimizes ensemble weights to maximize macro F1

**Methods Tested:**
1. **Weighted Ensemble**: Weights by individual model macro F1
2. **Equal Weight**: Simple averaging
3. **Classy Ensemble**: Per-class weighted (best for rare classes)
4. **Grid Search**: Exhaustive search (optional)

**Output:**
- `outputs/logs/ensemble_optimization_results.json`

**Usage:**
- Called automatically by Session 3
- Can be run standalone: `python scripts/ensemble_optimizer.py`

---

### **`scripts/final_submission.py`** - Final Submission Generation

**Purpose**: Generates and validates final submission

**What It Does:**
1. Optimizes ensemble weights (if enabled)
2. Generates eval predictions (for validation)
3. Generates test predictions (final submission)
4. Validates submission format
5. Saves metrics

**Output:**
- `outputs/predictions/final_submission.csv` - **Main submission file**
- `outputs/predictions/final_eval_predictions.csv`
- `outputs/logs/final_submission_metrics.json`

**Usage:**
- Called automatically by Session 3
- Can be run standalone: `python scripts/final_submission.py`

---

### **`scripts/validate_submission.py`** - Submission Validation

**Purpose**: Comprehensive submission format validation

**Checks:**
- Required columns (ID, labels)
- Duplicate IDs
- Valid class labels (13 classes)
- Missing values
- ID format (image filenames)
- Test set ID matching
- Class distribution

**Usage:**
```bash
python scripts/validate_submission.py
```

---

### **`scripts/session3_summary.py`** - Summary Report

**Purpose**: Generates Session 3 summary report

**What It Shows:**
- Final performance metrics
- Ensemble optimization results
- Submission file status
- Model breakdown

**Usage:**
```bash
python scripts/session3_summary.py
```

---

### **Optional Scripts**

#### **`scripts/hyperparameter_opt.py`** - Hyperparameter Optimization

**Purpose**: Optuna-based hyperparameter tuning

**Requires:** `pip install optuna`

**Optimizes:**
- Learning rate
- Weight decay
- Drop path rate

**Usage:**
```bash
python scripts/hyperparameter_opt.py
# Update config.py with best hyperparameters
# Retrain if needed
```

#### **`scripts/verify_eval_performance.py`** - Performance Verification

**Purpose**: Compares CV performance with eval set performance

**Usage:**
```bash
python scripts/verify_eval_performance.py
```

---

## ⚙️ Configuration Parameters

All configuration is in `src/config.py`. Here's a complete breakdown of all parameters and their effects:

### **Path Configuration**

```python
BASE_DIR = Path(__file__).parent.parent  # Project root
PHASE1_DIR = BASE_DIR / 'phase1'         # Phase 1 images
PHASE2_TRAIN_DIR = BASE_DIR / 'phase2' / 'train'  # Phase 2 train images
PHASE2_TEST_DIR = BASE_DIR / 'phase2' / 'test'     # Phase 2 test images
PHASE2_EVAL_DIR = BASE_DIR / 'phase2' / 'eval'     # Phase 2 eval images
OUTPUT_DIR = BASE_DIR / 'outputs'                  # Output directory
MODEL_DIR = OUTPUT_DIR / 'models'                   # Model checkpoints
PRED_DIR = OUTPUT_DIR / 'predictions'              # Prediction CSVs
LOG_DIR = OUTPUT_DIR / 'logs'                      # Metrics and plots
```

**Effect**: Defines all data and output paths. Change if your directory structure differs.

---

### **Model Configuration**

```python
MODEL_NAME = 'convnextv2_large'  # Default model (used if TRAIN_ALL_ENSEMBLE=False)
PRETRAINED = True                # Use ImageNet pretrained weights
NUM_CLASSES = 13                 # Number of output classes (fixed)
```

**Effect:**
- `MODEL_NAME`: Only used if `TRAIN_ALL_ENSEMBLE=False`
- `PRETRAINED=True`: Uses ImageNet weights (recommended)
- `NUM_CLASSES=13`: Fixed for this competition

---

### **Workflow Control Flags**

```python
TRAIN_ALL_ENSEMBLE = True        # Auto-train all 3 ensemble models
RUN_PSEUDO_LABELING = True       # Generate pseudo-labels after Session 1
RETRAIN_WITH_PSEUDO = True       # Retrain best models with pseudo-labels
RUN_FINAL_SUBMISSION = True      # Generate final submission after Session 2
```

**Effect:**
- `TRAIN_ALL_ENSEMBLE=True`: Trains all 3 models automatically (Session 1)
- `RUN_PSEUDO_LABELING=True`: Enables Session 2
- `RETRAIN_WITH_PSEUDO=True`: Retrains top 2 models with pseudo-labels
- `RUN_FINAL_SUBMISSION=True`: Enables Session 3

**Recommendation**: Keep all `True` for complete automated workflow.

---

### **Session 2: Pseudo-Labeling**

```python
USE_PSEUDO_LABELS = True         # Auto-use pseudo-labels if available
PSEUDO_LABEL_THRESHOLD = 0.95    # Confidence threshold (0.0-1.0)
```

**Effect:**
- `USE_PSEUDO_LABELS=True`: Automatically uses `merged_train_with_pseudo.csv` if exists
- `PSEUDO_LABEL_THRESHOLD=0.95`: Only accepts predictions with 95%+ confidence (very strict)

**Trade-off**: Higher threshold = fewer but more accurate pseudo-labels

---

### **Session 3: Ensemble Optimization**

```python
ENSEMBLE_OPTIMIZATION = True     # Optimize ensemble weights
ENSEMBLE_OPT_METHOD = 'all'     # 'classy', 'weighted', 'equal', 'grid_search', 'all'
FINAL_SUBMISSION_VALIDATION = True  # Validate submission format
SUBMISSION_FILENAME = 'final_submission.csv'  # Output filename
```

**Effect:**
- `ENSEMBLE_OPTIMIZATION=True`: Tests multiple ensemble methods and picks best
- `ENSEMBLE_OPT_METHOD='all'`: Tests all methods (recommended)
- `FINAL_SUBMISSION_VALIDATION=True`: Validates format before saving (recommended)

---

### **Training Hyperparameters**

```python
BATCH_SIZE = 64                  # Batch size (default, model-specific overrides)
NUM_EPOCHS = 50                  # Maximum training epochs
LEARNING_RATE = 5e-5             # Default learning rate (model-specific overrides)
WEIGHT_DECAY = 5e-4              # L2 regularization strength
```

**Effect:**
- `BATCH_SIZE`: Larger = faster training but more memory. Model-specific settings override this.
- `NUM_EPOCHS`: Maximum epochs. Early stopping may stop earlier.
- `LEARNING_RATE`: Higher = faster convergence but may overshoot. Model-specific settings override.
- `WEIGHT_DECAY`: Higher = more regularization (prevents overfitting)

**Model-Specific Settings** (auto-applied):
```python
MODEL_SPECIFIC_SETTINGS = {
    'convnextv2_large': {
        'learning_rate': 5e-5,      # Optimized for Large models
        'batch_size': 64,            # Fits in ~24GB GPU
        'drop_path_rate': 0.3,       # Regularization
    },
    'swinv2_large_window12to16_192to256': {
        'learning_rate': 4e-5,
        'batch_size': 48,            # Smaller due to transformer memory
        'drop_path_rate': 0.2,
    },
    'maxvit_xlarge_tf_384': {
        'learning_rate': 3e-5,
        'batch_size': 32,            # Smallest due to XLarge size
        'drop_path_rate': 0.2,
    },
}
```

**Effect**: Each model gets optimized hyperparameters automatically when `TRAIN_ALL_ENSEMBLE=True`.

---

### **Data Loading**

```python
NUM_WORKERS = 8                  # DataLoader workers (parallel loading)
PIN_MEMORY = True                 # Faster GPU transfer
```

**Effect:**
- `NUM_WORKERS=8`: More workers = faster data loading (use CPU cores available)
- `PIN_MEMORY=True`: Faster CPU→GPU transfer (recommended if GPU available)

---

### **GPU Optimization**

```python
USE_MIXED_PRECISION = True       # Automatic Mixed Precision (AMP)
USE_TORCH_COMPILE = True         # PyTorch 2.0+ compilation
```

**Effect:**
- `USE_MIXED_PRECISION=True`: ~2x speedup, ~50% memory reduction (recommended)
- `USE_TORCH_COMPILE=True`: 20-30% speedup (requires PyTorch 2.0+)

**Recommendation**: Keep both `True` for maximum performance.

---

### **Learning Rate Scheduling**

```python
USE_WARMUP = True                 # Enable learning rate warmup
WARMUP_EPOCHS = 5                 # Warmup duration
USE_GRADIENT_CLIPPING = True      # Prevent gradient explosion
GRADIENT_CLIP_VALUE = 1.0         # Gradient clipping threshold
```

**Effect:**
- `USE_WARMUP=True`: Gradually increases LR at start (better convergence)
- `WARMUP_EPOCHS=5`: 5 epochs of warmup (longer = more stable)
- `USE_GRADIENT_CLIPPING=True`: Prevents exploding gradients (recommended)
- `GRADIENT_CLIP_VALUE=1.0`: Clips gradients at this value

**Schedule**: Linear warmup → Slow cosine annealing (optimized for validation learning)

---

### **Image Settings**

```python
IMG_SIZE = 512                   # Input image size (was 448, increased for accuracy)
MEAN = [0.485, 0.456, 0.406]     # ImageNet normalization mean
STD = [0.229, 0.224, 0.225]      # ImageNet normalization std
```

**Effect:**
- `IMG_SIZE=512`: Larger = better accuracy but more memory. 512px optimized for ensemble models.
- `MEAN/STD`: ImageNet normalization (required for pretrained models)

**Trade-off**: 512px gives better accuracy but uses 1.3x more memory than 448px.

---

### **Data Augmentation**

```python
USE_MIXUP = True                 # Enable Mixup augmentation
MIXUP_ALPHA = 0.2                # Mixup alpha (lower = less aggressive)
USE_CUTMIX = True                # Enable CutMix augmentation
CUTMIX_ALPHA = 1.0               # CutMix alpha
USE_AUTOAUGMENT = True           # Enable AutoAugment
MIXUP_CUTMIX_PROB = 0.2          # Probability of applying mixup/cutmix (20%)
```

**Effect:**
- `USE_MIXUP=True`: Mixes two images (reduces overfitting)
- `MIXUP_ALPHA=0.2`: Lower = less aggressive mixing (reduced from 0.4 for better validation)
- `USE_CUTMIX=True`: Cuts and pastes patches (complementary to Mixup)
- `MIXUP_CUTMIX_PROB=0.2`: 20% chance per batch (reduced from 0.3)

**Domain Adaptation Augmentation** (automatically enabled):
- Increased Gaussian noise (var_limit: 10-30)
- Increased Gaussian blur (blur_limit: 5-9)
- Motion blur (blur_limit: 7)
- Enhanced brightness/contrast (limits: 0.3)
- Enhanced color jittering (hue: 10, sat: 30, val: 30)

**Effect**: Simulates test-time degradation for better generalization.

---

### **Class-Aware Augmentation**

```python
USE_CLASS_AWARE_AUG = True       # Enable stronger augmentation for rare classes
RARE_CLASS_THRESHOLD = 0.1       # Classes with <10% of median are rare
```

**Effect:**
- `USE_CLASS_AWARE_AUG=True`: Rare classes get stronger augmentation
- `RARE_CLASS_THRESHOLD=0.1`: Classes with <10% of median samples are considered rare

**Benefit**: Helps rare classes (especially BL) by providing more diverse training samples.

---

### **Validation & Cross-Validation**

```python
VAL_SPLIT = 0.2                  # Validation split (not used if N_FOLDS > 1)
N_FOLDS = 5                       # Number of CV folds
RANDOM_SEED = 42                  # Reproducibility seed
```

**Effect:**
- `N_FOLDS=5`: 5-fold stratified cross-validation (ensures robust validation)
- `RANDOM_SEED=42`: Ensures reproducibility

---

### **Inference Settings**

```python
TTA = True                        # Enable Test Time Augmentation
TTA_NUM = 15                      # Number of TTA transforms (increased from 5)
```

**Effect:**
- `TTA=True`: Applies multiple augmentations during inference
- `TTA_NUM=15`: 15 different transforms (original, flips, rotations, brightness, noise, blur, color, CLAHE)

**Benefit**: Averages predictions across transforms for more robust results.

---

### **Ensemble Models**

```python
ENSEMBLE_MODELS = [
    'convnextv2_large',                    # Best ConvNeXt V2
    'swinv2_large_window12to16_192to256', # Best Swin V2 (optimized window)
    'maxvit_xlarge_tf_384',               # Best MaxViT
]
```

**Effect**: Defines which models to train when `TRAIN_ALL_ENSEMBLE=True`.

**Note**: Swin V2 window size optimized for 512px images (was window12to24, changed to window12to16).

---

### **Class Imbalance Handling**

```python
USE_CLASS_WEIGHTS = True         # Use weighted loss
USE_LABEL_SMOOTHING = True       # Enable label smoothing
LABEL_SMOOTHING = 0.1            # Smoothing factor (reduced from 0.15)
```

**Effect:**
- `USE_CLASS_WEIGHTS=True`: Weights loss by inverse class frequency (helps rare classes)
- `USE_LABEL_SMOOTHING=True`: Reduces overfitting
- `LABEL_SMOOTHING=0.1`: 10% smoothing (reduced from 0.15 for better learning)

**Additional**: Weighted random sampling also used for balanced batches.

---

### **Early Stopping**

```python
EARLY_STOPPING_PATIENCE = 10     # Epochs to wait before stopping
EARLY_STOPPING_MIN_DELTA = 0.001 # Minimum improvement required
```

**Effect:**
- `EARLY_STOPPING_PATIENCE=10`: Stops if no improvement for 10 epochs (reduced from 15)
- `EARLY_STOPPING_MIN_DELTA=0.001`: Requires 0.1% improvement (increased from 0.0005)

**Benefit**: Prevents overfitting, saves training time.

---

### **Multi-Scale Ensemble (Optional)**

```python
USE_MULTI_SCALE_ENSEMBLE = False # Enable multi-scale ensemble
MULTI_SCALE_SIZES = [448, 512, 576]  # Image sizes for multi-scale
```

**Effect:**
- `USE_MULTI_SCALE_ENSEMBLE=False`: Disabled by default (memory intensive)
- If enabled: Trains models at multiple image sizes and ensembles predictions

**Trade-off**: Better accuracy but 3x memory and training time.

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd White-Blood-Cell-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` - PyTorch (with CUDA if GPU available)
- `timm` - Model architectures
- `albumentations` - Data augmentation
- `pandas`, `numpy` - Data handling
- `scikit-learn` - Metrics and CV
- `tqdm` - Progress bars
- `matplotlib` - Plotting

### 3. Download Dataset

```bash
# If using gdown (Google Drive)
gdown 1nbRi3vuo414PigZGpNa6746t3Sy1EdZW -O dataset.zip
unzip dataset.zip

# Or download from Kaggle competition page
```

### 4. Verify Data Structure

Ensure you have:
- `phase1/` directory with images
- `phase2/train/`, `phase2/eval/`, `phase2/test/` directories
- `phase1_label.csv`, `phase2_train.csv`, `phase2_eval.csv`, `phase2_test.csv`

### 5. Run Training

```bash
python scripts/train.py
```

**That's it!** Everything happens automatically.

---

## 📊 Expected Performance

### Performance Targets

| Stage | Models | Expected Macro F1 | Time |
|-------|--------|-------------------|------|
| Session 1 | 15 models (3×5 folds) | 0.68-0.72 | 2-4 hours |
| Session 2 | +10 models (2×5 folds) | 0.72-0.75 | +1-2 hours |
| Session 3 | Final ensemble | 0.75-0.78 (eval) | +30-60 min |
| **Total** | **25 models** | **0.75-0.78** | **4-7 hours** |

### Per-Class Performance

- **Common Classes** (LY, SNE, MO): F1 > 0.85
- **Rare Classes** (BL, BA, PC): F1 > 0.50 (BL is critical)
- **Overall Macro F1**: 0.75-0.78 on eval set

### BL Class Monitoring

The solution includes special monitoring for BL (Blast cell) class:
- Per-fold BL F1 tracked in checkpoints
- Warnings if BL F1 < 0.5
- BL performance saved in all metrics

---

## 🛠️ Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions:**
1. Reduce `BATCH_SIZE` in `MODEL_SPECIFIC_SETTINGS`:
   ```python
   'convnextv2_large': {'batch_size': 32},  # Reduce from 64
   'swinv2_large_window12to16_192to256': {'batch_size': 24},  # Reduce from 48
   'maxvit_xlarge_tf_384': {'batch_size': 16},  # Reduce from 32
   ```

2. Reduce `IMG_SIZE` to 448:
   ```python
   IMG_SIZE = 448  # Reduce from 512
   ```

3. Disable `USE_TORCH_COMPILE`:
   ```python
   USE_TORCH_COMPILE = False
   ```

### Training Too Slow

**Solutions:**
1. Increase `NUM_WORKERS` (if CPU cores available):
   ```python
   NUM_WORKERS = 16  # Increase from 8
   ```

2. Ensure `USE_MIXED_PRECISION = True`
3. Ensure `USE_TORCH_COMPILE = True` (PyTorch 2.0+)

### Low BL Class Performance

**Symptoms**: BL F1 < 0.5

**Solutions:**
1. Ensure `USE_CLASS_AWARE_AUG = True`
2. Increase `RARE_CLASS_THRESHOLD` to include more classes:
   ```python
   RARE_CLASS_THRESHOLD = 0.15  # Increase from 0.1
   ```
3. Reduce `LABEL_SMOOTHING`:
   ```python
   LABEL_SMOOTHING = 0.05  # Reduce from 0.1
   ```

### Pseudo-Labeling Fails

**Symptoms**: No pseudo-labels generated

**Solutions:**
1. Check Session 1 models exist in `outputs/models/`
2. Verify `phase2_eval.csv` has labels
3. Lower `PSEUDO_LABEL_THRESHOLD`:
   ```python
   PSEUDO_LABEL_THRESHOLD = 0.90  # Reduce from 0.95
   ```

### Submission Validation Fails

**Symptoms**: Format errors

**Solutions:**
1. Run validation manually: `python scripts/validate_submission.py`
2. Check for duplicate IDs
3. Verify all 13 classes are present
4. Ensure CSV has exactly 2 columns: `ID, labels`

---

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
- One row per test image (16,477 rows)
- No header row needed if submitting via Kaggle interface

**Final Submission File:**
- `outputs/predictions/final_submission.csv` - Ready for upload!

---

## 🎓 Best Practices Implemented

- ✅ **Stratified K-Fold Cross-Validation** - Robust validation strategy
- ✅ **Class Weights** - Handles severe class imbalance
- ✅ **Advanced Data Augmentation** - Domain adaptation for test-time degradation
- ✅ **Class-Aware Augmentation** - Special handling for rare classes
- ✅ **Model Ensembling** - Combines diverse architectures
- ✅ **Classy Ensemble** - Per-class weighted ensemble (best for rare classes)
- ✅ **Test Time Augmentation** - 15 transforms for robust predictions
- ✅ **Early Stopping** - Prevents overfitting
- ✅ **Learning Rate Scheduling** - Warmup + cosine annealing
- ✅ **Mixed Precision Training** - 2x speedup, 50% memory reduction
- ✅ **PyTorch 2.0+ Compilation** - 20-30% additional speedup
- ✅ **Pseudo-Labeling** - Expands training data with high-confidence predictions
- ✅ **BL Class Monitoring** - Special tracking for critical rare class
- ✅ **Comprehensive Validation** - Catches errors before submission

---

## 📚 References

- **ConvNeXt V2**: [Paper](https://arxiv.org/abs/2301.00808)
- **Swin Transformer V2**: [Paper](https://arxiv.org/abs/2111.09883)
- **MaxViT**: [Paper](https://arxiv.org/abs/2204.01697)
- **Classy Ensemble**: [Paper](https://arxiv.org/abs/2302.10580)
- **Focal Loss**: [Paper](https://arxiv.org/abs/1708.02002)
- **Albumentations**: [Library](https://albumentations.ai/)

---

## 🤝 Contributing

This is a competition solution. Feel free to experiment and improve upon it!

---

## 📄 License

This solution is provided for educational purposes in the context of the Kaggle competition.

---

**Everything is automated. Just run `python scripts/train.py` and wait for your submission file!** 🚀
