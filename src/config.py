"""
Configuration file for WBC-Bench-2026 competition
"""

import os
from pathlib import Path
import torch

class Config:
    # Paths - BASE_DIR is project root (parent of src/)
    BASE_DIR = Path(__file__).parent.parent
    PHASE1_DIR = BASE_DIR / 'phase1'
    PHASE2_TRAIN_DIR = BASE_DIR / 'phase2' / 'train'
    PHASE2_TEST_DIR = BASE_DIR / 'phase2' / 'test'
    PHASE2_EVAL_DIR = BASE_DIR / 'phase2' / 'eval'
    
    PHASE1_CSV = BASE_DIR / 'phase1_label.csv'
    PHASE2_TRAIN_CSV = BASE_DIR / 'phase2_train.csv'
    PHASE2_TEST_CSV = BASE_DIR / 'phase2_test.csv'
    PHASE2_EVAL_CSV = BASE_DIR / 'phase2_eval.csv'
    
    OUTPUT_DIR = BASE_DIR / 'outputs'
    MODEL_DIR = OUTPUT_DIR / 'models'
    PRED_DIR = OUTPUT_DIR / 'predictions'
    LOG_DIR = OUTPUT_DIR / 'logs'
    
    # Model settings
    MODEL_NAME = 'convnext_base'  # ConvNeXt-Base: Modern CNN architecture, better than EfficientNet
    PRETRAINED = True
    NUM_CLASSES = 13  # Will be updated based on actual classes
    
    # Training settings
    BATCH_SIZE = 128  # Optimized for H200 (141GB memory) - was 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5  # Further reduced from 5e-5 - validation loss still too high (5.4)
    WEIGHT_DECAY = 5e-4  # Reduced from 1e-3 - too strong regularization causing underfitting
    NUM_WORKERS = 8  # Increased for faster data loading (was 4)
    PIN_MEMORY = True
    
    # GPU Optimization
    USE_MIXED_PRECISION = True  # Automatic Mixed Precision (AMP) - 2x speedup
    USE_TORCH_COMPILE = True  # PyTorch 2.0+ compilation - 20-30% speedup
    
    # Learning rate scheduling
    USE_WARMUP = True  # Warmup for better convergence
    WARMUP_EPOCHS = 5  # Increased from 3 - slower warmup for better validation learning
    USE_GRADIENT_CLIPPING = True  # Prevent gradient explosion
    GRADIENT_CLIP_VALUE = 1.0  # Clip gradients at this value
    
    # Image settings
    IMG_SIZE = 448  # Increased from 384 for better accuracy (use 512 if memory allows)
    MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    STD = [0.229, 0.224, 0.225]
    
    # Data augmentation (FURTHER REDUCED to improve validation)
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2  # Reduced from 0.4 - less aggressive mixing
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    USE_AUTOAUGMENT = True
    MIXUP_CUTMIX_PROB = 0.2  # Further reduced from 0.3 to 0.2 - less frequent mixing (20% chance)
    
    # Class-aware augmentation (for rare classes)
    USE_CLASS_AWARE_AUG = True  # Enable stronger augmentation for rare classes
    RARE_CLASS_THRESHOLD = 0.1  # Classes with <10% of median samples are considered rare
    
    # Validation
    VAL_SPLIT = 0.2
    N_FOLDS = 5  # For cross-validation
    RANDOM_SEED = 42
    
    # Inference
    TTA = True  # Test Time Augmentation
    TTA_NUM = 5  # Number of TTA iterations
    
    # Ensemble (ordered by priority)
    ENSEMBLE_MODELS = [
        'convnext_base',  # Primary: Best architecture
        'efficientnet_b5',  # Secondary: Backup
        # 'swin_base_patch4_window7_224',  # Optional: Transformer variant
    ]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class weights (for handling imbalance)
    USE_CLASS_WEIGHTS = True
    
    # Label smoothing (reduces overfitting)
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1  # Reduced from 0.15 - too much smoothing hurting learning
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10  # Reduced from 15 - stop earlier if not improving
    EARLY_STOPPING_MIN_DELTA = 0.001  # Increased from 0.0005 - require meaningful improvement
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / 'metrics').mkdir(exist_ok=True)
    (LOG_DIR / 'plots').mkdir(exist_ok=True)

