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
    MODEL_NAME = 'convnextv2_large'  # Optimized: ConvNeXt V2 Large (strongest), or change to 'swinv2_base_window8_256' / 'maxvit_base_tf_384' for other models
    PRETRAINED = True
    NUM_CLASSES = 13  # Will be updated based on actual classes
    
    # Auto-train all ensemble models
    TRAIN_ALL_ENSEMBLE = True  # If True, trains all ENSEMBLE_MODELS sequentially with optimized settings
    
    # Session 2: Pseudo-labeling and optimization
    USE_PSEUDO_LABELS = True  # If True, automatically uses pseudo-labels if available (merged_train_with_pseudo.csv)
    PSEUDO_LABEL_THRESHOLD = 0.95  # Confidence threshold for pseudo-labels
    RUN_PSEUDO_LABELING = False  # Skip pseudo-labeling to save time/credits
    RETRAIN_WITH_PSEUDO = False  # Skip retraining with pseudo-labels
    
    # Session 3: Final ensemble optimization and submission
    ENSEMBLE_OPTIMIZATION = True  # If True, optimizes ensemble weights before final submission
    ENSEMBLE_OPT_METHOD = 'all'  # 'classy', 'weighted', 'equal', 'grid_search', or 'all' (tests all methods)
    FINAL_SUBMISSION_VALIDATION = True  # If True, validates submission format before saving
    RUN_FINAL_SUBMISSION = True  # If True, generates final submission after Session 2
    
    # Multi-scale ensemble (optional, if memory allows)
    USE_MULTI_SCALE_ENSEMBLE = False  # If True, ensembles predictions from multiple image sizes
    MULTI_SCALE_SIZES = [448, 512, 576]  # Image sizes for multi-scale ensemble
    
    # Final submission settings
    SUBMISSION_FILENAME = 'final_submission.csv'  # Final submission filename
    SAVE_PREDICTION_PROBABILITIES = True  # Save probability distributions for analysis
    
    # Training settings
    # Note: For best models (V2 Large, Swin V2 Large, MaxViT XLarge), reduce BATCH_SIZE if OOM
    # Recommended: V2 Large=64, Swin V2 Large=48, MaxViT XLarge=32 (adjust based on GPU memory)
    BATCH_SIZE = 64  # Reduced for best models - increase to 96-128 for smaller models if memory allows
    INFER_BATCH_SIZE = 16  # Smaller batch for inference/ensembles to avoid GPU indexing limits
    NUM_EPOCHS = 50
    # FIXED: Learning rate tuned for Large models (was 3e-5, too low for Large/XLarge)
    # Large models need slightly higher LR: 5e-5 for V2 Large, 4e-5 for Swin V2 Large, 3e-5 for MaxViT XLarge
    LEARNING_RATE = 5e-5  # Optimized for Large models - will be auto-adjusted per model if TRAIN_ALL_ENSEMBLE=True
    
    # Model-specific settings (auto-applied when TRAIN_ALL_ENSEMBLE=True)
    # Can be updated after hyperparameter optimization
    MODEL_SPECIFIC_SETTINGS = {
        'convnextv2_large': {
            'learning_rate': 5e-5,
            'batch_size': 64,
            'drop_path_rate': 0.3,  # Can be optimized via hyperparameter_opt.py
        },
        'swinv2_large_window12to16_192to256': {
            'learning_rate': 4e-5,
            'batch_size': 48,
            'drop_path_rate': 0.2,  # Can be optimized via hyperparameter_opt.py
            'img_size': 256,  # Swin V2 Large expects 256x256 inputs
        },
        'maxvit_xlarge_tf_384': {
            'learning_rate': 3e-5,
            'batch_size': 32,
            'drop_path_rate': 0.2,  # Can be optimized via hyperparameter_opt.py
            'img_size': 384,  # MaxViT XLarge requires 384x384 images
        },
    }
    WEIGHT_DECAY = 5e-4  # Reduced from 1e-3 - too strong regularization causing underfitting
    NUM_WORKERS = 8  # Increased for faster data loading (was 4)
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4  # Number of batches to prefetch per worker (4-8 optimal, default=2)
    PERSISTENT_WORKERS = True  # Keep workers alive across epochs (faster epoch 2+)
    
    # GPU Optimization
    USE_MIXED_PRECISION = True  # Automatic Mixed Precision (AMP) - 2x speedup
    USE_TORCH_COMPILE = True  # PyTorch 2.0+ compilation - 20-30% speedup
    
    # Learning rate scheduling
    USE_WARMUP = True  # Warmup for better convergence
    WARMUP_EPOCHS = 5  # Increased from 3 - slower warmup for better validation learning
    USE_GRADIENT_CLIPPING = True  # Prevent gradient explosion
    GRADIENT_CLIP_VALUE = 1.0  # Clip gradients at this value
    
    # Image settings
    IMG_SIZE = 512  # Increased for better accuracy (was 448)
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
    TTA_NUM = 15  # Increased from 5 for better predictions
    
    # Ensemble (BEST AVAILABLE: 3 diverse architectures, maximum performance)
    # Strategy: V2 Large (strongest ConvNeXt) + Swin V2 Large (best transformer) + MaxViT XLarge (best multi-axis)
    # FIXED: Swin V2 window size optimized for 512 image size
    ENSEMBLE_MODELS = [
        'convnextv2_large',                    # BEST ConvNeXt V2 - strongest CNN architecture
        'maxvit_xlarge_tf_384',               # BEST MaxViT - XLarge variant (maximum performance)
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

