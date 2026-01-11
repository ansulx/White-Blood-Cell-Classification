"""
Quick script to retrain just fold 4 of convnextv2_large
Run this on RunPod: python retrain_fold4.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from scripts.train import train_fold

def main():
    config = Config()
    
    # Set model to convnextv2_large
    config.MODEL_NAME = 'convnextv2_large'
    print(f"\n{'='*60}")
    print(f"RETRAINING FOLD 4 ONLY - {config.MODEL_NAME}")
    print(f"{'='*60}\n")
    
    # Apply model-specific settings
    if config.MODEL_NAME in config.MODEL_SPECIFIC_SETTINGS:
        settings = config.MODEL_SPECIFIC_SETTINGS[config.MODEL_NAME]
        config.LEARNING_RATE = settings.get('learning_rate', config.LEARNING_RATE)
        config.BATCH_SIZE = settings.get('batch_size', config.BATCH_SIZE)
        print(f"Using model-specific settings:")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
        print(f"  Batch Size: {config.BATCH_SIZE}\n")
    
    # Load data (same logic as train_single_model)
    print("Loading data...")
    
    # Check if pseudo-labels exist
    merged_train_path = config.PRED_DIR / 'merged_train_with_pseudo.csv'
    use_pseudo_labels = merged_train_path.exists()
    
    if use_pseudo_labels:
        print("✓ Found merged training data with pseudo-labels")
        all_train = pd.read_csv(merged_train_path)
        print(f"  Total samples (including pseudo-labels): {len(all_train)}")
    else:
        print("Using original training data only")
        phase2_train = pd.read_csv(config.PHASE2_TRAIN_CSV)
        phase2_eval = pd.read_csv(config.PHASE2_EVAL_CSV)
        print(f"Including Phase 2 Eval set: {len(phase2_eval)} additional samples")
        
        phase1 = pd.read_csv(config.PHASE1_CSV)
        phase1 = phase1[phase1['split'] == 'phase1_train']
        phase1 = phase1[['ID', 'labels']].copy()
        
        all_train = pd.concat([
            phase2_train[['ID', 'labels']],
            phase1[['ID', 'labels']],
            phase2_eval[['ID', 'labels']]
        ], ignore_index=True)
    
    print(f"Total training samples: {len(all_train)}")
    print(f"Class distribution:\n{all_train['labels'].value_counts()}\n")
    
    # Create StratifiedKFold splitter (same settings as main training)
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )
    
    # Get fold 4's train/val split
    print("Generating fold splits...")
    splits = list(skf.split(all_train, all_train['labels']))
    
    # Get fold 4 (index 4)
    fold_num = 4
    train_idx, val_idx = splits[fold_num]
    train_df = all_train.iloc[train_idx].reset_index(drop=True)
    val_df = all_train.iloc[val_idx].reset_index(drop=True)
    
    print(f"Fold {fold_num} split:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}\n")
    
    # Train fold 4
    print(f"Starting training for fold {fold_num}...")
    val_macro_f1, val_metrics = train_fold(fold_num, train_df, val_df, config)
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num} COMPLETE!")
    print(f"Val Macro-F1: {val_macro_f1:.4f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
