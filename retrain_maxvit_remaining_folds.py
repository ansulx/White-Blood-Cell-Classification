"""
Train only the remaining MaxViT XLarge folds.
Run: python retrain_maxvit_remaining_folds.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from scripts.train import train_fold, is_checkpoint_valid


def main():
    config = Config()
    config.DISTRIBUTED = False
    config.RANK = 0
    config.WORLD_SIZE = 1
    config.LOCAL_RANK = 0

    # Set model to MaxViT XLarge
    config.MODEL_NAME = "maxvit_xlarge_tf_384"
    print(f"\n{'='*60}")
    print(f"TRAINING REMAINING FOLDS - {config.MODEL_NAME}")
    print(f"{'='*60}\n")

    # Apply model-specific settings (no parameter changes)
    if config.MODEL_NAME in config.MODEL_SPECIFIC_SETTINGS:
        settings = config.MODEL_SPECIFIC_SETTINGS[config.MODEL_NAME]
        config.LEARNING_RATE = settings.get("learning_rate", config.LEARNING_RATE)
        config.BATCH_SIZE = settings.get("batch_size", config.BATCH_SIZE)
        if "img_size" in settings:
            config.IMG_SIZE = settings.get("img_size", config.IMG_SIZE)
        print("Using model-specific settings:")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Image Size: {config.IMG_SIZE}\n")

    # Find remaining folds by valid checkpoints
    remaining_folds = []
    for fold in range(config.N_FOLDS):
        model_path = config.MODEL_DIR / f"{config.MODEL_NAME}_fold{fold}_best.pth"
        if not is_checkpoint_valid(model_path):
            remaining_folds.append(fold)

    if not remaining_folds:
        print("All folds already have valid checkpoints. Nothing to train.")
        return

    print(f"Remaining folds to train: {remaining_folds}\n")

    # Load data (same logic as train_single_model)
    print("Loading data...")
    merged_train_path = config.PRED_DIR / "merged_train_with_pseudo.csv"
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
        phase1 = phase1[phase1["split"] == "phase1_train"]
        phase1 = phase1[["ID", "labels"]].copy()

        all_train = pd.concat(
            [
                phase2_train[["ID", "labels"]],
                phase1[["ID", "labels"]],
                phase2_eval[["ID", "labels"]],
            ],
            ignore_index=True,
        )

    print(f"Total training samples: {len(all_train)}")
    print(f"Class distribution:\n{all_train['labels'].value_counts()}\n")

    # Create StratifiedKFold splitter
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    splits = list(skf.split(all_train, all_train["labels"]))

    # Train only remaining folds
    for fold_num in remaining_folds:
        train_idx, val_idx = splits[fold_num]
        train_df = all_train.iloc[train_idx].reset_index(drop=True)
        val_df = all_train.iloc[val_idx].reset_index(drop=True)

        print(f"\nStarting training for fold {fold_num}...")
        val_macro_f1, _ = train_fold(fold_num, train_df, val_df, config)
        print(f"Fold {fold_num} complete. Val Macro-F1: {val_macro_f1:.4f}")

    print(f"\n{'='*60}")
    print("REMAINING FOLDS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
