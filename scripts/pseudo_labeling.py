"""
Pseudo-labeling script for WBC-Bench-2026
Generates high-confidence predictions on eval set to expand training data
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_val_transforms, get_tta_transforms
from scripts.inference import load_model, predict_ensemble

def generate_pseudo_labels(
    config,
    model_paths,
    confidence_threshold=0.95,
    min_samples_per_class=10,
    output_path=None
):
    """
    Generate pseudo-labels from eval set using ensemble predictions.
    
    Args:
        config: Config object
        model_paths: List of model checkpoint paths
        confidence_threshold: Minimum confidence to accept pseudo-label (0.0-1.0)
        min_samples_per_class: Minimum samples per class to include
        output_path: Path to save pseudo-labels CSV
    
    Returns:
        DataFrame with pseudo-labels
    """
    print(f"\n{'='*60}")
    print("Generating Pseudo-Labels")
    print(f"{'='*60}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Models: {len(model_paths)}")
    
    # Load eval data
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    eval_dataset = WBCDataset(
        csv_path=config.PHASE2_EVAL_CSV,
        img_dir=config.PHASE2_EVAL_DIR,
        transform=get_val_transforms(config.IMG_SIZE),
        is_train=False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms(config.IMG_SIZE) if config.TTA else None
    
    # Get ensemble predictions with probabilities
    print("\nGetting ensemble predictions...")
    preds, probs, names, idx_to_class = predict_ensemble(
        model_paths, eval_loader, config.DEVICE, 
        tta=config.TTA, tta_transforms=tta_transforms
    )
    
    # Convert to class labels
    pred_labels = [idx_to_class[pred] for pred in preds]
    
    # Get max probabilities (confidence)
    max_probs = np.max(probs, axis=1)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'ID': names,
        'labels': pred_labels,
        'confidence': max_probs
    })
    
    # Filter by confidence threshold
    high_confidence = results_df[results_df['confidence'] >= confidence_threshold].copy()
    
    print(f"\nTotal predictions: {len(results_df)}")
    print(f"High confidence (≥{confidence_threshold}): {len(high_confidence)}")
    print(f"Percentage: {100*len(high_confidence)/len(results_df):.2f}%")
    
    # Check class distribution
    class_counts = high_confidence['labels'].value_counts()
    print(f"\nHigh-confidence samples per class:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Filter by minimum samples per class
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    filtered = high_confidence[high_confidence['labels'].isin(valid_classes)].copy()
    
    print(f"\nAfter min_samples filter (≥{min_samples_per_class}): {len(filtered)}")
    
    # Save pseudo-labels
    if output_path is None:
        output_path = config.PRED_DIR / f'pseudo_labels_thresh{confidence_threshold}.csv'
    
    # Save only ID and labels (matching training CSV format)
    pseudo_df = filtered[['ID', 'labels']].copy()
    pseudo_df.to_csv(output_path, index=False)
    
    print(f"\nPseudo-labels saved to: {output_path}")
    print(f"Total pseudo-labeled samples: {len(pseudo_df)}")
    
    # Also save full results with confidence scores
    full_output_path = config.PRED_DIR / f'pseudo_labels_full_thresh{confidence_threshold}.csv'
    filtered.to_csv(full_output_path, index=False)
    print(f"Full results (with confidence) saved to: {full_output_path}")
    
    return pseudo_df, filtered

def merge_pseudo_labels_with_training(
    config,
    pseudo_labels_path,
    output_path=None
):
    """
    Merge pseudo-labels with original training data.
    
    Args:
        config: Config object
        pseudo_labels_path: Path to pseudo-labels CSV
        output_path: Path to save merged CSV
    
    Returns:
        Merged DataFrame
    """
    print(f"\n{'='*60}")
    print("Merging Pseudo-Labels with Training Data")
    print(f"{'='*60}")
    
    # Load original training data
    phase1 = pd.read_csv(config.PHASE1_CSV)
    phase1 = phase1[phase1['split'] == 'phase1_train']
    phase1 = phase1[['ID', 'labels']].copy()
    
    phase2_train = pd.read_csv(config.PHASE2_TRAIN_CSV)
    phase2_train = phase2_train[['ID', 'labels']].copy()
    
    # CRITICAL: Include Phase 2 Eval in original training (as per Session 1 fix)
    phase2_eval = pd.read_csv(config.PHASE2_EVAL_CSV)
    phase2_eval = phase2_eval[['ID', 'labels']].copy()
    
    # Load pseudo-labels
    pseudo_labels = pd.read_csv(pseudo_labels_path)
    
    # Combine all training data
    all_train = pd.concat([
        phase1,
        phase2_train,
        phase2_eval,  # Original eval set (with labels)
        pseudo_labels  # Pseudo-labels from eval set (high confidence predictions)
    ], ignore_index=True)
    
    print(f"Original Phase 1: {len(phase1)} samples")
    print(f"Original Phase 2 Train: {len(phase2_train)} samples")
    print(f"Original Phase 2 Eval: {len(phase2_eval)} samples")
    print(f"Pseudo-labels: {len(pseudo_labels)} samples")
    print(f"Total merged: {len(all_train)} samples")
    
    # Check for duplicates (same ID in multiple sources)
    duplicates = all_train[all_train.duplicated(subset=['ID'], keep=False)]
    if len(duplicates) > 0:
        print(f"\nWARNING: {len(duplicates)} duplicate IDs found!")
        print("Removing duplicates: keeping original training labels first, then pseudo-labels")
        # Priority: phase1 > phase2_train > phase2_eval > pseudo_labels
        # Mark source
        phase1['source'] = 1
        phase2_train['source'] = 2
        phase2_eval['source'] = 3
        pseudo_labels['source'] = 4
        
        all_train_with_source = pd.concat([
            phase1[['ID', 'labels', 'source']],
            phase2_train[['ID', 'labels', 'source']],
            phase2_eval[['ID', 'labels', 'source']],
            pseudo_labels[['ID', 'labels', 'source']]
        ], ignore_index=True)
        
        # Keep first occurrence (lowest source number = highest priority)
        all_train = all_train_with_source.sort_values('source').drop_duplicates(subset=['ID'], keep='first')
        all_train = all_train[['ID', 'labels']].copy()
        print(f"After deduplication: {len(all_train)} samples")
    else:
        all_train = all_train[['ID', 'labels']].copy()
    
    # Save merged data
    if output_path is None:
        output_path = config.PRED_DIR / 'merged_train_with_pseudo.csv'
    
    all_train.to_csv(output_path, index=False)
    print(f"\nMerged training data saved to: {output_path}")
    
    # Show class distribution
    print(f"\nClass distribution in merged data:")
    print(all_train['labels'].value_counts().sort_index())
    
    return all_train

def main():
    """Main pseudo-labeling function"""
    config = Config()
    
    # Find all trained models
    model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
    
    if not model_paths:
        print("ERROR: No trained models found. Please train models first.")
        print("Run: python scripts/train.py")
        return
    
    print(f"Found {len(model_paths)} models")
    
    # Generate pseudo-labels with high confidence threshold
    confidence_threshold = 0.95  # Very high confidence
    pseudo_df, full_results = generate_pseudo_labels(
        config,
        model_paths,
        confidence_threshold=confidence_threshold,
        min_samples_per_class=10
    )
    
    # Merge with training data
    pseudo_path = config.PRED_DIR / f'pseudo_labels_thresh{confidence_threshold}.csv'
    merged_train = merge_pseudo_labels_with_training(
        config,
        pseudo_path
    )
    
    print(f"\n{'='*60}")
    print("Pseudo-Labeling Complete!")
    print(f"{'='*60}")
    print(f"Use merged_train_with_pseudo.csv for training with pseudo-labels")
    print(f"Total training samples: {len(merged_train)}")

if __name__ == '__main__':
    main()

