"""
Optimized script to generate submission using convnextv2_large models with ensemble weight optimization.
Uses eval set to find best ensemble weights (equal, weighted, or classy), then generates optimized submission.
Results saved in 'convnextv2_large only results' folder.
This takes ~30-60 minutes but gives better results (expected +0.5-1% improvement).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_val_transforms, get_tta_transforms
from scripts.inference import load_model, predict_ensemble, predict_ensemble_classy, predict_ensemble_optimized

def main():
    """Generate optimized submission using convnextv2_large models"""
    print("="*70)
    print("ConvNeXt V2 Large OPTIMIZED 5-Fold Ensemble Submission Generator")
    print("="*70)
    
    config = Config()
    torch.backends.cudnn.benchmark = True
    
    # Create output directory
    output_dir = Path('convnextv2_large only results')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Find all convnextv2_large fold models
    model_paths = []
    for fold in range(5):
        model_path = config.MODEL_DIR / f'convnextv2_large_fold{fold}_best.pth'
        if model_path.exists():
            model_paths.append(model_path)
            print(f"  Found: {model_path.name}")
        else:
            print(f"  Warning: {model_path.name} not found")
    
    if len(model_paths) == 0:
        print("\nERROR: No convnextv2_large models found!")
        return
    
    print(f"\nUsing {len(model_paths)} models for ensemble")
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms(config.IMG_SIZE)
    print(f"\nTTA transforms: {len(tta_transforms)}")
    
    # STEP 1: Optimize ensemble weights on eval set
    print("\n" + "="*70)
    print("STEP 1: Optimizing Ensemble Weights on Eval Set")
    print("="*70)
    
    # Load eval data
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    eval_labels = eval_df['labels'].values.tolist()
    
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
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
    )
    
    # Get class names
    temp_model, idx_to_class = load_model(model_paths[0], config.DEVICE)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    del temp_model
    torch.cuda.empty_cache()
    
    # Test different ensemble methods
    from sklearn.metrics import f1_score, accuracy_score
    
    best_f1 = 0.0
    best_method = 'equal'
    best_weights = np.ones(len(model_paths)) / len(model_paths)
    
    # Method 1: Equal weights
    print("\n1. Testing Equal Weights (baseline)...")
    preds, probs, names, idx_to_class = predict_ensemble(
        model_paths, eval_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms
    )
    pred_labels = [idx_to_class[p] for p in preds]
    f1_equal = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
    acc_equal = accuracy_score(eval_labels, pred_labels)
    print(f"   Equal Weights - Macro F1: {f1_equal:.5f}, Acc: {acc_equal:.4f}")
    
    if f1_equal > best_f1:
        best_f1 = f1_equal
        best_method = 'equal'
        best_weights = np.ones(len(model_paths)) / len(model_paths)
    
    # Method 2: Weighted by individual model F1
    print("\n2. Testing Weighted Ensemble (by individual F1)...")
    model_f1s = []
    for i, model_path in enumerate(model_paths):
        print(f"   Evaluating model {i+1}/{len(model_paths)}...")
        preds, probs, names = predict_ensemble(
            [model_path], eval_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms
        )
        pred_labels = [idx_to_class[p] for p in preds]
        f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
        model_f1s.append(f1)
        print(f"     Model {i+1} Macro F1: {f1:.5f}")
    
    # Normalize weights
    model_f1s = np.array(model_f1s)
    model_f1s = model_f1s + 1e-8  # Avoid division by zero
    weighted_weights = model_f1s / model_f1s.sum()
    
    # Evaluate weighted ensemble
    preds, probs, names, idx_to_class = predict_ensemble_optimized(
        model_paths, eval_loader, config.DEVICE,
        weights=weighted_weights,
        tta=True, tta_transforms=tta_transforms
    )
    pred_labels = [idx_to_class[p] for p in preds]
    f1_weighted = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
    acc_weighted = accuracy_score(eval_labels, pred_labels)
    print(f"   Weighted Ensemble - Macro F1: {f1_weighted:.5f}, Acc: {acc_weighted:.4f}")
    print(f"   Weights: {weighted_weights}")
    
    if f1_weighted > best_f1:
        best_f1 = f1_weighted
        best_method = 'weighted'
        best_weights = weighted_weights
    
    # Method 3: Classy Ensemble (per-class weights)
    print("\n3. Testing Classy Ensemble (per-class weights)...")
    try:
        preds, probs, names, idx_to_class = predict_ensemble_classy(
            model_paths, eval_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms,
            eval_labels=eval_labels, class_names=class_names
        )
        pred_labels = [idx_to_class[p] for p in preds]
        f1_classy = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
        acc_classy = accuracy_score(eval_labels, pred_labels)
        print(f"   Classy Ensemble - Macro F1: {f1_classy:.5f}, Acc: {acc_classy:.4f}")
        
        if f1_classy > best_f1:
            best_f1 = f1_classy
            best_method = 'classy'
            best_weights = 'classy'  # Special marker
    except Exception as e:
        print(f"   Classy Ensemble failed: {e}")
        print("   Continuing with weighted ensemble...")

    # Method 4: Geometric Mean Ensemble (often improves calibration)
    print("\n4. Testing Geometric Mean Ensemble (probability fusion)...")
    try:
        all_model_probs = []
        for i, model_path in enumerate(model_paths):
            print(f"   Evaluating model {i+1}/{len(model_paths)}...")
            preds, probs, names, idx_to_class = predict_ensemble(
                [model_path], eval_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms
            )
            all_model_probs.append(probs)
        probs_geo = np.exp(np.mean([np.log(p + 1e-8) for p in all_model_probs], axis=0))
        probs_geo = probs_geo / probs_geo.sum(axis=1, keepdims=True)
        preds_geo = np.argmax(probs_geo, axis=1)
        pred_labels_geo = [idx_to_class[p] for p in preds_geo]
        f1_geo = f1_score(eval_labels, pred_labels_geo, average='macro', zero_division=0)
        acc_geo = accuracy_score(eval_labels, pred_labels_geo)
        print(f"   Geometric Mean - Macro F1: {f1_geo:.5f}, Acc: {acc_geo:.4f}")
        if f1_geo > best_f1:
            best_f1 = f1_geo
            best_method = 'geometric_mean'
            best_weights = np.ones(len(model_paths)) / len(model_paths)
    except Exception as e:
        print(f"   Geometric mean failed: {e}")
    
    print("\n" + "="*70)
    print(f"Best Method: {best_method.upper()}")
    print(f"Best Eval F1: {best_f1:.5f}")
    if best_method != 'classy':
        print(f"Best Weights: {best_weights}")
    print("="*70)
    
    # STEP 2: Generate optimized test predictions
    print("\n" + "="*70)
    print("STEP 2: Generating Optimized Test Predictions")
    print("="*70)
    
    # Load test data
    test_df = pd.read_csv(config.PHASE2_TEST_CSV)
    test_dataset = WBCDataset(
        csv_path=config.PHASE2_TEST_CSV,
        img_dir=config.PHASE2_TEST_DIR,
        transform=get_val_transforms(config.IMG_SIZE),
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
    )
    
    # Generate predictions with best method
    if best_method == 'classy':
        # Classy Ensemble (doesn't need eval_labels for test set - uses pre-computed weights)
        preds, probs, names, idx_to_class = predict_ensemble_classy(
            model_paths, test_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms,
            eval_labels=None, class_names=class_names
        )
    elif best_method == 'weighted':
        # Weighted ensemble
        preds, probs, names, idx_to_class = predict_ensemble_optimized(
            model_paths, test_loader, config.DEVICE,
            weights=best_weights,
            tta=True, tta_transforms=tta_transforms
        )
    elif best_method == 'geometric_mean':
        # Geometric mean ensemble
        all_model_probs = []
        for i, model_path in enumerate(model_paths):
            preds_i, probs_i, names_i, idx_to_class = predict_ensemble(
                [model_path], test_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms
            )
            all_model_probs.append(probs_i)
            if i == 0:
                names = names_i
        probs = np.exp(np.mean([np.log(p + 1e-8) for p in all_model_probs], axis=0))
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
    else:
        # Equal weights
        preds, probs, names, idx_to_class = predict_ensemble(
            model_paths, test_loader, config.DEVICE, tta=True, tta_transforms=tta_transforms
        )
    
    # Convert predictions to class labels
    pred_labels = [idx_to_class[pred] for pred in preds]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': names,
        'labels': pred_labels
    })
    
    # Save submission
    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n✓ Submission saved to: {submission_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Submission Summary")
    print("="*70)
    print(f"Ensemble Method: {best_method.upper()}")
    print(f"Expected Eval F1: {best_f1:.5f}")
    print(f"Total predictions: {len(submission)}")
    print(f"Unique IDs: {submission['ID'].nunique()}")
    print(f"Unique labels: {submission['labels'].nunique()}")
    print(f"\nClass distribution:")
    print(submission['labels'].value_counts().sort_index())
    
    # Save probabilities
    prob_df = pd.DataFrame(probs, columns=[idx_to_class[i] for i in range(len(idx_to_class))])
    prob_df['ID'] = names
    prob_df = prob_df[['ID'] + [idx_to_class[i] for i in range(len(idx_to_class))]]
    prob_path = output_dir / 'prediction_probabilities.csv'
    prob_df.to_csv(prob_path, index=False)
    print(f"\n✓ Prediction probabilities saved to: {prob_path}")
    
    # Save optimization results
    opt_results = {
        'best_method': best_method,
        'best_eval_f1': float(best_f1),
        'weights': best_weights.tolist() if best_method != 'classy' else 'classy (per-class)',
        'model_f1s': model_f1s.tolist() if 'model_f1s' in locals() else None
    }
    opt_path = output_dir / 'optimization_results.json'
    with open(opt_path, 'w') as f:
        json.dump(opt_results, f, indent=2)
    print(f"✓ Optimization results saved to: {opt_path}")
    
    print("\n" + "="*70)
    print("Done! Optimized results saved in 'convnextv2_large only results' folder")
    print("="*70)

if __name__ == '__main__':
    main()
