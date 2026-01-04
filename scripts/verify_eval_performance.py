"""
Script to verify eval set performance after training
Compares CV performance with eval set performance to ensure consistency
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from scripts.inference import predict_eval_set

def main():
    """Verify eval set performance matches CV expectations"""
    config = Config()
    
    print("="*60)
    print("Eval Set Performance Verification")
    print("="*60)
    
    # Load CV results
    metrics_dir = config.LOG_DIR / 'metrics'
    cv_scores = []
    model_name = config.MODEL_NAME
    
    print(f"\nLoading CV results for {model_name}...")
    for fold in range(config.N_FOLDS):
        summary_file = metrics_dir / f'{model_name}_fold{fold}_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                cv_scores.append(data['best_val_macro_f1'])
                print(f"  Fold {fold}: {data['best_val_macro_f1']:.4f}")
    
    if not cv_scores:
        print(f"ERROR: No CV results found for {model_name}")
        print("Please train models first using: python scripts/train.py")
        return
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCV Mean Macro-F1: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Get predictions on eval set
    print(f"\n{'='*60}")
    print("Predicting on Eval Set...")
    print("="*60)
    
    # Find all trained models
    model_paths = list(config.MODEL_DIR.glob(f'{model_name}_*_best.pth'))
    
    if not model_paths:
        print(f"ERROR: No trained models found for {model_name}")
        return
    
    print(f"Found {len(model_paths)} models")
    
    # Predict on eval set
    eval_submission = predict_eval_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=True
    )
    
    # Load ground truth
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    
    # Merge predictions with ground truth
    merged = eval_submission.merge(eval_df, on='ID', how='inner')
    y_true = merged['labels_y'].values
    y_pred = merged['labels_x'].values
    
    # Compute metrics
    eval_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    eval_acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("Eval Set Performance")
    print("="*60)
    print(f"Macro-F1: {eval_macro_f1:.4f}")
    print(f"Accuracy: {eval_acc:.4f}")
    
    # Compare with CV
    print(f"\n{'='*60}")
    print("Comparison: CV vs Eval Set")
    print("="*60)
    print(f"CV Mean Macro-F1:     {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Eval Set Macro-F1:    {eval_macro_f1:.4f}")
    print(f"Difference:           {eval_macro_f1 - cv_mean:.4f}")
    
    if abs(eval_macro_f1 - cv_mean) < 0.02:
        print("\n✓ GOOD: Eval performance matches CV (within 2%)")
    elif eval_macro_f1 < cv_mean - 0.05:
        print("\n⚠ WARNING: Eval performance significantly lower than CV")
        print("  Possible causes: Overfitting, domain shift, or data leakage")
    else:
        print("\n✓ EXCELLENT: Eval performance matches or exceeds CV")
    
    # Per-class analysis (especially BL)
    print(f"\n{'='*60}")
    print("Per-Class F1 Scores (Eval Set)")
    print("="*60)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Check BL class specifically
    bl_f1 = f1_score(
        [1 if label == 'BL' else 0 for label in y_true],
        [1 if label == 'BL' else 0 for label in y_pred],
        zero_division=0
    )
    bl_count = sum(1 for label in y_true if label == 'BL')
    print(f"\n{'='*60}")
    print(f"BL (Blast) Class - CRITICAL RARE CLASS")
    print("="*60)
    print(f"BL F1 Score: {bl_f1:.4f}")
    print(f"BL Samples: {bl_count}")
    if bl_f1 < 0.5:
        print("⚠ WARNING: BL F1 is below 0.5 - needs improvement!")
    elif bl_f1 < 0.7:
        print("⚠ CAUTION: BL F1 is below 0.7 - consider improvements")
    else:
        print("✓ GOOD: BL F1 is above 0.7")
    
    # Save results
    results = {
        'model_name': model_name,
        'cv_mean_macro_f1': float(cv_mean),
        'cv_std_macro_f1': float(cv_std),
        'eval_macro_f1': float(eval_macro_f1),
        'eval_accuracy': float(eval_acc),
        'bl_f1': float(bl_f1),
        'bl_count': int(bl_count),
        'difference': float(eval_macro_f1 - cv_mean)
    }
    
    results_path = config.LOG_DIR / f'{model_name}_eval_verification.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()

