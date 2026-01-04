"""
Final Submission Preparation for WBC-Bench-2026
Validates predictions, ensures correct format, and prepares final submission
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from scripts.inference import predict_test_set, predict_eval_set
from scripts.ensemble_optimizer import optimize_ensemble_weights

def validate_submission_format(submission_df, test_df=None):
    """
    Validate submission format matches competition requirements.
    
    Args:
        submission_df: DataFrame with predictions
        test_df: Original test CSV (optional, for ID validation)
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required columns
    required_cols = ['ID', 'labels']
    if not all(col in submission_df.columns for col in required_cols):
        errors.append(f"Missing required columns. Found: {list(submission_df.columns)}, Required: {required_cols}")
        return False, errors, warnings
    
    # Check for duplicates
    if submission_df['ID'].duplicated().any():
        duplicates = submission_df[submission_df['ID'].duplicated()]['ID'].tolist()
        errors.append(f"Duplicate IDs found: {duplicates[:10]}...")
        return False, errors, warnings
    
    # Check valid class labels
    valid_classes = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    invalid_labels = submission_df[~submission_df['labels'].isin(valid_classes)]['labels'].unique()
    if len(invalid_labels) > 0:
        errors.append(f"Invalid class labels found: {list(invalid_labels)}")
        return False, errors, warnings
    
    # Check for missing values
    if submission_df.isnull().any().any():
        null_cols = submission_df.columns[submission_df.isnull().any()].tolist()
        errors.append(f"Missing values in columns: {null_cols}")
        return False, errors, warnings
    
    # Check ID format (should be image filenames)
    id_format_check = submission_df['ID'].astype(str).str.match(r'.*\.(jpg|jpeg|png|JPG|JPEG|PNG)$')
    if not id_format_check.all():
        warnings.append("Some IDs don't appear to be image filenames (should end with .jpg/.jpeg/.png)")
    
    # Check against test set if provided
    if test_df is not None:
        test_ids = set(test_df['ID'].values)
        submission_ids = set(submission_df['ID'].values)
        
        missing = test_ids - submission_ids
        extra = submission_ids - test_ids
        
        if missing:
            errors.append(f"Missing {len(missing)} IDs from test set")
            if len(missing) <= 10:
                errors.append(f"  Missing IDs: {list(missing)[:10]}")
        if extra:
            warnings.append(f"Found {len(extra)} extra IDs not in test set")
        
        if len(submission_df) != len(test_df):
            warnings.append(f"Row count mismatch: submission={len(submission_df)}, test={len(test_df)}")
    
    return len(errors) == 0, errors, warnings

def prepare_final_submission(
    config,
    use_optimized_weights=True,
    validate=True
):
    """
    Prepare final submission with optimized ensemble.
    
    Args:
        config: Config object
        use_optimized_weights: If True, uses optimized ensemble weights
        validate: If True, validates submission format
    
    Returns:
        submission_df, metrics
    """
    print(f"\n{'='*60}")
    print("Final Submission Preparation")
    print(f"{'='*60}")
    
    # Find all trained models
    model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
    
    if not model_paths:
        raise ValueError("No trained models found. Please train models first.")
    
    print(f"Found {len(model_paths)} models")
    
    # Optimize ensemble weights if requested
    best_method = 'classy'  # Default
    optimized_weights = None
    
    if use_optimized_weights and hasattr(config, 'ENSEMBLE_OPTIMIZATION') and config.ENSEMBLE_OPTIMIZATION:
        print("\nOptimizing ensemble weights...")
        try:
            best_weights, best_f1, best_method, results = optimize_ensemble_weights(
                config,
                model_paths,
                method=getattr(config, 'ENSEMBLE_OPT_METHOD', 'all')
            )
            print(f"Best ensemble method: {best_method}")
            print(f"Expected macro F1: {best_f1:.5f}")
            
            if best_method != 'classy' and isinstance(best_weights, np.ndarray):
                optimized_weights = best_weights
            elif best_method == 'classy':
                optimized_weights = 'classy'
        except Exception as e:
            print(f"Warning: Ensemble optimization failed: {e}")
            print("Using default Classy Ensemble method")
            best_method = 'classy'
            optimized_weights = 'classy'
    
    # Generate predictions on eval set first (for validation)
    print("\n" + "="*60)
    print("Step 1: Generating predictions on eval set (for validation)...")
    print("="*60)
    
    use_classy = (best_method == 'classy' or optimized_weights == 'classy')
    eval_submission = predict_eval_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=use_classy,
        optimized_weights=optimized_weights if optimized_weights != 'classy' else None
    )
    
    # Validate eval predictions
    if validate:
        eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
        is_valid, errors, warnings_list = validate_submission_format(eval_submission, eval_df)
        if not is_valid:
            print("WARNING: Eval predictions have format issues:")
            for error in errors:
                print(f"  - {error}")
            for warning in warnings_list:
                print(f"  ⚠ {warning}")
        else:
            print("✓ Eval predictions format is valid")
            if warnings_list:
                for warning in warnings_list:
                    print(f"  ⚠ {warning}")
    
    # Compute eval metrics
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    merged = eval_submission.merge(eval_df, on='ID', how='inner')
    y_true = merged['labels_y'].values
    y_pred = merged['labels_x'].values
    
    eval_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    eval_acc = accuracy_score(y_true, y_pred)
    
    print(f"\nEval Set Performance:")
    print(f"  Macro F1: {eval_macro_f1:.5f}")
    print(f"  Accuracy: {eval_acc:.5f}")
    
    # Save eval predictions
    eval_submission.to_csv(config.PRED_DIR / 'final_eval_predictions.csv', index=False)
    print(f"Eval predictions saved to: {config.PRED_DIR / 'final_eval_predictions.csv'}")
    
    # Generate final test predictions
    print("\n" + "="*60)
    print("Step 2: Generating final test predictions...")
    print("="*60)
    
    test_submission = predict_test_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=use_classy,
        optimized_weights=optimized_weights
    )
    
    # Validate test predictions
    if validate:
        test_df = pd.read_csv(config.PHASE2_TEST_CSV)
        is_valid, errors, warnings_list = validate_submission_format(test_submission, test_df)
        
        if not is_valid:
            print("ERROR: Test predictions have format issues:")
            for error in errors:
                print(f"  ✗ {error}")
            raise ValueError("Submission validation failed. Please fix errors above.")
        else:
            print("✓ Test predictions format is valid")
            if warnings_list:
                for warning in warnings_list:
                    print(f"  ⚠ {warning}")
    
    # Final checks
    print("\n" + "="*60)
    print("Final Submission Checks")
    print("="*60)
    print(f"Total predictions: {len(test_submission)}")
    print(f"Unique IDs: {test_submission['ID'].nunique()}")
    print(f"Unique labels: {test_submission['labels'].nunique()}")
    print(f"\nClass distribution:")
    print(test_submission['labels'].value_counts().sort_index())
    
    # Save final submission
    submission_path = config.PRED_DIR / getattr(config, 'SUBMISSION_FILENAME', 'final_submission.csv')
    test_submission.to_csv(submission_path, index=False)
    print(f"\n✓ Final submission saved to: {submission_path}")
    
    # Save probabilities if requested
    if getattr(config, 'SAVE_PREDICTION_PROBABILITIES', False):
        # Note: Probabilities would need to be saved during prediction
        print("  (Probabilities can be saved by modifying inference to return them)")
    
    # Save metrics
    metrics = {
        'eval_macro_f1': float(eval_macro_f1),
        'eval_accuracy': float(eval_acc),
        'ensemble_method': best_method,
        'num_models': len(model_paths),
        'tta_enabled': config.TTA,
        'tta_num': config.TTA_NUM,
        'img_size': config.IMG_SIZE
    }
    
    metrics_path = config.LOG_DIR / 'final_submission_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    return test_submission, metrics

def main():
    """Main final submission function"""
    config = Config()
    
    try:
        submission, metrics = prepare_final_submission(
            config,
            use_optimized_weights=getattr(config, 'ENSEMBLE_OPTIMIZATION', True),
            validate=getattr(config, 'FINAL_SUBMISSION_VALIDATION', True)
        )
        
        print(f"\n{'='*60}")
        print("Final Submission Ready!")
        print(f"{'='*60}")
        print(f"File: {config.PRED_DIR / getattr(config, 'SUBMISSION_FILENAME', 'final_submission.csv')}")
        print(f"Eval Macro F1: {metrics['eval_macro_f1']:.5f}")
        print(f"Ensemble Method: {metrics['ensemble_method']}")
        print(f"\nReady for submission to Kaggle!")
        
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())

