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
    validate=True,
    precomputed_method=None,
    precomputed_weights=None,
    skip_optimization=False
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

    if skip_optimization and precomputed_method is not None:
        best_method = precomputed_method
        if isinstance(precomputed_weights, np.ndarray):
            optimized_weights = precomputed_weights
        elif precomputed_weights == 'classy':
            optimized_weights = 'classy'
        else:
            optimized_weights = precomputed_weights
    elif use_optimized_weights and hasattr(config, 'ENSEMBLE_OPTIMIZATION') and config.ENSEMBLE_OPTIMIZATION:
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
    eval_submission, eval_probs, eval_names, eval_idx_to_class = predict_eval_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=use_classy,
        optimized_weights=optimized_weights if optimized_weights != 'classy' else None,
        return_probs=True
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
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    merged = eval_submission.merge(eval_df, on='ID', how='inner')
    y_true = merged['labels_y'].values
    y_pred = merged['labels_x'].values
    
    eval_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    eval_acc = accuracy_score(y_true, y_pred)
    
    print(f"\nEval Set Performance:")
    print(f"  Macro F1: {eval_macro_f1:.5f}")
    print(f"  Accuracy: {eval_acc:.5f}")

    ece_value = None
    # Save Session 3 eval plots (non-blocking)
    try:
        import matplotlib.pyplot as plt
        plots_dir = config.LOG_DIR / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Class order for plots
        if isinstance(eval_idx_to_class, dict):
            class_names = [eval_idx_to_class[i] for i in range(len(eval_idx_to_class))]
        elif eval_idx_to_class is not None:
            class_names = list(eval_idx_to_class)
        else:
            class_names = sorted(eval_df['labels'].unique())

        # Per-class F1 bar chart
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=class_names)
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, per_class_f1, color='steelblue')
        plt.title('Session 3 Eval Per-Class F1')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        f1_plot_path = plots_dir / 'session3_eval_per_class_f1.png'
        plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved eval per-class F1 plot to {f1_plot_path}")

        # Confusion matrix heatmap
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Session 3 Eval Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_plot_path = plots_dir / 'session3_eval_confusion_matrix.png'
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved eval confusion matrix to {cm_plot_path}")

        # Normalized confusion matrix
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.title('Session 3 Eval Confusion Matrix (Normalized)')
        plt.colorbar()
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_norm_plot_path = plots_dir / 'session3_eval_confusion_matrix_normalized.png'
        plt.savefig(cm_norm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved normalized confusion matrix to {cm_norm_plot_path}")

        # Per-class precision/recall bar chart + CSV
        report = classification_report(y_true, y_pred, labels=class_names, output_dict=True, zero_division=0)
        pr_df = pd.DataFrame({
            'class': class_names,
            'precision': [report[c]['precision'] for c in class_names],
            'recall': [report[c]['recall'] for c in class_names],
            'f1': [report[c]['f1-score'] for c in class_names],
            'support': [report[c]['support'] for c in class_names]
        })
        pr_csv_path = plots_dir / 'session3_eval_per_class_metrics.csv'
        pr_df.to_csv(pr_csv_path, index=False)
        print(f"Saved per-class metrics to {pr_csv_path}")

        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_names))
        width = 0.35
        plt.bar(x - width / 2, pr_df['precision'], width, label='Precision', color='tab:blue')
        plt.bar(x + width / 2, pr_df['recall'], width, label='Recall', color='tab:orange')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.title('Session 3 Eval Precision/Recall by Class')
        plt.legend()
        plt.tight_layout()
        pr_plot_path = plots_dir / 'session3_eval_precision_recall.png'
        plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved precision/recall plot to {pr_plot_path}")

        # Reliability diagram + ECE (requires probabilities)
        if eval_probs is not None and eval_names is not None and eval_idx_to_class is not None:
            class_to_idx = {v: k for k, v in (eval_idx_to_class.items() if isinstance(eval_idx_to_class, dict) else enumerate(eval_idx_to_class))}
            prob_by_id = {name: eval_probs[i] for i, name in enumerate(eval_names)}
            try:
                probs_aligned = np.stack([prob_by_id[_id] for _id in merged['ID'].values])
                conf = probs_aligned.max(axis=1)
                pred_idx = probs_aligned.argmax(axis=1)
                true_idx = np.array([class_to_idx[label] for label in y_true])
                acc = (pred_idx == true_idx).astype(float)

                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_acc = np.zeros(n_bins)
                bin_conf = np.zeros(n_bins)
                bin_count = np.zeros(n_bins)
                for i in range(n_bins):
                    mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
                    if np.any(mask):
                        bin_acc[i] = acc[mask].mean()
                        bin_conf[i] = conf[mask].mean()
                        bin_count[i] = mask.sum()
                ece = np.sum((bin_count / len(conf)) * np.abs(bin_acc - bin_conf))
                ece_value = float(ece)

                plt.figure(figsize=(7, 6))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, label='Accuracy')
                plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
                plt.plot(bin_centers, bin_conf, marker='o', color='black', label='Confidence')
                plt.title(f'Session 3 Eval Reliability Diagram (ECE={ece:.4f})')
                plt.xlabel('Confidence')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                plt.xlim(0, 1)
                plt.legend()
                plt.tight_layout()
                calib_plot_path = plots_dir / 'session3_eval_reliability_diagram.png'
                plt.savefig(calib_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved reliability diagram to {calib_plot_path}")
            except Exception as e:
                print(f"⚠ Skipped reliability diagram due to error: {e}")

        # Ablation plot from optimization results (if present)
        opt_path = config.LOG_DIR / 'ensemble_optimization_results.json'
        if opt_path.exists():
            with open(opt_path) as f:
                opt_results = json.load(f)
            methods = []
            f1s = []
            for key in ['weighted', 'equal', 'classy']:
                if key in opt_results and 'macro_f1' in opt_results[key]:
                    methods.append(key)
                    f1s.append(opt_results[key]['macro_f1'])
            if methods:
                plt.figure(figsize=(6, 4))
                plt.bar(methods, f1s, color='seagreen')
                plt.ylim(0, 1)
                plt.title('Session 3 Ensemble Method Comparison')
                plt.ylabel('Macro F1')
                plt.tight_layout()
                ablation_plot_path = plots_dir / 'session3_ensemble_ablation.png'
                plt.savefig(ablation_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved ensemble ablation plot to {ablation_plot_path}")
    except Exception as e:
        print(f"⚠ Skipped Session 3 plots due to error: {e}")
    
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
    if ece_value is not None:
        metrics['eval_ece'] = ece_value
    
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

