"""
Generate Session 3 summary report
"""

import json
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

def generate_summary():
    """Generate comprehensive Session 3 summary"""
    config = Config()
    
    print("="*60)
    print("Session 3 Summary Report")
    print("="*60)
    
    # Load metrics
    try:
        metrics_path = config.LOG_DIR / 'final_submission_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"\nFinal Performance:")
            print(f"  Eval Macro F1: {metrics['eval_macro_f1']:.5f}")
            print(f"  Eval Accuracy: {metrics['eval_accuracy']:.5f}")
            print(f"  Ensemble Method: {metrics['ensemble_method']}")
            print(f"  Number of Models: {metrics['num_models']}")
            print(f"  TTA Enabled: {metrics['tta_enabled']}")
            print(f"  TTA Iterations: {metrics['tta_num']}")
            print(f"  Image Size: {metrics['img_size']}")
        else:
            print("\n  Metrics not available (run final_submission.py first)")
    except Exception as e:
        print(f"\n  Error loading metrics: {e}")
    
    # Load ensemble optimization results
    try:
        opt_path = config.LOG_DIR / 'ensemble_optimization_results.json'
        if opt_path.exists():
            with open(opt_path) as f:
                opt_results = json.load(f)
            print(f"\nEnsemble Optimization Results:")
            print(f"  Best Method: {opt_results.get('best_method', 'N/A')}")
            print(f"  Best Macro F1: {opt_results.get('best_macro_f1', 0):.5f}")
            if 'weighted' in opt_results:
                print(f"  Weighted Ensemble F1: {opt_results['weighted']['macro_f1']:.5f}")
            if 'equal' in opt_results:
                print(f"  Equal Weight F1: {opt_results['equal']['macro_f1']:.5f}")
            if 'classy' in opt_results:
                print(f"  Classy Ensemble F1: {opt_results['classy']['macro_f1']:.5f}")
    except Exception as e:
        print(f"\n  Ensemble optimization results not available: {e}")
    
    # Check submission
    submission_path = config.PRED_DIR / getattr(config, 'SUBMISSION_FILENAME', 'final_submission.csv')
    if submission_path.exists():
        submission = pd.read_csv(submission_path)
        print(f"\nSubmission File:")
        print(f"  Path: {submission_path}")
        print(f"  Rows: {len(submission)}")
        print(f"  Classes: {submission['labels'].nunique()}")
        print(f"  Class distribution:")
        for cls, count in submission['labels'].value_counts().sort_index().items():
            print(f"    {cls}: {count}")
    else:
        print("\n  Submission file not found!")
        print("  Generate with: python scripts/final_submission.py")
    
    # Check models
    model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
    print(f"\nTrained Models: {len(model_paths)}")
    
    # Group by model type
    from collections import defaultdict
    model_groups = defaultdict(int)
    for path in model_paths:
        name = Path(path).stem
        # Extract model name (before _fold)
        model_name = '_'.join(name.split('_')[:-2]) if '_fold' in name else name
        model_groups[model_name] += 1
    
    print("  Model breakdown:")
    for model_name, count in sorted(model_groups.items()):
        print(f"    {model_name}: {count} models")
    
    print(f"\n{'='*60}")
    print("Session 3 Complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    generate_summary()

