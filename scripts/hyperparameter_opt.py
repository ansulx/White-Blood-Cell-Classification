"""
Quick hyperparameter optimization using Optuna
Optimizes learning rate, weight decay, and drop path rate

Install Optuna: pip install optuna
"""

import sys
from pathlib import Path
import json

try:
    import optuna
except ImportError:
    print("ERROR: Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
import src.config as config_module

def objective(trial):
    """Optuna objective function"""
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    drop_path_rate = trial.suggest_float('drop_path_rate', 0.1, 0.5)
    
    # Update config
    config_module.Config.LEARNING_RATE = lr
    config_module.Config.WEIGHT_DECAY = weight_decay
    
    # Train single fold (quick evaluation)
    from scripts.train import train_fold
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    
    config = Config()
    
    # Load data (with pseudo-labels if available)
    merged_train_path = config.PRED_DIR / 'merged_train_with_pseudo.csv'
    if merged_train_path.exists():
        all_train = pd.read_csv(merged_train_path)
    else:
        phase2_train = pd.read_csv(config.PHASE2_TRAIN_CSV)
        phase1 = pd.read_csv(config.PHASE1_CSV)
        phase1 = phase1[phase1['split'] == 'phase1_train']
        phase1 = phase1[['ID', 'labels']].copy()
        phase2_eval = pd.read_csv(config.PHASE2_EVAL_CSV)
        all_train = pd.concat([
            phase2_train[['ID', 'labels']],
            phase1[['ID', 'labels']],
            phase2_eval[['ID', 'labels']]
        ], ignore_index=True)
    
    # Single fold for quick evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(all_train, all_train['labels'])))
    train_df = all_train.iloc[train_idx].reset_index(drop=True)
    val_df = all_train.iloc[val_idx].reset_index(drop=True)
    
    # Train single fold with custom drop_path_rate
    # Temporarily override drop_path_rate in model creation
    original_drop_path = getattr(config, 'drop_path_rate', None)
    config.drop_path_rate = drop_path_rate
    
    try:
        val_macro_f1, _ = train_fold(0, train_df, val_df, config)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Return low score for failed trials
    finally:
        if original_drop_path is not None:
            config.drop_path_rate = original_drop_path
    
    return val_macro_f1

def main():
    """Run hyperparameter optimization"""
    print("="*60)
    print("Hyperparameter Optimization (Optuna)")
    print("="*60)
    print("This will take ~1 hour (20 trials)")
    print("Optimizing: Learning Rate, Weight Decay, Drop Path Rate")
    print("="*60)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='wbc_hyperopt',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    print("\n" + "="*60)
    print("Best Hyperparameters:")
    print("="*60)
    print(f"Learning Rate: {study.best_params['learning_rate']:.6f}")
    print(f"Weight Decay: {study.best_params['weight_decay']:.6f}")
    print(f"Drop Path Rate: {study.best_params['drop_path_rate']:.4f}")
    print(f"Best Macro F1: {study.best_value:.5f}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': float(study.best_value),
        'n_trials': len(study.trials)
    }
    
    results_path = Config().LOG_DIR / 'best_hyperparameters.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\nUpdate config.py with these values for final training:")
    print(f"  LEARNING_RATE = {study.best_params['learning_rate']:.6f}")
    print(f"  WEIGHT_DECAY = {study.best_params['weight_decay']:.6f}")
    print(f"  Note: Drop Path Rate is model-specific, update in MODEL_SPECIFIC_SETTINGS")

if __name__ == '__main__':
    main()

