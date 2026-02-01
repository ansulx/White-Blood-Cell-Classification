"""
Ensemble Weight Optimization for WBC-Bench-2026
Tests different ensemble weighting strategies to maximize macro F1
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
from itertools import product
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_val_transforms, get_tta_transforms
from scripts.inference import load_model, predict_ensemble, predict_ensemble_classy, predict_ensemble_mixed_sizes, get_model_img_size
from sklearn.metrics import f1_score, accuracy_score

_MEAN = np.array(Config.MEAN)
_STD = np.array(Config.STD)

def denorm_to_uint8(img_tensor):
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * _STD + _MEAN) * 255
    return img_np.astype(np.uint8)

def evaluate_ensemble_weights(
    model_paths,
    eval_loader,
    eval_labels,
    device,
    weights,
    tta=False,
    tta_transforms=None
):
    """
    Evaluate ensemble with given weights.
    
    Args:
        model_paths: List of model paths
        eval_loader: DataLoader for eval set
        eval_labels: Ground truth labels
        weights: Array of shape (num_models,) or (num_models, num_classes) for per-class weights
        tta: Whether to use TTA
        tta_transforms: TTA transforms
    
    Returns:
        macro_f1, accuracy, per_class_f1
    """
    models = []
    idx_to_class_list = []
    
    for model_path in model_paths:
        model, idx_to_class = load_model(model_path, device)
        models.append(model)
        idx_to_class_list.append(idx_to_class)
    
    idx_to_class = idx_to_class_list[0]
    num_classes = len(idx_to_class)
    
    all_probs = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating ensemble', leave=False):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch
            
            images = images.to(device)
            model_probs_list = []
            
            for model in models:
                if tta and tta_transforms:
                    tta_preds = []
                    for tta_transform in tta_transforms:
                        tta_images = []
                        for img in images:
                            img_np = denorm_to_uint8(img)
                            transformed = tta_transform(image=img_np)
                            tta_images.append(transformed['image'])
                        tta_images = torch.stack(tta_images).to(device)
                        outputs = model(tta_images)
                        probs = F.softmax(outputs, dim=1)
                        tta_preds.append(probs)
                    avg_probs = torch.stack(tta_preds).mean(0)
                else:
                    outputs = model(images)
                    avg_probs = F.softmax(outputs, dim=1)
                
                model_probs_list.append(avg_probs.cpu().numpy())
            
            # Apply weights
            if weights.ndim == 1:
                # Global weights: shape (num_models,)
                weighted_probs = np.zeros_like(model_probs_list[0])
                for i, probs in enumerate(model_probs_list):
                    weighted_probs += weights[i] * probs
            else:
                # Per-class weights: shape (num_models, num_classes)
                weighted_probs = np.zeros_like(model_probs_list[0])
                for i, probs in enumerate(model_probs_list):
                    for j in range(num_classes):
                        weighted_probs[:, j] += weights[i, j] * probs[:, j]
            
            all_probs.append(weighted_probs)
            all_names.extend(names)
    
    all_probs = np.vstack(all_probs)
    preds = np.argmax(all_probs, axis=1)
    pred_labels = [idx_to_class[p] for p in preds]
    
    # Compute metrics
    macro_f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
    acc = accuracy_score(eval_labels, pred_labels)
    per_class_f1 = f1_score(eval_labels, pred_labels, average=None, zero_division=0)
    
    return macro_f1, acc, per_class_f1

def optimize_ensemble_weights(
    config,
    model_paths,
    method='all'
):
    """
    Optimize ensemble weights using different strategies.
    
    Args:
        config: Config object
        model_paths: List of model paths
        method: 'weighted' (by macro F1), 'equal', 'classy' (per-class), 'grid_search', or 'all'
    
    Returns:
        Best weights and performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Weight Optimization: {method}")
    print(f"{'='*60}")
    
    # Load eval data
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    eval_labels = eval_df['labels'].values.tolist()
    
    sizes = {get_model_img_size(p, config) for p in model_paths}
    mixed_sizes = len(sizes) > 1

    if not mixed_sizes:
        eval_dataset = WBCDataset(
            csv_path=config.PHASE2_EVAL_CSV,
            img_dir=config.PHASE2_EVAL_DIR,
            transform=get_val_transforms(config.IMG_SIZE),
            is_train=False
        )
        
        infer_batch_size = getattr(config, 'INFER_BATCH_SIZE', config.BATCH_SIZE)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
            persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
        )
        
        tta_transforms = None
        if config.TTA:
            tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    num_models = len(model_paths)
    
    # Get class names
    temp_model, temp_idx_to_class = load_model(model_paths[0], config.DEVICE)
    num_classes = len(temp_idx_to_class)
    class_names = [temp_idx_to_class[i] for i in range(num_classes)]
    del temp_model
    torch.cuda.empty_cache()

    def eval_mixed(weights):
        preds, probs, names, idx_to_class = predict_ensemble_mixed_sizes(
            model_paths,
            config,
            csv_path=config.PHASE2_EVAL_CSV,
            img_dir=config.PHASE2_EVAL_DIR,
            tta=config.TTA,
            weights=weights,
            eval_labels=eval_labels
        )
        pred_labels = [idx_to_class[p] for p in preds]
        macro_f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
        acc = accuracy_score(eval_labels, pred_labels)
        per_class_f1 = f1_score(eval_labels, pred_labels, average=None, zero_division=0)
        return macro_f1, acc, per_class_f1

    def eval_single(model_path):
        preds, probs, names, idx_to_class = predict_ensemble_mixed_sizes(
            [model_path],
            config,
            csv_path=config.PHASE2_EVAL_CSV,
            img_dir=config.PHASE2_EVAL_DIR,
            tta=config.TTA,
            weights=None,
            eval_labels=eval_labels
        )
        pred_labels = [idx_to_class[p] for p in preds]
        macro_f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
        return macro_f1
    
    best_weights = None
    best_f1 = 0.0
    best_method = None
    results = {}
    
    if method in ['weighted', 'all']:
        print("\n1. Testing Weighted Ensemble (by macro F1)...")
        # Get individual model performances
        model_f1s = []
        for i, model_path in enumerate(model_paths):
            print(f"  Evaluating model {i+1}/{num_models}...")
            if mixed_sizes:
                f1 = eval_single(model_path)
            else:
                weights = np.zeros(num_models)
                weights[i] = 1.0
                f1, acc, _ = evaluate_ensemble_weights(
                    [model_path], eval_loader, eval_labels, config.DEVICE,
                    weights, tta=config.TTA, tta_transforms=tta_transforms
                )
            model_f1s.append(f1)
            print(f"    Model {i+1} Macro F1: {f1:.5f}")
        
        # Normalize weights by F1 scores
        model_f1s = np.array(model_f1s)
        model_f1s = model_f1s + 1e-8  # Avoid division by zero
        weights = model_f1s / model_f1s.sum()
        
        # Evaluate weighted ensemble
        if mixed_sizes:
            f1, acc, per_class_f1 = eval_mixed(weights)
        else:
            f1, acc, per_class_f1 = evaluate_ensemble_weights(
                model_paths, eval_loader, eval_labels, config.DEVICE,
                weights, tta=config.TTA, tta_transforms=tta_transforms
            )
        
        print(f"  Weighted Ensemble Macro F1: {f1:.5f}")
        results['weighted'] = {
            'weights': weights.tolist(),
            'macro_f1': float(f1),
            'accuracy': float(acc),
            'per_class_f1': per_class_f1.tolist()
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_method = 'weighted'
    
    if method in ['equal', 'all']:
        print("\n2. Testing Equal Weight Ensemble...")
        weights = np.ones(num_models) / num_models
        if mixed_sizes:
            f1, acc, per_class_f1 = eval_mixed(weights)
        else:
            f1, acc, per_class_f1 = evaluate_ensemble_weights(
                model_paths, eval_loader, eval_labels, config.DEVICE,
                weights, tta=config.TTA, tta_transforms=tta_transforms
            )
        
        print(f"  Equal Weight Ensemble Macro F1: {f1:.5f}")
        results['equal'] = {
            'weights': weights.tolist(),
            'macro_f1': float(f1),
            'accuracy': float(acc),
            'per_class_f1': per_class_f1.tolist()
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_method = 'equal'
    
    if method in ['classy', 'all']:
        print("\n3. Testing Classy Ensemble (per-class weights)...")
        try:
            if mixed_sizes:
                f1, acc, per_class_f1 = eval_mixed('classy')
            else:
                preds, probs, names, idx_to_class = predict_ensemble_classy(
                    model_paths, eval_loader, config.DEVICE,
                    tta=config.TTA, tta_transforms=tta_transforms,
                    eval_labels=eval_labels, class_names=class_names
                )
                
                pred_labels = [idx_to_class[p] for p in preds]
                f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
                acc = accuracy_score(eval_labels, pred_labels)
                per_class_f1 = f1_score(eval_labels, pred_labels, average=None, zero_division=0)
            
            print(f"  Classy Ensemble Macro F1: {f1:.5f}")
            results['classy'] = {
                'weights': 'per-class',  # Per-class weights computed internally
                'macro_f1': float(f1),
                'accuracy': float(acc),
                'per_class_f1': per_class_f1.tolist()
            }
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = 'classy'  # Special marker for classy ensemble
                best_method = 'classy'
        except Exception as e:
            print(f"  ⚠ Classy Ensemble failed: {e}")
            print("  Continuing with other methods...")
    
    if method == 'grid_search':
        if mixed_sizes:
            print("\n4. Grid Search skipped for mixed image sizes.")
            return best_weights, best_f1, best_method, results
        print("\n4. Grid Search for Optimal Weights...")
        # Simple grid search (limited to avoid too many combinations)
        best_grid_f1 = 0.0
        best_grid_weights = None
        
        # Generate weight combinations (normalized)
        weight_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
        total_combinations = len(weight_steps) ** num_models
        print(f"  Testing {total_combinations} weight combinations...")
        
        tested = 0
        for weights in product(weight_steps, repeat=num_models):
            weights = np.array(weights)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            
            try:
                f1, acc, _ = evaluate_ensemble_weights(
                    model_paths, eval_loader, eval_labels, config.DEVICE,
                    weights, tta=config.TTA, tta_transforms=tta_transforms
                )
                
                if f1 > best_grid_f1:
                    best_grid_f1 = f1
                    best_grid_weights = weights
                
                tested += 1
                if tested % 100 == 0:
                    print(f"    Tested {tested}/{total_combinations} combinations, best F1: {best_grid_f1:.5f}")
            except Exception as e:
                print(f"    Warning: Combination failed: {e}")
                continue
        
        print(f"  Grid Search Best Macro F1: {best_grid_f1:.5f}")
        results['grid_search'] = {
            'weights': best_grid_weights.tolist() if best_grid_weights is not None else None,
            'macro_f1': float(best_grid_f1)
        }
        
        if best_grid_f1 > best_f1:
            best_f1 = best_grid_f1
            best_weights = best_grid_weights
            best_method = 'grid_search'
    
    print(f"\n{'='*60}")
    print(f"Best Ensemble Method: {best_method}")
    print(f"Best Macro F1: {best_f1:.5f}")
    print(f"{'='*60}")
    
    # Save results
    results['best_method'] = best_method
    results['best_macro_f1'] = float(best_f1)
    if isinstance(best_weights, np.ndarray):
        results['best_weights'] = best_weights.tolist()
    elif best_weights == 'classy':
        results['best_weights'] = 'classy'
    else:
        results['best_weights'] = best_weights
    
    results_path = config.LOG_DIR / 'ensemble_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return best_weights, best_f1, best_method, results

def main():
    """Main ensemble optimization function"""
    config = Config()
    
    # Find all trained models
    model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
    
    if not model_paths:
        print("ERROR: No trained models found. Please train models first.")
        print("Run: python scripts/train.py")
        return 1
    
    print(f"Found {len(model_paths)} models")
    print(f"Models: {[Path(p).stem for p in model_paths[:5]]}...")
    
    # Optimize ensemble weights
    try:
        best_weights, best_f1, best_method, results = optimize_ensemble_weights(
            config,
            model_paths,
            method=getattr(config, 'ENSEMBLE_OPT_METHOD', 'all')
        )
        
        print(f"\n{'='*60}")
        print("Ensemble Optimization Complete!")
        print(f"{'='*60}")
        print(f"Use method '{best_method}' for final predictions")
        print(f"Expected macro F1: {best_f1:.5f}")
        return 0
    except Exception as e:
        print(f"\nERROR: Ensemble optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())

