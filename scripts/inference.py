"""
Inference script for WBC-Bench-2026 competition
Supports single model and ensemble predictions with TTA
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
from src.models import get_model
from PIL import Image

def load_model(model_path, device='cuda'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    num_classes = len(class_to_idx)
    
    # Infer model name from path (prioritize best models)
    model_name = Config.MODEL_NAME
    # Swin V2 variants (check Large first, then Base)
    # FIXED: Updated to use window12to16_192to256 (better for 512 images)
    if 'swinv2_large_window12to16_192to256' in str(model_path):
        model_name = 'swinv2_large_window12to16_192to256'
    elif 'swinv2_large_window12to24_192to384' in str(model_path):
        model_name = 'swinv2_large_window12to24_192to384'  # Fallback to old name
    elif 'swinv2_large' in str(model_path):
        model_name = 'swinv2_large_window12to16_192to256'  # Default to optimized Swin V2 Large
    elif 'swinv2_base_window8_256' in str(model_path) or 'swin_v2' in str(model_path):
        model_name = 'swinv2_base_window8_256'
    elif 'swinv2' in str(model_path):
        model_name = 'swinv2_large_window12to24_192to384'  # Default to best Swin V2 Large
    # MaxViT variants (check XLarge first, then Large, then Base)
    elif 'maxvit_xlarge_tf_384' in str(model_path):
        model_name = 'maxvit_xlarge_tf_384'
    elif 'maxvit_xlarge' in str(model_path):
        model_name = 'maxvit_xlarge_tf_384'  # Default to best MaxViT XLarge
    elif 'maxvit_large_tf_384' in str(model_path):
        model_name = 'maxvit_large_tf_384'
    elif 'maxvit_large' in str(model_path):
        model_name = 'maxvit_xlarge_tf_384'  # Default to best MaxViT XLarge
    elif 'maxvit_base_tf_384' in str(model_path):
        model_name = 'maxvit_base_tf_384'
    elif 'maxvit' in str(model_path):
        model_name = 'maxvit_xlarge_tf_384'  # Default to best MaxViT XLarge
    elif 'convnextv2_large' in str(model_path) or 'convnext_v2_large' in str(model_path):
        model_name = 'convnextv2_large'
    elif 'convnextv2_base' in str(model_path) or 'convnext_v2_base' in str(model_path):
        model_name = 'convnextv2_base'
    elif 'convnextv2' in str(model_path) or 'convnext_v2' in str(model_path):
        model_name = 'convnextv2_base'  # Default ConvNeXt V2
    elif 'convnext_base' in str(model_path):
        model_name = 'convnext_base'
    elif 'convnext' in str(model_path):
        model_name = 'convnext_base'  # Default ConvNeXt
    elif 'efficientnet_b5' in str(model_path):
        model_name = 'efficientnet_b5'
    elif 'efficientnet_b4' in str(model_path):
        model_name = 'efficientnet_b4'
    elif 'efficientnet' in str(model_path):
        model_name = 'efficientnet_b4'  # Default fallback
    
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Handle torch.compile prefix (_orig_mod.) - remove it if present
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, idx_to_class

def predict_single_model(model, dataloader, device, tta=False, tta_transforms=None):
    """Make predictions with a single model"""
    all_preds = []
    all_probs = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch
            
            images = images.to(device)
            
            if tta and tta_transforms:
                # Test Time Augmentation
                tta_preds = []
                for tta_transform in tta_transforms:
                    # Apply TTA transform
                    tta_images = []
                    for img in images:
                        img_np = img.cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                        img_np = img_np.astype(np.uint8)
                        transformed = tta_transform(image=img_np)
                        tta_images.append(transformed['image'])
                    tta_images = torch.stack(tta_images).to(device)
                    
                    outputs = model(tta_images)
                    probs = F.softmax(outputs, dim=1)
                    tta_preds.append(probs)
                
                # Average TTA predictions
                avg_probs = torch.stack(tta_preds).mean(0)
            else:
                outputs = model(images)
                avg_probs = F.softmax(outputs, dim=1)
            
            preds = torch.argmax(avg_probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(avg_probs.cpu().numpy())
            all_names.extend(names)
    
    all_probs = np.vstack(all_probs)
    return all_preds, all_probs, all_names

def predict_ensemble(model_paths, dataloader, device, tta=False, tta_transforms=None):
    """Make predictions with ensemble of models"""
    models = []
    idx_to_class_list = []
    
    print(f"Loading {len(model_paths)} models...")
    for model_path in model_paths:
        model, idx_to_class = load_model(model_path, device)
        models.append(model)
        idx_to_class_list.append(idx_to_class)
    
    # Verify all models have same classes
    assert all(
        idx_to_class_list[0] == idx_to_class 
        for idx_to_class in idx_to_class_list
    ), "All models must have same classes"
    
    all_ensemble_probs = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Ensemble Predicting'):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch
            
            images = images.to(device)
            model_probs = []
            
            for model in models:
                if tta and tta_transforms:
                    tta_preds = []
                    for tta_transform in tta_transforms:
                        tta_images = []
                        for img in images:
                            img_np = img.cpu().permute(1, 2, 0).numpy()
                            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                            img_np = img_np.astype(np.uint8)
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
                
                model_probs.append(avg_probs.cpu().numpy())
            
            # Average across models
            ensemble_probs = np.stack(model_probs).mean(0)
            all_ensemble_probs.append(ensemble_probs)
            all_names.extend(names)
    
    all_ensemble_probs = np.vstack(all_ensemble_probs)
    all_preds = np.argmax(all_ensemble_probs, axis=1)
    
    return all_preds, all_ensemble_probs, all_names, idx_to_class_list[0]

def predict_ensemble_classy(model_paths, dataloader, device, tta=False, tta_transforms=None, 
                          eval_labels=None, class_names=None):
    """
    Classy Ensemble: Per-class weighted ensemble based on per-class F1 scores.
    
    Paper: "Classy Ensemble: A Novel Ensemble Algorithm for Classification" (2023)
    https://arxiv.org/abs/2302.10580
    
    Args:
        model_paths: List of model checkpoint paths
        dataloader: DataLoader for predictions
        device: Device to run on
        tta: Whether to use TTA
        tta_transforms: List of TTA transforms
        eval_labels: Ground truth labels for computing per-class weights (optional)
        class_names: List of class names
    """
    from sklearn.metrics import f1_score
    
    models = []
    idx_to_class_list = []
    
    print(f"Loading {len(model_paths)} models for Classy Ensemble...")
    for model_path in model_paths:
        model, idx_to_class = load_model(model_path, device)
        models.append(model)
        idx_to_class_list.append(idx_to_class)
    
    # Verify all models have same classes
    assert all(
        idx_to_class_list[0] == idx_to_class 
        for idx_to_class in idx_to_class_list
    ), "All models must have same classes"
    
    idx_to_class = idx_to_class_list[0]
    num_classes = len(idx_to_class)
    
    # Compute per-class weights if eval_labels provided
    class_weights = None
    if eval_labels is not None and class_names is not None:
        print("Computing per-class F1 scores for Classy Ensemble...")
        # Get predictions from all models on eval set
        all_model_preds = []
        all_model_probs = []
        
        with torch.no_grad():
            for model in models:
                model_preds = []
                model_probs = []
                for batch in tqdm(dataloader, desc=f'Evaluating model {models.index(model)+1}/{len(models)}'):
                    if len(batch) == 2:
                        images, names = batch
                    else:
                        images, labels, names = batch
                    
                    images = images.to(device)
                    
                    if tta and tta_transforms:
                        tta_preds = []
                        for tta_transform in tta_transforms:
                            tta_images = []
                            for img in images:
                                img_np = img.cpu().permute(1, 2, 0).numpy()
                                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                                img_np = img_np.astype(np.uint8)
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
                    
                    preds = torch.argmax(avg_probs, dim=1)
                    model_preds.extend(preds.cpu().numpy())
                    model_probs.append(avg_probs.cpu().numpy())
                
                all_model_preds.append(model_preds)
                all_model_probs.append(np.vstack(model_probs))
        
        # Compute per-class F1 for each model
        class_f1_scores = np.zeros((len(models), num_classes))
        for i, (preds, probs) in enumerate(zip(all_model_preds, all_model_probs)):
            # Convert predictions to class names
            pred_labels = [idx_to_class[p] for p in preds]
            # Compute per-class F1
            for j, class_name in enumerate(class_names):
                y_true_binary = [1 if label == class_name else 0 for label in eval_labels]
                y_pred_binary = [1 if label == class_name else 0 for label in pred_labels]
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                class_f1_scores[i, j] = f1
        
        # Normalize F1 scores to get weights (per-class, per-model)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        class_f1_scores = class_f1_scores + epsilon
        # Normalize so each class has weights summing to 1 across models
        class_weights = class_f1_scores / class_f1_scores.sum(axis=0, keepdims=True)
        
        print(f"Per-class weights computed. Shape: {class_weights.shape}")
        print(f"Average weight per model: {class_weights.mean(axis=1)}")
    else:
        # Fallback to equal weights
        print("No eval labels provided, using equal weights for Classy Ensemble")
        class_weights = np.ones((len(models), num_classes)) / len(models)
    
    # Now make predictions with weighted ensemble
    all_ensemble_probs = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Classy Ensemble Predicting'):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch
            
            images = images.to(device)
            model_probs_list = []
            
            for model_idx, model in enumerate(models):
                if tta and tta_transforms:
                    tta_preds = []
                    for tta_transform in tta_transforms:
                        tta_images = []
                        for img in images:
                            img_np = img.cpu().permute(1, 2, 0).numpy()
                            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                            img_np = img_np.astype(np.uint8)
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
            
            # Weighted ensemble: Apply per-class weights
            batch_size = model_probs_list[0].shape[0]
            weighted_probs = np.zeros((batch_size, num_classes))
            
            for model_idx in range(len(models)):
                for class_idx in range(num_classes):
                    weight = class_weights[model_idx, class_idx]
                    weighted_probs[:, class_idx] += weight * model_probs_list[model_idx][:, class_idx]
            
            all_ensemble_probs.append(weighted_probs)
            all_names.extend(names)
    
    all_ensemble_probs = np.vstack(all_ensemble_probs)
    all_preds = np.argmax(all_ensemble_probs, axis=1)
    
    return all_preds, all_ensemble_probs, all_names, idx_to_class

def predict_ensemble_optimized(
    model_paths,
    dataloader,
    device,
    weights,
    tta=False,
    tta_transforms=None
):
    """
    Ensemble prediction with optimized weights.
    
    Args:
        model_paths: List of model paths
        dataloader: DataLoader
        device: Device
        weights: Array of shape (num_models,) for global weights or 'classy' for per-class
        tta: Whether to use TTA
        tta_transforms: TTA transforms
    
    Returns:
        predictions, probabilities, names, idx_to_class
    """
    # Check if weights is string 'classy' before comparing (avoid numpy array comparison error)
    if isinstance(weights, str) and weights == 'classy':
        # Use Classy Ensemble (needs eval_labels for weight computation)
        # For test set, use regular ensemble with equal weights
        return predict_ensemble(
            model_paths, dataloader, device, tta=tta, tta_transforms=tta_transforms
        )
    
    # Use weighted ensemble
    models = []
    idx_to_class_list = []
    
    for model_path in model_paths:
        model, idx_to_class = load_model(model_path, device)
        models.append(model)
        idx_to_class_list.append(idx_to_class)
    
    idx_to_class = idx_to_class_list[0]
    all_ensemble_probs = []
    all_names = []
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Optimized Ensemble'):
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
                            img_np = img.cpu().permute(1, 2, 0).numpy()
                            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                            img_np = img_np.astype(np.uint8)
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
            
            # Weighted average
            weighted_probs = np.zeros_like(model_probs_list[0])
            for i, probs in enumerate(model_probs_list):
                weighted_probs += weights[i] * probs
            
            all_ensemble_probs.append(weighted_probs)
            all_names.extend(names)
    
    all_ensemble_probs = np.vstack(all_ensemble_probs)
    all_preds = np.argmax(all_ensemble_probs, axis=1)
    
    return all_preds, all_ensemble_probs, all_names, idx_to_class

def predict_test_set(config, model_paths=None, use_ensemble=True, tta=True, use_classy_ensemble=True, 
                    class_weights=None, optimized_weights=None):
    """Predict on test set"""
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
    
    # Get TTA transforms if needed
    tta_transforms = None
    if tta:
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    # Make predictions - Support optimized weights (Session 3)
    if use_ensemble and model_paths:
        if optimized_weights is not None:
            # Use optimized weights from Session 3
            preds, probs, names, idx_to_class = predict_ensemble_optimized(
                model_paths, test_loader, config.DEVICE,
                weights=optimized_weights,
                tta=tta, tta_transforms=tta_transforms
            )
        elif use_classy_ensemble and class_weights is not None:
            # Use pre-computed class weights from eval set
            temp_model, temp_idx_to_class = load_model(model_paths[0], config.DEVICE)
            class_names = [temp_idx_to_class[i] for i in range(len(temp_idx_to_class))]
            del temp_model
            torch.cuda.empty_cache()
            
            preds, probs, names, idx_to_class = predict_ensemble_classy(
                model_paths, test_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms,
                eval_labels=None, class_names=class_names
            )
        else:
            preds, probs, names, idx_to_class = predict_ensemble(
                model_paths, test_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms
            )
    else:
        # Single model
        if model_paths is None:
            model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
        if not model_paths:
            raise ValueError("No model paths provided")
        
        model, idx_to_class = load_model(model_paths[0], config.DEVICE)
        preds, probs, names = predict_single_model(
            model, test_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms
        )
    
    # Convert predictions to class labels
    pred_labels = [idx_to_class[pred] for pred in preds]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': names,
        'labels': pred_labels
    })
    
    return submission

def predict_eval_set(config, model_paths=None, use_ensemble=True, tta=True, use_classy_ensemble=True, optimized_weights=None):
    """Predict on eval set with optional Classy Ensemble"""
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    
    # Get ground truth labels for Classy Ensemble
    eval_labels = eval_df['labels'].values.tolist() if 'labels' in eval_df.columns else None
    
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
    
    tta_transforms = None
    if tta:
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    if use_ensemble and model_paths:
        if use_classy_ensemble and eval_labels is not None:
            # Get class names from first model
            temp_model, temp_idx_to_class = load_model(model_paths[0], config.DEVICE)
            class_names = [temp_idx_to_class[i] for i in range(len(temp_idx_to_class))]
            del temp_model
            torch.cuda.empty_cache()
            
            preds, probs, names, idx_to_class = predict_ensemble_classy(
                model_paths, eval_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms,
                eval_labels=eval_labels, class_names=class_names
            )
        else:
            preds, probs, names, idx_to_class = predict_ensemble(
            model_paths, eval_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms
            )
    else:
        if model_paths is None:
            model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
        if not model_paths:
            raise ValueError("No model paths provided")
        
        model, idx_to_class = load_model(model_paths[0], config.DEVICE)
        preds, probs, names = predict_single_model(
            model, eval_loader, config.DEVICE, tta=tta, tta_transforms=tta_transforms
        )
    
    pred_labels = [idx_to_class[pred] for pred in preds]
    
    submission = pd.DataFrame({
        'ID': names,
        'labels': pred_labels
    })
    
    return submission

def main():
    """Main inference function"""
    config = Config()
    
    # Find all trained models
    model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
    
    if not model_paths:
        print("No trained models found. Please train models first.")
        return
    
    print(f"Found {len(model_paths)} models")
    
    # First, compute Classy Ensemble weights on eval set
    print("\n" + "="*60)
    print("Step 1: Computing Classy Ensemble weights on eval set...")
    print("="*60)
    eval_submission = predict_eval_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=True
    )
    eval_submission.to_csv(config.PRED_DIR / 'eval_predictions.csv', index=False)
    print(f"Eval predictions saved to {config.PRED_DIR / 'eval_predictions.csv'}")
    
    # Then predict on test set with Classy Ensemble
    print("\n" + "="*60)
    print("Step 2: Predicting on test set with Classy Ensemble...")
    print("="*60)
    test_submission = predict_test_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA,
        use_classy_ensemble=True
    )
    test_submission.to_csv(config.PRED_DIR / 'test_predictions.csv', index=False)
    print(f"Test predictions saved to {config.PRED_DIR / 'test_predictions.csv'}")

if __name__ == '__main__':
    main()

