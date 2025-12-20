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
    
    # Infer model name from path
    model_name = Config.MODEL_NAME
    if 'efficientnet' in str(model_path):
        model_name = 'efficientnet_b4'
    elif 'convnext' in str(model_path):
        model_name = 'convnext_base'
    
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
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

def predict_test_set(config, model_paths=None, use_ensemble=True, tta=True):
    """Predict on test set"""
    config = Config()
    
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
        pin_memory=config.PIN_MEMORY
    )
    
    # Get TTA transforms if needed
    tta_transforms = None
    if tta:
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    # Make predictions
    if use_ensemble and model_paths:
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

def predict_eval_set(config, model_paths=None, use_ensemble=True, tta=True):
    """Predict on eval set"""
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
    
    tta_transforms = None
    if tta:
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
    
    if use_ensemble and model_paths:
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
    
    # Predict on test set
    print("\nPredicting on test set...")
    test_submission = predict_test_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA
    )
    test_submission.to_csv(config.PRED_DIR / 'test_predictions.csv', index=False)
    print(f"Test predictions saved to {config.PRED_DIR / 'test_predictions.csv'}")
    
    # Predict on eval set
    print("\nPredicting on eval set...")
    eval_submission = predict_eval_set(
        config,
        model_paths=model_paths,
        use_ensemble=True,
        tta=config.TTA
    )
    eval_submission.to_csv(config.PRED_DIR / 'eval_predictions.csv', index=False)
    print(f"Eval predictions saved to {config.PRED_DIR / 'eval_predictions.csv'}")

if __name__ == '__main__':
    main()

