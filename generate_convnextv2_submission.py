"""
Standalone script to generate submission using convnextv2_large models from all 5 folds.
Results saved in 'convnextv2_large only results' folder.
This script does NOT modify any existing code or outputs.
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
project_root = Path(__file__).parent
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
    
    model_name = 'convnextv2_large'
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Handle torch.compile prefix (_orig_mod.) - remove it if present
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, idx_to_class

def predict_ensemble(model_paths, dataloader, device, tta=False, tta_transforms=None):
    """Make predictions with ensemble of models"""
    models = []
    idx_to_class_list = []
    
    print(f"Loading {len(model_paths)} convnextv2_large models...")
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

def main():
    """Generate submission using convnextv2_large models from all 5 folds"""
    print("="*70)
    print("ConvNeXt V2 Large 5-Fold Ensemble Submission Generator")
    print("="*70)
    
    # Initialize config
    config = Config()
    
    # Create output directory
    output_dir = Path('convnextv2_large only results')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Find all convnextv2_large fold models (fold 0-4)
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
        print(f"Expected models in: {config.MODEL_DIR}")
        print("Expected format: convnextv2_large_fold0_best.pth, convnextv2_large_fold1_best.pth, ...")
        return
    
    if len(model_paths) < 5:
        print(f"\nWarning: Only found {len(model_paths)} out of 5 expected models")
    
    print(f"\nUsing {len(model_paths)} models for ensemble")
    
    # Load test data
    print("\n" + "="*70)
    print("Loading test data...")
    print("="*70)
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
    
    # Get TTA transforms
    print("\nUsing Test Time Augmentation (TTA)...")
    tta_transforms = get_tta_transforms(config.IMG_SIZE)
    print(f"TTA transforms: {len(tta_transforms)}")
    
    # Make predictions
    print("\n" + "="*70)
    print("Generating predictions with 5-fold ensemble + TTA...")
    print("="*70)
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
    print(f"Total predictions: {len(submission)}")
    print(f"Unique IDs: {submission['ID'].nunique()}")
    print(f"Unique labels: {submission['labels'].nunique()}")
    print(f"\nClass distribution:")
    print(submission['labels'].value_counts().sort_index())
    
    # Save probabilities (optional, for analysis)
    prob_df = pd.DataFrame(probs, columns=[idx_to_class[i] for i in range(len(idx_to_class))])
    prob_df['ID'] = names
    prob_df = prob_df[['ID'] + [idx_to_class[i] for i in range(len(idx_to_class))]]
    prob_path = output_dir / 'prediction_probabilities.csv'
    prob_df.to_csv(prob_path, index=False)
    print(f"\n✓ Prediction probabilities saved to: {prob_path}")
    
    print("\n" + "="*70)
    print("Done! Results saved in 'convnextv2_large only results' folder")
    print("="*70)

if __name__ == '__main__':
    main()
