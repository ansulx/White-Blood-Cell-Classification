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
    # Check file exists and is readable
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check file size (models should be > 100MB typically)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 10:
        raise ValueError(f"Model file seems too small ({file_size_mb:.1f}MB): {model_path}. File may be corrupted or incomplete.")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except RuntimeError as e:
        if "central directory" in str(e) or "zip archive" in str(e):
            raise RuntimeError(
                f"Model file is corrupted or incomplete: {model_path}\n"
                f"File size: {file_size_mb:.1f}MB\n"
                f"Please re-copy this file from the source. The file transfer may have been interrupted."
            ) from e
        raise
    
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
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  Found: {model_path.name} ({file_size_mb:.1f}MB)")
            model_paths.append(model_path)
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
    
    # Use larger batch size for faster inference (if memory allows)
    fast_batch_size = min(config.BATCH_SIZE * 2, 96)  # Double batch size, cap at 96
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=fast_batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
    )
    
    # Use fast TTA (5 transforms instead of 15 for ~3x speedup)
    print("\nUsing Fast Test Time Augmentation (TTA) - 5 transforms...")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    base_transform = [
        A.LongestMaxSize(max_size=config.IMG_SIZE, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=config.IMG_SIZE, min_width=config.IMG_SIZE, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
    ]
    normalize = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    # Fast TTA: 5 essential transforms (original, h-flip, v-flip, rotate90, transpose)
    tta_transforms = [
        A.Compose(base_transform + normalize),  # 1. Original
        A.Compose(base_transform + [A.HorizontalFlip(p=1.0)] + normalize),  # 2. H-flip
        A.Compose(base_transform + [A.VerticalFlip(p=1.0)] + normalize),  # 3. V-flip
        A.Compose(base_transform + [A.RandomRotate90(p=1.0)] + normalize),  # 4. Rotate 90
        A.Compose(base_transform + [A.Transpose(p=1.0)] + normalize),  # 5. Transpose
    ]
    print(f"TTA transforms: {len(tta_transforms)} (Fast mode - ~3x faster than 15 transforms)")
    print(f"Batch size: {fast_batch_size} (increased for faster processing)")
    
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
