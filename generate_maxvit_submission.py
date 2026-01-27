"""
Generate submission using MaxViT XLarge folds (0-2) with weighted TTA ensemble.
Single-method: weighted ensemble only (no method sweep).
Results saved to 'maxvit_xlarge_tf_384 only results'.
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
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_val_transforms, get_tta_transforms
from src.models import get_model
from sklearn.metrics import f1_score, accuracy_score


def load_model(model_path, device='cuda'):
    """Load trained MaxViT model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 10:
        raise ValueError(f"Model file too small ({file_size_mb:.1f}MB): {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except RuntimeError as e:
        if "central directory" in str(e) or "zip archive" in str(e):
            raise RuntimeError(
                f"Model file is corrupted/incomplete: {model_path}\n"
                f"File size: {file_size_mb:.1f}MB"
            ) from e
        raise

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    num_classes = len(class_to_idx)

    model = get_model('maxvit_xlarge_tf_384', num_classes=num_classes, pretrained=False)

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


def predict_with_tta(model, images, device, tta_transforms):
    """Predict probabilities with TTA for a batch."""
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
        with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
            outputs = model(tta_images)
            probs = F.softmax(outputs, dim=1)
        tta_preds.append(probs)
    return torch.stack(tta_preds).mean(0)


def predict_single_model(model_path, dataloader, device, tta_transforms):
    """Predict with a single model using full TTA."""
    model, idx_to_class = load_model(model_path, device)
    all_preds = []
    all_probs = []
    all_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Predicting {Path(model_path).name}'):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch
            images = images.to(device)

            avg_probs = predict_with_tta(model, images, device, tta_transforms)
            preds = torch.argmax(avg_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.append(avg_probs.cpu().numpy())
            all_names.extend(names)

    all_probs = np.vstack(all_probs)
    return all_preds, all_probs, all_names, idx_to_class


def predict_weighted_ensemble(model_paths, dataloader, device, tta_transforms, weights):
    """Weighted ensemble with full TTA."""
    models = []
    idx_to_class_list = []

    for model_path in model_paths:
        model, idx_to_class = load_model(model_path, device)
        models.append(model)
        idx_to_class_list.append(idx_to_class)

    assert all(idx_to_class_list[0] == idx_to_class for idx_to_class in idx_to_class_list), \
        "All models must have same class mapping"

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    all_probs = []
    all_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Weighted Ensemble Predicting'):
            if len(batch) == 2:
                images, names = batch
            else:
                images, labels, names = batch

            images = images.to(device)
            model_probs = []
            for model in models:
                avg_probs = predict_with_tta(model, images, device, tta_transforms)
                model_probs.append(avg_probs.cpu().numpy())

            weighted_probs = np.zeros_like(model_probs[0])
            for i, probs in enumerate(model_probs):
                weighted_probs += weights[i] * probs

            all_probs.append(weighted_probs)
            all_names.extend(names)

    all_probs = np.vstack(all_probs)
    all_preds = np.argmax(all_probs, axis=1)
    return all_preds, all_probs, all_names, idx_to_class_list[0]


def main():
    print("=" * 70)
    print("MaxViT XLarge (Folds 0-2) Weighted TTA Ensemble Submission")
    print("=" * 70)

    torch.backends.cudnn.benchmark = True

    config = Config()

    # Apply model-specific settings
    settings = config.MODEL_SPECIFIC_SETTINGS.get('maxvit_xlarge_tf_384', {})
    config.IMG_SIZE = settings.get('img_size', config.IMG_SIZE)

    output_dir = Path('maxvit_xlarge_tf_384 only results')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Use folds 0-2
    model_paths = []
    for fold in range(3):
        model_path = config.MODEL_DIR / f'maxvit_xlarge_tf_384_fold{fold}_best.pth'
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  Found: {model_path.name} ({size_mb:.1f}MB)")
            model_paths.append(model_path)
        else:
            print(f"  Missing: {model_path.name}")

    if len(model_paths) == 0:
        print("\nERROR: No MaxViT fold models found (0-2).")
        return

    print(f"\nUsing {len(model_paths)} models for weighted ensemble")

    tta_transforms = get_tta_transforms(config.IMG_SIZE)
    print(f"Using FULL TTA: {len(tta_transforms)} transforms")

    # Batch size tuned for H200 (safe default for TTA)
    batch_size = min(config.BATCH_SIZE, 32)
    print(f"Batch size: {batch_size}")

    # Eval set for weights
    eval_df = pd.read_csv(config.PHASE2_EVAL_CSV)
    eval_labels = eval_df['labels'].values.tolist()
    eval_dataset = WBCDataset(
        csv_path=config.PHASE2_EVAL_CSV,
        img_dir=config.PHASE2_EVAL_DIR,
        transform=get_val_transforms(config.IMG_SIZE),
        is_train=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
    )

    # Compute per-model F1 weights
    print("\nComputing per-model weights on eval set...")
    model_f1s = []
    for i, model_path in enumerate(model_paths):
        preds, _, _, idx_to_class = predict_single_model(
            model_path, eval_loader, config.DEVICE, tta_transforms
        )
        pred_labels = [idx_to_class[p] for p in preds]
        f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
        model_f1s.append(f1)
        print(f"  Fold {i} Macro F1: {f1:.5f}")

    weights = np.array(model_f1s, dtype=np.float32) + 1e-8
    weights = weights / weights.sum()
    print(f"\nEnsemble weights: {weights}")

    # Weighted ensemble on eval for reporting
    print("\nRunning weighted ensemble on eval set...")
    eval_preds, eval_probs, eval_names, idx_to_class = predict_weighted_ensemble(
        model_paths, eval_loader, config.DEVICE, tta_transforms, weights
    )
    eval_pred_labels = [idx_to_class[p] for p in eval_preds]
    eval_f1 = f1_score(eval_labels, eval_pred_labels, average='macro', zero_division=0)
    eval_acc = accuracy_score(eval_labels, eval_pred_labels)
    print(f"Eval Macro F1: {eval_f1:.5f}, Acc: {eval_acc:.5f}")

    eval_out = pd.DataFrame({'ID': eval_names, 'labels': eval_pred_labels})
    eval_out.to_csv(output_dir / 'eval_predictions.csv', index=False)

    # Test predictions
    print("\nGenerating test predictions...")
    test_dataset = WBCDataset(
        csv_path=config.PHASE2_TEST_CSV,
        img_dir=config.PHASE2_TEST_DIR,
        transform=get_val_transforms(config.IMG_SIZE),
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 4),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True) if config.NUM_WORKERS > 0 else False
    )

    test_preds, test_probs, test_names, idx_to_class = predict_weighted_ensemble(
        model_paths, test_loader, config.DEVICE, tta_transforms, weights
    )
    test_labels = [idx_to_class[p] for p in test_preds]
    submission = pd.DataFrame({'ID': test_names, 'labels': test_labels})
    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    # Save probabilities + weights
    prob_df = pd.DataFrame(test_probs, columns=[idx_to_class[i] for i in range(len(idx_to_class))])
    prob_df['ID'] = test_names
    prob_df = prob_df[['ID'] + [idx_to_class[i] for i in range(len(idx_to_class))]]
    prob_df.to_csv(output_dir / 'prediction_probabilities.csv', index=False)

    metrics = {
        'eval_macro_f1': float(eval_f1),
        'eval_accuracy': float(eval_acc),
        'weights': weights.tolist(),
        'model_paths': [str(p) for p in model_paths],
        'img_size': config.IMG_SIZE,
        'tta_transforms': len(tta_transforms),
        'batch_size': batch_size
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(f"Submission saved: {submission_path}")
    print(f"Results folder: {output_dir}")


if __name__ == '__main__':
    main()
