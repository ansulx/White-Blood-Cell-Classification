"""
Training script for WBC-Bench-2026 competition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import json
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_train_transforms, get_val_transforms
from src.models import get_model, get_loss_fn
from src.metrics import compute_all_metrics, print_classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

def init_distributed_mode():
    """Initialize torch.distributed if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def is_checkpoint_valid(model_path):
    """Check if a checkpoint exists and can be loaded."""
    if not model_path.exists():
        return False
    if model_path.stat().st_size == 0:
        return False
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
    except Exception:
        return False

def compute_class_weights(labels, method='balanced', power=1.0):
    """
    Compute class weights for imbalanced dataset with stronger weighting for rare classes
    
    Args:
        labels: Array of class labels
        method: 'balanced' or 'inverse_freq'
        power: Power to apply to inverse frequency (higher = stronger weighting)
    """
    unique_labels = np.unique(labels)
    
    if method == 'balanced':
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
    else:  # inverse_freq
        from collections import Counter
        label_counts = Counter(labels)
        total = len(labels)
        class_weights = []
        for label in unique_labels:
            freq = label_counts[label] / total
            # Inverse frequency with power scaling
            weight = (1.0 / freq) ** power
            class_weights.append(weight)
        # Normalize to have mean of 1
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.mean()
    
    return dict(zip(unique_labels, class_weights))

def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply cutmix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, 
                gradient_clip_value=None, use_mixup=False, mixup_alpha=0.4,
                use_cutmix=False, cutmix_alpha=1.0, use_amp=False, scaler=None, config=None,
                is_main_process=True):
    """Train for one epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', disable=not is_main_process)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)  # Non-blocking transfer
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast(enabled=use_amp):
            # Apply Mixup or CutMix (with configurable probability)
            mixup_cutmix_prob = getattr(config, 'MIXUP_CUTMIX_PROB', 0.5) if config is not None else 0.5
            
            if use_cutmix and np.random.rand() < mixup_cutmix_prob:
                images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            elif use_mixup and np.random.rand() < mixup_cutmix_prob:
                images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        # Backward pass with mixed precision
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, class_names=None, use_amp=False, is_main_process=True):
    """Validate model and compute competition metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for debugging
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validating', disable=not is_main_process):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Get probabilities for debugging
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # Debug: Print prediction confidence
    all_probs = np.array(all_probs)
    avg_confidence = np.mean(np.max(all_probs, axis=1))
    print(f"  Validation Debug - Avg Prediction Confidence: {avg_confidence:.4f}")
    print(f"  Validation Debug - Max Confidence: {np.max(all_probs):.4f}, Min Confidence: {np.min(np.max(all_probs, axis=1)):.4f}")
    
    # Debug: Check if model is predicting same class always
    unique_preds = len(np.unique(all_preds))
    print(f"  Validation Debug - Unique predicted classes: {unique_preds}/13")
    
    # Debug: Per-class accuracy to identify which classes are failing
    # CRITICAL: Special monitoring for BL (Blast) - most critical rare class
    if class_names:
        from collections import Counter
        from sklearn.metrics import f1_score
        pred_counter = Counter(all_preds)
        label_counter = Counter(all_labels)
        print(f"  Validation Debug - Prediction distribution:")
        
        # Track BL class specifically (most critical rare class)
        bl_class_idx = None
        for i, cls_name in enumerate(class_names):
            if cls_name == 'BL':
                bl_class_idx = i
                break
        
        for i, cls_name in enumerate(class_names):
            pred_count = pred_counter.get(i, 0)
            label_count = label_counter.get(i, 0)
            if label_count > 0:
                correct = sum(1 for p, l in zip(all_preds, all_labels) if p == i and l == i)
                acc = 100 * correct / label_count if label_count > 0 else 0
                
                # Compute per-class F1
                y_true_binary = [1 if l == i else 0 for l in all_labels]
                y_pred_binary = [1 if p == i else 0 for p in all_preds]
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                # Highlight BL class and rare classes
                marker = " ⚠️ [BL - CRITICAL]" if cls_name == 'BL' else (" [RARE]" if label_count < 100 else "")
                print(f"    {cls_name}: {correct}/{label_count} correct ({acc:.1f}%) | F1: {f1:.4f} | Predicted: {pred_count} times{marker}")
        
        # Special warning if BL performance is poor
        if bl_class_idx is not None:
            bl_label_count = label_counter.get(bl_class_idx, 0)
            if bl_label_count > 0:
                bl_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == bl_class_idx and l == bl_class_idx)
                bl_acc = 100 * bl_correct / bl_label_count
                bl_y_true = [1 if l == bl_class_idx else 0 for l in all_labels]
                bl_y_pred = [1 if p == bl_class_idx else 0 for p in all_preds]
                bl_f1 = f1_score(bl_y_true, bl_y_pred, zero_division=0)
                if bl_f1 < 0.5 or bl_acc < 50:
                    print(f"  ⚠️ WARNING: BL (Blast) class performance is LOW - F1: {bl_f1:.4f}, Acc: {bl_acc:.1f}%")
                    print(f"     Consider: Increasing rare class augmentation, adjusting class weights, or using focal loss")
    
    # Compute competition metrics
    metrics = compute_all_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        class_names=class_names
    )
    
    return epoch_loss, epoch_acc, all_preds, all_labels, metrics

def save_training_results(fold, history, config, best_val_macro_f1, best_val_metrics=None):
    """Save training history, plots, and metrics"""
    
    # Create directories
    plots_dir = config.LOG_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    metrics_dir = config.LOG_DIR / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_path = metrics_dir / f'{config.MODEL_NAME}_fold{fold}_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Save summary JSON
    best_epoch_idx = np.argmax(history['val_macro_f1']) if history['val_macro_f1'] else -1
    
    # CRITICAL: Extract BL class performance from best metrics
    bl_performance = None
    if best_val_metrics and best_val_metrics.get('per_class') and 'BL' in best_val_metrics['per_class']:
        bl_performance = best_val_metrics['per_class']['BL']
    
    summary = {
        'fold': fold,
        'model_name': config.MODEL_NAME,
        'best_val_macro_f1': best_val_macro_f1,  # Primary competition metric
        'best_val_acc': history['val_acc'][best_epoch_idx] if best_epoch_idx >= 0 and history['val_acc'] else None,
        'bl_performance': bl_performance,  # Track BL class (most critical rare class)
        'total_epochs': len(history['epoch']),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
        'final_val_macro_f1': history['val_macro_f1'][-1] if history['val_macro_f1'] else None,
        'best_epoch': history['epoch'][best_epoch_idx] if best_epoch_idx >= 0 else None,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'img_size': config.IMG_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'label_smoothing': config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0,
            'use_mixup': config.USE_MIXUP,
            'use_cutmix': config.USE_CUTMIX,
        }
    }
    
    summary_path = metrics_dir / f'{config.MODEL_NAME}_fold{fold}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Create plots
    epochs = history['epoch']
    
    # Get best epoch info for display
    best_epoch_idx = np.argmax(history['val_macro_f1']) if history['val_macro_f1'] else -1
    best_val_acc_display = history['val_acc'][best_epoch_idx] if best_epoch_idx >= 0 and history['val_acc'] else 0.0
    
    # 1. Loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Curves - Fold {fold}\nBest Macro-F1: {best_val_macro_f1:.4f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Curves - Fold {fold}\nBest Val Acc: {best_val_acc_display:.2f}%', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / f'{config.MODEL_NAME}_fold{fold}_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {plot_path}")
    
    # 3. Learning rate curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['learning_rate'], label='Learning Rate', marker='o', linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'Learning Rate Schedule - Fold {fold}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    lr_plot_path = plots_dir / f'{config.MODEL_NAME}_fold{fold}_lr.png'
    plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved LR plot to {lr_plot_path}")
    
    # 4. Combined plot (all metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', marker='o', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Val', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rate'], marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap (train acc - val acc)
    if len(history['train_acc']) == len(history['val_acc']):
        gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(epochs, gap, marker='o', linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Acc - Val Acc (%)')
        axes[1, 1].set_title('Overfitting Gap (Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Summary - Fold {fold} | Model: {config.MODEL_NAME} | Best Macro-F1: {best_val_macro_f1:.4f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    combined_plot_path = plots_dir / f'{config.MODEL_NAME}_fold{fold}_summary.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {combined_plot_path}")

def train_fold(fold, train_df, val_df, config):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")

    distributed = getattr(config, 'DISTRIBUTED', False)
    main_process = is_main_process()
    
    # Determine image directory based on split
    # Check if images are in phase1, phase2/train, or phase2/eval
    def get_img_path(row):
        img_name = row['ID']
        phase1_path = config.PHASE1_DIR / img_name
        phase2_train_path = config.PHASE2_TRAIN_DIR / img_name
        phase2_eval_path = config.PHASE2_EVAL_DIR / img_name
        if phase1_path.exists():
            return config.PHASE1_DIR
        elif phase2_train_path.exists():
            return config.PHASE2_TRAIN_DIR
        elif phase2_eval_path.exists():
            return config.PHASE2_EVAL_DIR
        else:
            # Try phase2/train as default, but this should rarely happen
            return config.PHASE2_TRAIN_DIR
    
    # Identify rare classes for class-aware augmentation
    from src.data.dataset import identify_rare_classes
    
    if config.USE_CLASS_AWARE_AUG:
        rare_classes = identify_rare_classes(train_df, threshold=config.RARE_CLASS_THRESHOLD)
        print(f"\nRare classes identified (threshold={config.RARE_CLASS_THRESHOLD}): {sorted(rare_classes)}")
        class_counts = train_df['labels'].value_counts()
        print("Class distribution:")
        for cls in sorted(class_counts.index):
            count = class_counts[cls]
            marker = " [RARE]" if cls in rare_classes else ""
            print(f"  {cls}: {count} samples{marker}")
    else:
        rare_classes = set()
    
    # Create custom dataset class that handles both directories
    class CombinedDataset(WBCDataset):
        def __init__(self, df, transform, is_train=True, rare_classes=None, class_aware_aug=False):
            # Create temporary CSV for parent initialization
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_csv.name, index=False)
            temp_csv_path = temp_csv.name
            temp_csv.close()
            
            # Initialize parent class (will be overridden but needed for inheritance)
            super().__init__(
                csv_path=temp_csv_path,
                img_dir=config.PHASE1_DIR,  # Dummy, we use img_paths
                transform=transform,
                is_train=is_train,
                rare_classes=rare_classes,
                class_aware_aug=class_aware_aug
            )
            
            # Override with actual data
            self.df = df.copy()
            self.rare_classes = rare_classes if rare_classes is not None else set()
            self.class_aware_aug = class_aware_aug
            
            # Determine image directories - check phase1, phase2/train, and phase2/eval
            self.img_paths = []
            for _, row in df.iterrows():
                img_name = row['ID']
                phase1_path = config.PHASE1_DIR / img_name
                phase2_train_path = config.PHASE2_TRAIN_DIR / img_name
                phase2_eval_path = config.PHASE2_EVAL_DIR / img_name
                if phase1_path.exists():
                    self.img_paths.append(config.PHASE1_DIR)
                elif phase2_train_path.exists():
                    self.img_paths.append(config.PHASE2_TRAIN_DIR)
                elif phase2_eval_path.exists():
                    self.img_paths.append(config.PHASE2_EVAL_DIR)
                else:
                    # Default to phase2/train (should rarely happen)
                    self.img_paths.append(config.PHASE2_TRAIN_DIR)
            
            if is_train:
                self.labels = df['labels'].values
                self.unique_classes = sorted(df['labels'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
                self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                self.num_classes = len(self.unique_classes)
            
            # Clean up temp file
            try:
                os.unlink(temp_csv_path)
            except:
                pass
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_name = self.df.iloc[idx]['ID']
            img_dir = self.img_paths[idx]
            img_path = img_dir / img_name
            
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply transforms with class-aware augmentation
            if self.transform:
                # Class-aware augmentation: use different transform for rare classes
                if self.class_aware_aug and self.is_train:
                    label = self.df.iloc[idx]['labels']
                    is_rare = label in self.rare_classes
                    
                    if isinstance(self.transform, dict):
                        # Transform is a dict with 'normal' and 'rare' keys
                        transform_to_use = self.transform['rare'] if is_rare else self.transform['normal']
                    else:
                        # Single transform - use as is (backward compatibility)
                        transform_to_use = self.transform
                else:
                    # No class-aware augmentation or not training
                    transform_to_use = self.transform
                
                if isinstance(transform_to_use, A.Compose):
                    transformed = transform_to_use(image=image)
                    image = transformed['image']
                else:
                    image = transform_to_use(image)
            
            if self.is_train:
                label = self.df.iloc[idx]['labels']
                label_idx = self.class_to_idx[label]
                return image, label_idx, img_name
            else:
                return image, img_name
    
    # Create augmentation transforms
    if config.USE_CLASS_AWARE_AUG:
        # Create separate transforms for normal and rare classes
        normal_transform = get_train_transforms(
            config.IMG_SIZE, 
            is_rare_class=False,
            domain_adaptation=True  # Enable domain adaptation
        )
        rare_transform = get_train_transforms(
            config.IMG_SIZE, 
            is_rare_class=True,
            domain_adaptation=True  # Enable domain adaptation
        )
        train_transform = {'normal': normal_transform, 'rare': rare_transform}
    else:
        # Use standard augmentation for all classes
        train_transform = get_train_transforms(
            config.IMG_SIZE, 
            is_rare_class=False,
            domain_adaptation=True  # Enable domain adaptation
        )
    
    # CRITICAL FIX: Create train dataset first to establish class mapping
    train_dataset = CombinedDataset(
        train_df, 
        train_transform, 
        is_train=True,
        rare_classes=rare_classes,
        class_aware_aug=config.USE_CLASS_AWARE_AUG
    )
    
    # CRITICAL FIX: Val dataset MUST use the SAME class_to_idx as train_dataset
    # Create val dataset with shared class mapping
    val_dataset = CombinedDataset(
        val_df, 
        get_val_transforms(config.IMG_SIZE), 
        is_train=True,
        rare_classes=set(),  # No class-aware aug for validation
        class_aware_aug=False
    )
    
    # CRITICAL FIX: Ensure validation uses same class mapping as training
    val_dataset.unique_classes = train_dataset.unique_classes
    val_dataset.class_to_idx = train_dataset.class_to_idx
    val_dataset.idx_to_class = train_dataset.idx_to_class
    val_dataset.num_classes = train_dataset.num_classes
    
    # Verify all validation labels exist in training classes
    val_labels = set(val_df['labels'].unique())
    train_labels = set(train_df['labels'].unique())
    if val_labels != train_labels:
        print(f"WARNING: Validation has different classes!")
        print(f"Train classes: {sorted(train_labels)}")
        print(f"Val classes: {sorted(val_labels)}")
        missing_in_train = val_labels - train_labels
        if missing_in_train:
            print(f"ERROR: Validation has classes not in training: {missing_in_train}")
            raise ValueError("Validation set contains classes not in training set!")
    
    # Debug: Print label mapping verification
    if main_process:
        print(f"\nLabel Mapping Verification:")
        print(f"  Train classes: {len(train_dataset.unique_classes)} classes")
        print(f"  Val classes: {len(val_dataset.unique_classes)} classes")
        print(f"  Class mapping: {train_dataset.class_to_idx}")
        
        # Verify a few samples
        print(f"\nSample Verification (first 3 train samples):")
        for i in range(min(3, len(train_dataset))):
            _, label_idx, img_name = train_dataset[i]
            label_name = train_dataset.idx_to_class[label_idx]
            print(f"  {img_name}: label_idx={label_idx}, label_name={label_name}")
        
        print(f"\nSample Verification (first 3 val samples):")
        for i in range(min(3, len(val_dataset))):
            _, label_idx, img_name = val_dataset[i]
            label_name = val_dataset.idx_to_class[label_idx]
            print(f"  {img_name}: label_idx={label_idx}, label_name={label_name}")
    
    # Compute class weights with moderate weighting for rare classes
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(
            train_df['labels'].values, 
            method='balanced',
            power=1.0  # Reduced from 1.5 - extreme weights causing instability
        )
        if main_process:
            print(f"\nClass Weights (for imbalanced handling):")
            for cls in sorted(class_weights.keys()):
                print(f"  {cls}: {class_weights[cls]:.4f}")

    train_sampler = None
    loader_kwargs = {
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = getattr(config, 'PREFETCH_FACTOR', 4)
        loader_kwargs['persistent_workers'] = (
            getattr(config, 'PERSISTENT_WORKERS', True) if not distributed else False
        )
        if distributed:
            loader_kwargs['timeout'] = 600
            if main_process:
                print("DDP: persistent_workers disabled and timeout set to avoid worker hangs.")

    if distributed:
        if config.USE_CLASS_WEIGHTS and main_process:
            print("Distributed training: WeightedRandomSampler is disabled; using class weights in loss only.")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=getattr(config, 'WORLD_SIZE', 1),
            rank=getattr(config, 'RANK', 0),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            **loader_kwargs
        )
    else:
        if config.USE_CLASS_WEIGHTS:
            weights = [class_weights[label] for label in train_df['labels'].values]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                sampler=sampler,
                **loader_kwargs
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                **loader_kwargs
            )
    
    val_loader = None
    if main_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            **loader_kwargs
        )
    
    # Create model with moderate regularization (reduced to improve validation)
    # Support model-specific drop_path_rate from MODEL_SPECIFIC_SETTINGS
    model_specific_drop_path = None
    if config.MODEL_NAME in config.MODEL_SPECIFIC_SETTINGS:
        model_specific_drop_path = config.MODEL_SPECIFIC_SETTINGS[config.MODEL_NAME].get('drop_path_rate')
    
    # Use model-specific drop_path_rate if available, otherwise use defaults
    if config.MODEL_NAME.startswith('convnext'):
        # ConvNeXt uses drop_path_rate only (no drop_rate)
        drop_path = model_specific_drop_path if model_specific_drop_path is not None else 0.2
        model = get_model(
            model_name=config.MODEL_NAME,
            num_classes=train_dataset.num_classes,
            pretrained=config.PRETRAINED,
            drop_rate=None,  # ConvNeXt doesn't use drop_rate
            drop_path_rate=drop_path
        ).to(config.DEVICE)
    else:
        # Other models (EfficientNet, Swin, MaxViT, etc.)
        drop_path = model_specific_drop_path if model_specific_drop_path is not None else 0.2
        model = get_model(
            model_name=config.MODEL_NAME,
            num_classes=train_dataset.num_classes,
            pretrained=config.PRETRAINED,
            drop_rate=0.3,  # Reduced from 0.4
            drop_path_rate=drop_path
        ).to(config.DEVICE)

    if distributed:
        model = DDP(
            model,
            device_ids=[getattr(config, 'LOCAL_RANK', 0)],
            output_device=getattr(config, 'LOCAL_RANK', 0)
        )
    
    # Compile model for faster training (PyTorch 2.0+)
    # Skip compilation for MaxViT models (they have compatibility issues with torch.compile)
    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile') and not distributed:
        if 'maxvit' in config.MODEL_NAME.lower():
            print("Skipping torch.compile for MaxViT (compatibility issue)")
        else:
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled with torch.compile for faster training")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Loss and optimizer
    label_smoothing = config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0
    
    # Use weighted loss with label smoothing for better rare class handling
    # CRITICAL FIX: Cap extreme class weights to prevent instability
    if config.USE_CLASS_WEIGHTS:
        class_weights_list = [class_weights[cls] for cls in train_dataset.unique_classes]
        # Convert to tensor and normalize
        class_weights_tensor = torch.FloatTensor(class_weights_list).to(config.DEVICE)
        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()  # Normalize
        
        # CRITICAL: Cap weights to max 3x to prevent extreme values causing validation issues
        max_weight = 3.0
        class_weights_tensor = torch.clamp(class_weights_tensor, min=0.1, max=max_weight)
        # Renormalize after clamping
        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()
        
        print(f"\nLoss Configuration:")
        print(f"  Using weighted CrossEntropyLoss (weights capped at {max_weight}x)")
        print(f"  Label smoothing: {label_smoothing}")
        print(f"  Class weights (normalized & capped): {class_weights_tensor.cpu().numpy()}")
        
        criterion = get_loss_fn(
            'ce', 
            class_weights=class_weights_tensor, 
            device=config.DEVICE,
            label_smoothing=label_smoothing
        )
    else:
        criterion = get_loss_fn(
            'focal', 
            device=config.DEVICE,
            label_smoothing=label_smoothing
        )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler with warmup and slower cosine annealing (for better validation)
    if config.USE_WARMUP:
        # Warmup + Slower Cosine Annealing with minimum LR
        def lr_lambda(epoch):
            if epoch < config.WARMUP_EPOCHS:
                # Linear warmup
                return (epoch + 1) / config.WARMUP_EPOCHS
            else:
                # Slower cosine annealing - keep LR higher longer for validation learning
                progress = (epoch - config.WARMUP_EPOCHS) / (config.NUM_EPOCHS - config.WARMUP_EPOCHS)
                min_lr_ratio = 1e-6 / config.LEARNING_RATE  # Minimum LR as ratio
                # Slower decay - multiply progress by 0.8 to slow down
                adjusted_progress = progress * 0.8  # Slower decay
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * adjusted_progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        if is_main_process():
            print(f"\nLearning Rate Schedule:")
            print(f"  Warmup epochs: {config.WARMUP_EPOCHS}")
            print(f"  Initial LR: {config.LEARNING_RATE}")
            print(f"  Schedule: Linear warmup → Slow Cosine annealing (for better validation)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_val_macro_f1 = 0.0  # Track Macro-F1 (primary competition metric)
    best_val_acc = 0.0  # Also track accuracy for reference
    best_val_metrics = None
    patience_counter = 0
    
    # Track training history (main process only)
    history = None
    if main_process:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],  # Primary competition metric
            'learning_rate': [],
            'epoch': []
        }
    
    for epoch in range(config.NUM_EPOCHS):
        if main_process:
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Gradient clipping value
        grad_clip = config.GRADIENT_CLIP_VALUE if config.USE_GRADIENT_CLIPPING else None
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scheduler,
            gradient_clip_value=grad_clip,
            use_mixup=config.USE_MIXUP,
            mixup_alpha=config.MIXUP_ALPHA,
            use_cutmix=config.USE_CUTMIX,
            cutmix_alpha=config.CUTMIX_ALPHA,
            use_amp=config.USE_MIXED_PRECISION,
            scaler=scaler,
            config=config,
            is_main_process=main_process
        )
        
        # Get class names for metrics
        class_names = list(train_dataset.idx_to_class.values())
        
        if main_process:
            val_loss, val_acc, _, _, val_metrics = validate(
                model, val_loader, criterion, config.DEVICE, 
                class_names=class_names, use_amp=config.USE_MIXED_PRECISION,
                is_main_process=main_process
            )
        else:
            val_loss, val_acc, val_metrics = 0.0, 0.0, {'macro_f1': 0.0}

        # Ensure all ranks wait for validation before proceeding
        if distributed:
            dist.barrier()
        
        # Step scheduler after each epoch
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        macro_f1 = val_metrics['macro_f1']
        
        if main_process:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val Macro-F1: {macro_f1:.4f} (Primary Metric)")
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Track history
        if history is not None:
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_macro_f1'].append(macro_f1)
            history['learning_rate'].append(current_lr)
        
        # Save best model based on Macro-F1 (primary metric)
        if main_process:
            if macro_f1 > best_val_macro_f1 + config.EARLY_STOPPING_MIN_DELTA:
                best_val_macro_f1 = macro_f1
                best_val_acc = val_acc  # Also track accuracy
                best_val_metrics = val_metrics
                patience_counter = 0
                model_path = config.MODEL_DIR / f'{config.MODEL_NAME}_fold{fold}_best.pth'
                
                # CRITICAL: Extract BL class performance for monitoring
                bl_performance = None
                if val_metrics.get('per_class') and 'BL' in val_metrics['per_class']:
                    bl_performance = val_metrics['per_class']['BL']
                
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_macro_f1': macro_f1,
                    'val_metrics': val_metrics,
                    'bl_performance': bl_performance,  # Track BL class specifically
                    'epoch': epoch,
                    'class_to_idx': train_dataset.class_to_idx,
                    'idx_to_class': train_dataset.idx_to_class,
                }, model_path)
                
                # Print BL performance if available
                bl_info = ""
                if bl_performance:
                    bl_info = f" | BL F1: {bl_performance['f1']:.4f}"
                print(f"✓ Saved best model - Val Acc: {val_acc:.2f}%, Macro-F1: {macro_f1:.4f}{bl_info}")
            else:
                patience_counter += 1
        
        # Early stopping (broadcast to all ranks)
        stop_training = 0
        if main_process and patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            stop_training = 1
        if distributed:
            stop_tensor = torch.tensor(stop_training, device=config.DEVICE)
            dist.broadcast(stop_tensor, src=0)
            stop_training = int(stop_tensor.item())
        if stop_training == 1:
            break
    
    # Save training history and plots
    if main_process and history is not None:
        save_training_results(fold, history, config, best_val_macro_f1, best_val_metrics)
    
    return best_val_macro_f1, best_val_metrics

def train_all_ensemble_models(config):
    """
    Train all ensemble models sequentially with optimized settings for each.
    This eliminates the need to manually change config for each model.
    """
    main_process = is_main_process()
    model_list = list(config.ENSEMBLE_MODELS)
    if 'maxvit_xlarge_tf_384' in model_list:
        model_list = ['maxvit_xlarge_tf_384'] + [m for m in model_list if m != 'maxvit_xlarge_tf_384']

    if main_process:
        print("\n" + "="*60)
        print("AUTO-TRAINING ALL ENSEMBLE MODELS")
        print("="*60)
        print(f"Models to train: {len(model_list)}")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
        print("="*60 + "\n")
    
    # Store original settings
    original_model_name = config.MODEL_NAME
    original_lr = config.LEARNING_RATE
    original_batch_size = config.BATCH_SIZE
    original_img_size = config.IMG_SIZE
    
    all_results = []
    
    for model_idx, model_name in enumerate(model_list, 1):
        if main_process:
            print("\n" + "="*60)
            print(f"Training Model {model_idx}/{len(model_list)}: {model_name}")
            print("="*60)
        
        # Update model name
        config.MODEL_NAME = model_name

        # Skip model entirely if all folds already exist
        existing_folds = [
            fold for fold in range(config.N_FOLDS)
            if is_checkpoint_valid(config.MODEL_DIR / f'{model_name}_fold{fold}_best.pth')
        ]
        if len(existing_folds) == config.N_FOLDS:
            if main_process:
                print(f"All {config.N_FOLDS} folds already exist for {model_name}. Skipping training.")
            continue
        
        # Apply model-specific settings if available
        if model_name in config.MODEL_SPECIFIC_SETTINGS:
            settings = config.MODEL_SPECIFIC_SETTINGS[model_name]
            config.LEARNING_RATE = settings.get('learning_rate', config.LEARNING_RATE)
            config.BATCH_SIZE = settings.get('batch_size', config.BATCH_SIZE)
            # MaxViT models need specific image sizes (384 for maxvit_xlarge_tf_384)
            if 'img_size' in settings:
                config.IMG_SIZE = settings.get('img_size', config.IMG_SIZE)
            if main_process:
                print(f"Applied model-specific settings:")
                print(f"  Learning Rate: {config.LEARNING_RATE}")
                print(f"  Batch Size: {config.BATCH_SIZE}")
                if 'img_size' in settings:
                    print(f"  Image Size: {config.IMG_SIZE}")
        else:
            if main_process:
                print(f"Using default settings:")
                print(f"  Learning Rate: {config.LEARNING_RATE}")
                print(f"  Batch Size: {config.BATCH_SIZE}")
        
        try:
            # Train this model
            val_macro_f1, val_metrics = train_single_model(config)
            
            all_results.append({
                'model_name': model_name,
                'val_macro_f1': val_macro_f1,
                'val_metrics': val_metrics
            })
            
            if main_process:
                print(f"\n✓ Model {model_idx} completed - Macro-F1: {val_macro_f1:.4f}")
            
        except Exception as e:
            if main_process:
                print(f"\n✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
            continue
        finally:
            # Restore settings after each model (important for img_size)
            config.LEARNING_RATE = original_lr
            config.BATCH_SIZE = original_batch_size
            config.IMG_SIZE = original_img_size
    
    # Restore original settings
    config.MODEL_NAME = original_model_name
    config.LEARNING_RATE = original_lr
    config.BATCH_SIZE = original_batch_size
    config.IMG_SIZE = original_img_size
    
    # Print summary
    if main_process:
        print("\n" + "="*60)
        print("ENSEMBLE TRAINING SUMMARY")
        print("="*60)
        for i, result in enumerate(all_results, 1):
            print(f"{i}. {result['model_name']}: Macro-F1 = {result['val_macro_f1']:.4f}")
    
    if all_results and main_process:
        mean_f1 = np.mean([r['val_macro_f1'] for r in all_results])
        print(f"\nMean Macro-F1 across all models: {mean_f1:.4f}")
        print("="*60)
    
    return all_results

def train_single_model(config):
    """
    Train a single model (extracted from main() for reuse)
    Supports pseudo-labels automatically if available
    """
    # Load data - support pseudo-labels
    if is_main_process():
        print("Loading data...")
    
    # Check if pseudo-labels exist (Session 2 feature)
    merged_train_path = config.PRED_DIR / 'merged_train_with_pseudo.csv'
    use_pseudo_labels = merged_train_path.exists()
    
    if use_pseudo_labels:
        if is_main_process():
            print("✓ Found merged training data with pseudo-labels (Session 2)")
        all_train = pd.read_csv(merged_train_path)
        if is_main_process():
            print(f"  Total samples (including pseudo-labels): {len(all_train)}")
    else:
        if is_main_process():
            print("Using original training data only (Session 1)")
        phase2_train = pd.read_csv(config.PHASE2_TRAIN_CSV)
        
        # CRITICAL FIX: Include Phase 2 Eval set in training (README says "may be used for training")
        # This adds ~5,350 additional images (10% more data) for better performance
        phase2_eval = pd.read_csv(config.PHASE2_EVAL_CSV)
        if is_main_process():
            print(f"Including Phase 2 Eval set: {len(phase2_eval)} additional samples")
        
        # Combine phase1 and phase2 for training
        phase1 = pd.read_csv(config.PHASE1_CSV)
        phase1 = phase1[phase1['split'] == 'phase1_train']
        phase1 = phase1[['ID', 'labels']].copy()
        phase1['split'] = 'phase1_train'
        
        # Combine ALL available training data (Phase 1 + Phase 2 Train + Phase 2 Eval)
        all_train = pd.concat([
            phase2_train[['ID', 'labels']],
            phase1[['ID', 'labels']],
            phase2_eval[['ID', 'labels']]  # CRITICAL: Include eval set for maximum training data
        ], ignore_index=True)
    
    if is_main_process():
        print(f"Total training samples: {len(all_train)}")
        print(f"Class distribution:\n{all_train['labels'].value_counts()}")

    
    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )
    
    # Skip folds that already have saved checkpoints
    existing_folds = set()
    for fold in range(config.N_FOLDS):
        model_path = config.MODEL_DIR / f'{config.MODEL_NAME}_fold{fold}_best.pth'
        if is_checkpoint_valid(model_path):
            existing_folds.add(fold)
    if existing_folds and is_main_process():
        print(f"Found existing checkpoints for folds: {sorted(existing_folds)}")
        print("These folds will be skipped and not retrained.")

    fold_scores = []
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_train, all_train['labels'])):
        train_df = all_train.iloc[train_idx].reset_index(drop=True)
        val_df = all_train.iloc[val_idx].reset_index(drop=True)

        # Skip fold if checkpoint already exists
        if fold in existing_folds:
            summary_path = config.LOG_DIR / 'metrics' / f'{config.MODEL_NAME}_fold{fold}_summary.json'
            best_val_macro_f1 = None
            best_val_acc = 0.0
            summary_loaded = False
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    best_val_macro_f1 = summary.get('best_val_macro_f1')
                    best_val_acc = summary.get('best_val_acc', 0.0) or 0.0
                    summary_loaded = True
                except Exception as e:
                    print(f"Warning: Could not read summary for fold {fold}: {e}")
            if best_val_macro_f1 is not None:
                fold_scores.append(best_val_macro_f1)
            fold_details.append({
                'fold': fold,
                'val_macro_f1': best_val_macro_f1,
                'val_acc': best_val_acc,
                'val_balanced_accuracy': None,
                'val_macro_precision': None,
                'val_macro_specificity': None,
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'skipped': True,
                'summary_loaded': summary_loaded
            })
            if is_main_process():
                print(f"Skipping Fold {fold}: checkpoint already exists")
            continue

        val_macro_f1, val_metrics = train_fold(fold, train_df, val_df, config)
        fold_scores.append(val_macro_f1)
        fold_details.append({
            'fold': fold,
            'val_macro_f1': val_macro_f1,
            'val_acc': val_metrics.get('accuracy', 0.0) if val_metrics else 0.0,
            'val_balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0) if val_metrics else 0.0,
            'val_macro_precision': val_metrics.get('macro_precision', 0.0) if val_metrics else 0.0,
            'val_macro_specificity': val_metrics.get('macro_specificity', 0.0) if val_metrics else 0.0,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'skipped': False
        })
        # Print BL class performance if available
        bl_info = ""
        if val_metrics and val_metrics.get('per_class') and 'BL' in val_metrics['per_class']:
            bl_f1 = val_metrics['per_class']['BL']['f1']
            bl_info = f" | BL F1: {bl_f1:.4f}"
        print(f"Fold {fold} - Macro-F1: {val_macro_f1:.4f} (Primary Metric){bl_info}")
    
    # Calculate final statistics (main process only)
    if not is_main_process():
        return 0.0, None

    if fold_scores:
        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        min_acc = np.min(fold_scores)
        max_acc = np.max(fold_scores)
    else:
        mean_acc = 0.0
        std_acc = 0.0
        min_acc = 0.0
        max_acc = 0.0
        print("Warning: No folds were trained or loaded for this model.")
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results (Primary Metric - Macro-F1):")
    print(f"Mean Macro-F1: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Min Macro-F1: {min_acc:.4f}")
    print(f"Max Macro-F1: {max_acc:.4f}")
    print(f"{'='*60}")
    
    # Calculate tie-breaking metrics (exclude folds with missing metrics)
    balanced_values = [f['val_balanced_accuracy'] for f in fold_details if f.get('val_balanced_accuracy') is not None]
    macro_prec_values = [f['val_macro_precision'] for f in fold_details if f.get('val_macro_precision') is not None]
    macro_spec_values = [f['val_macro_specificity'] for f in fold_details if f.get('val_macro_specificity') is not None]
    mean_balanced_acc = float(np.mean(balanced_values)) if balanced_values else 0.0
    mean_macro_prec = float(np.mean(macro_prec_values)) if macro_prec_values else 0.0
    mean_macro_spec = float(np.mean(macro_spec_values)) if macro_spec_values else 0.0
    
    print(f"\nTie-Breaking Metrics:")
    print(f"Mean Balanced Accuracy: {mean_balanced_acc:.4f}")
    print(f"Mean Macro Precision: {mean_macro_prec:.4f}")
    print(f"Mean Macro Specificity: {mean_macro_spec:.4f}")
    print(f"{'='*60}")
    
    # Save final summary
    final_summary = {
        'model_name': config.MODEL_NAME,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_folds': len(fold_scores),
        'mean_macro_f1': float(mean_acc),  # Primary metric
        'std_macro_f1': float(std_acc),
        'min_macro_f1': float(min_acc),
        'max_macro_f1': float(max_acc),
        'mean_balanced_accuracy': float(mean_balanced_acc),
        'mean_macro_precision': float(mean_macro_prec),
        'mean_macro_specificity': float(mean_macro_spec),
        'fold_details': fold_details,
        'skipped_folds': [f['fold'] for f in fold_details if f.get('skipped')],
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'img_size': config.IMG_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'label_smoothing': config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0,
            'use_mixup': config.USE_MIXUP,
            'use_cutmix': config.USE_CUTMIX,
            'use_warmup': config.USE_WARMUP,
            'warmup_epochs': config.WARMUP_EPOCHS,
            'gradient_clipping': config.USE_GRADIENT_CLIPPING,
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        },
        'dataset_info': {
            'total_samples': len(all_train),
            'num_classes': len(all_train['labels'].unique()),
            'class_distribution': all_train['labels'].value_counts().to_dict()
        }
    }
    
    # Save final summary
    final_summary_path = config.LOG_DIR / 'metrics' / f'{config.MODEL_NAME}_final_summary.json'
    with open(final_summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    print(f"\nSaved final summary to {final_summary_path}")
    
    # Create final comparison plot
    plt.figure(figsize=(12, 6))
    folds = [f['fold'] for f in fold_details]
    macro_f1_scores = [f['val_macro_f1'] for f in fold_details]
    
    plt.bar(folds, macro_f1_scores, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
    plt.axhline(y=mean_acc + std_acc, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'±1 Std')
    plt.axhline(y=mean_acc - std_acc, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Macro-F1 Score (Primary Metric)', fontsize=12)
    plt.title(f'Cross-Validation Results - {config.MODEL_NAME}\nMean Macro-F1: {mean_acc:.4f} ± {std_acc:.4f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(folds)
    
    final_plot_path = config.LOG_DIR / 'plots' / f'{config.MODEL_NAME}_cv_comparison.png'
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CV comparison plot to {final_plot_path}")
    
    return mean_acc, fold_details[0]['val_metrics'] if fold_details else None

def main():
    """Main training function - Integrated Session 1 & 2 workflow"""
    config = Config()

    distributed, rank, world_size, local_rank = init_distributed_mode()
    config.DISTRIBUTED = distributed
    config.RANK = rank
    config.WORLD_SIZE = world_size
    config.LOCAL_RANK = local_rank
    if distributed:
        config.DEVICE = f'cuda:{local_rank}'
    else:
        config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    main_process = is_main_process()
    
    # Check if we should train all ensemble models automatically
    if config.TRAIN_ALL_ENSEMBLE:
        if main_process:
            print("\n" + "="*60)
            print("AUTO-TRAIN MODE ENABLED - SESSION 1")
            print("="*60)
            print("Will train all ensemble models sequentially with optimized settings.")
            print("No need to manually change config between models!")
            print("="*60)
        
        # Train all models (Session 1)
        all_results = train_all_ensemble_models(config)
        
        # Session 2: Pseudo-labeling and retraining
        if config.RUN_PSEUDO_LABELING and main_process:
            print("\n" + "="*60)
            print("SESSION 2: PSEUDO-LABELING")
            print("="*60)
            
            # Generate pseudo-labels
            try:
                from scripts.pseudo_labeling import generate_pseudo_labels, merge_pseudo_labels_with_training
                
                # Find all trained models
                model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
                
                if model_paths:
                    print(f"\nGenerating pseudo-labels using {len(model_paths)} models...")
                    pseudo_df, full_results = generate_pseudo_labels(
                        config,
                        model_paths,
                        confidence_threshold=config.PSEUDO_LABEL_THRESHOLD,
                        min_samples_per_class=10
                    )
                    
                    # Merge with training data
                    pseudo_path = config.PRED_DIR / f'pseudo_labels_thresh{config.PSEUDO_LABEL_THRESHOLD}.csv'
                    merged_train = merge_pseudo_labels_with_training(config, pseudo_path)
                    
                    print(f"\n✓ Pseudo-labeling complete: {len(pseudo_df)} high-confidence samples added")
                    
                    # Retrain best models with pseudo-labels
                    if config.RETRAIN_WITH_PSEUDO and len(all_results) > 0:
                        print("\n" + "="*60)
                        print("SESSION 2: RETRAINING WITH PSEUDO-LABELS")
                        print("="*60)
                        
                        # Load existing model results from summary files (for models not trained in this session)
                        # This includes models like convnextv2_large that were trained earlier
                        existing_models = set(r['model_name'] for r in all_results)
                        all_available_models = set()
                        
                        # Check all model files to find models not in all_results
                        model_files = list(config.MODEL_DIR.glob('*_best.pth'))
                        for model_path in model_files:
                            # Extract model name from filename (e.g., "convnextv2_large_fold0_best.pth" -> "convnextv2_large")
                            model_name = '_'.join(model_path.stem.split('_')[:-2])  # Remove "_fold0_best"
                            all_available_models.add(model_name)
                        
                        # Load results for models not in all_results
                        for model_name in all_available_models:
                            if model_name not in existing_models:
                                # Try to load final summary JSON
                                final_summary_path = config.LOG_DIR / 'metrics' / f'{model_name}_final_summary.json'
                                if final_summary_path.exists():
                                    try:
                                        with open(final_summary_path) as f:
                                            summary_data = json.load(f)
                                        all_results.append({
                                            'model_name': model_name,
                                            'val_macro_f1': summary_data.get('mean_macro_f1', 0.0),
                                            'val_metrics': None  # Not needed for retraining selection
                                        })
                                        print(f"✓ Loaded existing model results: {model_name} (Macro-F1: {summary_data.get('mean_macro_f1', 0.0):.4f})")
                                    except Exception as e:
                                        print(f"⚠ Could not load results for {model_name}: {e}")
                        
                        # Find best 2 models from all available (including existing ones)
                        sorted_results = sorted(all_results, key=lambda x: x['val_macro_f1'], reverse=True)
                        best_models = [r['model_name'] for r in sorted_results[:2]]
                        
                        print(f"Retraining top 2 models: {best_models}")
                        
                        for model_name in best_models:
                            print(f"\nRetraining {model_name} with pseudo-labels...")
                            
                            # Update model name
                            original_model = config.MODEL_NAME
                            config.MODEL_NAME = model_name
                            
                            # Apply model-specific settings
                            if model_name in config.MODEL_SPECIFIC_SETTINGS:
                                settings = config.MODEL_SPECIFIC_SETTINGS[model_name]
                                config.LEARNING_RATE = settings.get('learning_rate', config.LEARNING_RATE)
                                config.BATCH_SIZE = settings.get('batch_size', config.BATCH_SIZE)
                            
                            try:
                                # Train with pseudo-labels (train_single_model will auto-detect)
                                train_single_model(config)
                                print(f"✓ {model_name} retrained successfully")
                            except Exception as e:
                                print(f"✗ Error retraining {model_name}: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Restore original
                            config.MODEL_NAME = original_model
                        
                        print("\n" + "="*60)
                        print("SESSION 2 COMPLETE")
                        print("="*60)
                        print("All models trained with pseudo-labels!")
                else:
                    print("⚠ No models found for pseudo-labeling")
            except Exception as e:
                print(f"⚠ Pseudo-labeling failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without pseudo-labels...")
        
        # Session 3: Final ensemble optimization and submission
        if getattr(config, 'RUN_FINAL_SUBMISSION', False) and main_process:
            print("\n" + "="*60)
            print("SESSION 3: FINAL SUBMISSION")
            print("="*60)
            
            try:
                from scripts.ensemble_optimizer import optimize_ensemble_weights
                from scripts.final_submission import prepare_final_submission
                from scripts.validate_submission import comprehensive_validation
                
                # Step 1: Optimize ensemble weights
                if getattr(config, 'ENSEMBLE_OPTIMIZATION', True):
                    print("\nStep 1: Optimizing ensemble weights...")
                    try:
                        model_paths = list(config.MODEL_DIR.glob('*_best.pth'))
                        if model_paths:
                            best_weights, best_f1, best_method, results = optimize_ensemble_weights(
                                config,
                                model_paths,
                                method=getattr(config, 'ENSEMBLE_OPT_METHOD', 'all')
                            )
                            print(f"✓ Best ensemble method: {best_method} (F1: {best_f1:.5f})")
                        else:
                            print("⚠ No models found for optimization")
                    except Exception as e:
                        print(f"⚠ Ensemble optimization failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Step 2: Generate final submission
                print("\nStep 2: Generating final submission...")
                try:
                    submission, metrics = prepare_final_submission(
                        config,
                        use_optimized_weights=getattr(config, 'ENSEMBLE_OPTIMIZATION', True),
                        validate=getattr(config, 'FINAL_SUBMISSION_VALIDATION', True)
                    )
                    print(f"✓ Final submission generated!")
                    print(f"  Eval Macro F1: {metrics['eval_macro_f1']:.5f}")
                except Exception as e:
                    print(f"⚠ Final submission generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Step 3: Validate submission
                print("\nStep 3: Validating submission...")
                try:
                    submission_path = config.PRED_DIR / getattr(config, 'SUBMISSION_FILENAME', 'final_submission.csv')
                    if submission_path.exists():
                        is_valid, results = comprehensive_validation(
                            submission_path,
                            config.PHASE2_TEST_CSV
                        )
                        if is_valid:
                            print("✓ Submission validation passed!")
                        else:
                            print("⚠ Submission has validation issues (check warnings above)")
                    else:
                        print("⚠ Submission file not found")
                except Exception as e:
                    print(f"⚠ Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("\n" + "="*60)
                print("SESSION 3 COMPLETE")
                print("="*60)
                print("Final submission ready for Kaggle!")
                
            except Exception as e:
                print(f"⚠ Session 3 failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing...")
    else:
        # Original behavior: train single model
        train_single_model(config)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

