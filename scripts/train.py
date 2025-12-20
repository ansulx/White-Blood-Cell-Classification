"""
Training script for WBC-Bench-2026 competition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import albumentations as A

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_train_transforms, get_val_transforms
from src.models import get_model, get_loss_fn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    return dict(zip(unique_labels, class_weights))

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
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

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_fold(fold, train_df, val_df, config):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # Determine image directory based on split
    # Check if images are in phase1 or phase2
    def get_img_path(row):
        img_name = row['ID']
        phase1_path = config.PHASE1_DIR / img_name
        phase2_path = config.PHASE2_TRAIN_DIR / img_name
        if phase1_path.exists():
            return config.PHASE1_DIR
        elif phase2_path.exists():
            return config.PHASE2_TRAIN_DIR
        else:
            return config.PHASE2_TRAIN_DIR  # Default
    
    # Create custom dataset class that handles both directories
    class CombinedDataset(WBCDataset):
        def __init__(self, df, transform, is_train=True):
            self.df = df.copy()
            self.is_train = is_train
            self.transform = transform
            
            # Determine image directories
            self.img_paths = []
            for _, row in df.iterrows():
                img_name = row['ID']
                phase1_path = config.PHASE1_DIR / img_name
                phase2_path = config.PHASE2_TRAIN_DIR / img_name
                if phase1_path.exists():
                    self.img_paths.append(config.PHASE1_DIR)
                else:
                    self.img_paths.append(config.PHASE2_TRAIN_DIR)
            
            if is_train:
                self.labels = df['labels'].values
                self.unique_classes = sorted(df['labels'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
                self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                self.num_classes = len(self.unique_classes)
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_name = self.df.iloc[idx]['ID']
            img_dir = self.img_paths[idx]
            img_path = img_dir / img_name
            
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                if isinstance(self.transform, A.Compose):
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    image = self.transform(image)
            
            if self.is_train:
                label = self.df.iloc[idx]['labels']
                label_idx = self.class_to_idx[label]
                return image, label_idx, img_name
            else:
                return image, img_name
    
    # Create datasets
    train_dataset = CombinedDataset(train_df, get_train_transforms(config.IMG_SIZE), is_train=True)
    val_dataset = CombinedDataset(val_df, get_val_transforms(config.IMG_SIZE), is_train=True)
    
    # Compute class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_df['labels'].values)
        weights = [class_weights[label] for label in train_df['labels'].values]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=train_dataset.num_classes,
        pretrained=config.PRETRAINED
    ).to(config.DEVICE)
    
    # Loss and optimizer
    if config.USE_CLASS_WEIGHTS:
        class_weights_list = [class_weights[cls] for cls in train_dataset.unique_classes]
        criterion = get_loss_fn('ce', class_weights=class_weights_list, device=config.DEVICE)
    else:
        criterion = get_loss_fn('focal', device=config.DEVICE)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scheduler
        )
        
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc + config.EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = config.MODEL_DIR / f'{config.MODEL_NAME}_fold{fold}_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
            }, model_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc

def main():
    """Main training function"""
    config = Config()
    
    # Load data
    print("Loading data...")
    phase2_train = pd.read_csv(config.PHASE2_TRAIN_CSV)
    
    # Combine phase1 and phase2 for training
    phase1 = pd.read_csv(config.PHASE1_CSV)
    phase1 = phase1[phase1['split'] == 'phase1_train']
    phase1 = phase1[['ID', 'labels']].copy()
    phase1['split'] = 'phase1_train'
    
    # Combine datasets
    all_train = pd.concat([
        phase2_train[['ID', 'labels']],
        phase1[['ID', 'labels']]
    ], ignore_index=True)
    
    print(f"Total training samples: {len(all_train)}")
    print(f"Class distribution:\n{all_train['labels'].value_counts()}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_train, all_train['labels'])):
        train_df = all_train.iloc[train_idx].reset_index(drop=True)
        val_df = all_train.iloc[val_idx].reset_index(drop=True)
        
        val_acc = train_fold(fold, train_df, val_df, config)
        fold_scores.append(val_acc)
        print(f"Fold {fold} Val Acc: {val_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results:")
    print(f"Mean Val Acc: {np.mean(fold_scores):.2f}% ± {np.std(fold_scores):.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

