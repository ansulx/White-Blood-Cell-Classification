"""
Custom dataset class for WBC-Bench-2026
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Set, Optional


def identify_rare_classes(df: pd.DataFrame, threshold: float = 0.1) -> Set[str]:
    """
    Identify rare classes based on sample count threshold.
    
    Args:
        df: DataFrame with 'labels' column
        threshold: Classes with <threshold * median_samples are considered rare
    
    Returns:
        Set of rare class names
    """
    class_counts = df['labels'].value_counts()
    median_samples = class_counts.median()
    threshold_samples = median_samples * threshold
    
    rare_classes = set(class_counts[class_counts < threshold_samples].index)
    return rare_classes

class WBCDataset(Dataset):
    """Dataset class for WBC image classification"""
    
    def __init__(self, csv_path, img_dir, transform=None, is_train=True, 
                 rare_classes=None, class_aware_aug=False):
        """
        Args:
            csv_path: Path to CSV file with image IDs and labels
            img_dir: Directory containing images
            transform: Transform to apply to images (can be dict with 'normal' and 'rare' keys)
            is_train: Whether this is training data (has labels)
            rare_classes: Set of rare class names (for class-aware augmentation)
            class_aware_aug: Whether to use class-aware augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        self.transform = transform
        self.rare_classes = rare_classes if rare_classes is not None else set()
        self.class_aware_aug = class_aware_aug
        
        # Get unique classes and create label mapping
        if is_train:
            self.labels = self.df['labels'].values
            self.unique_classes = sorted(self.df['labels'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.num_classes = len(self.unique_classes)
        else:
            self.labels = None
            self.unique_classes = None
            self.num_classes = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ID']
        img_path = self.img_dir / img_name
        
        # Load image
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
                elif isinstance(self.transform, A.Compose):
                    # Single transform - use as is (backward compatibility)
                    transform_to_use = self.transform
                else:
                    transform_to_use = self.transform
            else:
                # No class-aware augmentation or not training
                transform_to_use = self.transform
            
            if isinstance(transform_to_use, A.Compose):
                # Albumentations transform
                transformed = transform_to_use(image=image)
                image = transformed['image']
            else:
                # Torchvision transform
                image = transform_to_use(image)
        
        if self.is_train:
            label = self.df.iloc[idx]['labels']
            label_idx = self.class_to_idx[label]
            return image, label_idx, img_name
        else:
            return image, img_name

def get_train_transforms(img_size=384, is_rare_class=False):
    """
    Get training data augmentation transforms.
    
    Args:
        img_size: Target image size
        is_rare_class: If True, applies stronger augmentation for rare classes
    
    Returns:
        Albumentations Compose object
    """
    # Base augmentation (same for all classes)
    base_transforms = [
        # Smart resizing - preserve aspect ratio better
        A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    
    # Standard augmentation parameters
    shift_scale_rotate = A.ShiftScaleRotate(
        shift_limit=0.15,
        scale_limit=0.15,
        rotate_limit=30,
        border_mode=0,
        p=0.7
    )
    
    # Medical image specific transforms
    medical_transforms = [
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.3
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.05,
            p=0.3
        ),
    ]
    
    # Color and brightness transforms
    color_transforms = [
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
    ]
    
    # Noise and blur (light)
    noise_blur = [
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]
    
    # Dropout and cutout
    dropout_transforms = [
        A.CoarseDropout(
            max_holes=12,
            max_height=48,
            max_width=48,
            min_holes=4,
            p=0.4
        ),
        A.Cutout(
            num_holes=8,
            max_h_size=32,
            max_w_size=32,
            p=0.3
        ),
    ]
    
    # Stronger augmentation for rare classes
    if is_rare_class:
        # More aggressive geometric transforms
        shift_scale_rotate = A.ShiftScaleRotate(
            shift_limit=0.2,  # Increased from 0.15
            scale_limit=0.2,  # Increased from 0.15
            rotate_limit=45,  # Increased from 30
            border_mode=0,
            p=0.9  # Increased probability
        )
        
        # More aggressive medical transforms
        medical_transforms = [
            A.ElasticTransform(
                alpha=150,  # Increased
                sigma=150 * 0.05,
                alpha_affine=150 * 0.03,
                p=0.5  # Increased probability
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.4,  # Increased
                p=0.5  # Increased probability
            ),
            A.OpticalDistortion(
                distort_limit=0.3,  # Increased
                shift_limit=0.1,  # Increased
                p=0.5  # Increased probability
            ),
        ]
        
        # More aggressive color transforms
        color_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=0.4,  # Increased
                contrast_limit=0.4,  # Increased
                p=0.8  # Increased probability
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,  # Increased
                sat_shift_limit=40,  # Increased
                val_shift_limit=30,  # Increased
                p=0.7  # Increased probability
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),  # Increased
        ]
        
        # More aggressive dropout
        dropout_transforms = [
            A.CoarseDropout(
                max_holes=16,  # Increased
                max_height=64,  # Increased
                max_width=64,  # Increased
                min_holes=6,  # Increased
                p=0.6  # Increased probability
            ),
            A.Cutout(
                num_holes=12,  # Increased
                max_h_size=48,  # Increased
                max_w_size=48,  # Increased
                p=0.5  # Increased probability
            ),
        ]
    
    # Combine all transforms
    all_transforms = (
        base_transforms +
        [shift_scale_rotate] +
        medical_transforms +
        color_transforms +
        noise_blur +
        dropout_transforms +
        [
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    )
    
    return A.Compose(all_transforms)

def get_val_transforms(img_size=384):
    """Get validation/test transforms (no augmentation, smart resizing)"""
    return A.Compose([
        # Smart resizing - preserve aspect ratio
        A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_tta_transforms(img_size=384):
    """Get Test Time Augmentation transforms (consistent with validation transforms)"""
    base_transform = [
        A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
    ]
    normalize = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return [
        A.Compose(base_transform + normalize),  # Original
        A.Compose(base_transform + [A.HorizontalFlip(p=1.0)] + normalize),  # Horizontal flip
        A.Compose(base_transform + [A.VerticalFlip(p=1.0)] + normalize),  # Vertical flip
        A.Compose(base_transform + [A.RandomRotate90(p=1.0)] + normalize),  # Rotate 90
        A.Compose(base_transform + [A.Transpose(p=1.0)] + normalize),  # Transpose
    ]

