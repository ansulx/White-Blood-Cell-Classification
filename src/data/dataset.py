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
                 rare_classes=None, class_aware_aug=False, domain_adaptation=False):
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
        self.domain_adaptation = domain_adaptation
        
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

def get_train_transforms(img_size=384, is_rare_class=False, domain_adaptation=False):
    """
    Get training data augmentation transforms with domain adaptation support.
    
    Args:
        img_size: Target image size
        is_rare_class: If True, applies stronger augmentation for rare classes
        domain_adaptation: If True, adds noise/blur/color jitter to match test distribution
    """
    # Base augmentation
    base_transforms = [
        A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    
    shift_scale_rotate = A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=20,
        border_mode=0,
        p=0.5
    )
    
    medical_transforms = [
        A.ElasticTransform(
            alpha=100,
            sigma=100 * 0.05,
            alpha_affine=100 * 0.03,
            p=0.2
        ),
    ]
    
    color_transforms = [
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.4
        ),
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.4),
    ]
    
    # Domain adaptation: Match test-time degradation
    if domain_adaptation:
        noise_blur = [
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),  # Increased for domain adaptation
            A.GaussianBlur(blur_limit=(5, 9), p=0.4),  # Increased for domain adaptation
            A.MotionBlur(blur_limit=7, p=0.3),  # Additional blur type
        ]
        color_transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Increased
                contrast_limit=0.3,  # Increased
                p=0.6
            ),
            # Additional color variation for domain adaptation (equivalent to ColorJitter)
            A.HueSaturationValue(
                hue_shift_limit=10,  # Equivalent to hue=0.1
                sat_shift_limit=30,  # Equivalent to saturation=0.3
                val_shift_limit=30,  # brightness handled by RandomBrightnessContrast above
                p=0.5
            ),
        ])
    else:
        noise_blur = [
            A.GaussNoise(var_limit=(3.0, 15.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ]
    
    dropout_transforms = [
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=2,
            p=0.3
        ),
    ]
    
    # Rare class handling
    if is_rare_class:
        shift_scale_rotate = A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.7
        )
        medical_transforms.append(
            A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.3)
        )
        color_transforms[0] = A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6
        )
        dropout_transforms[0] = A.CoarseDropout(
            max_holes=10,
            max_height=40,
            max_width=40,
            min_holes=3,
            p=0.4
        )
    
    all_transforms = (
        base_transforms +
        [shift_scale_rotate] +
        medical_transforms +
        color_transforms +
        noise_blur +
        dropout_transforms +
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    """Get Test Time Augmentation transforms - Enhanced to 15 transforms"""
    base_transform = [
        A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=0, value=0, mask_value=0, p=1.0),
    ]
    normalize = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    # Enhanced TTA: 15 transforms
    return [
        A.Compose(base_transform + normalize),  # 1. Original
        A.Compose(base_transform + [A.HorizontalFlip(p=1.0)] + normalize),  # 2. H-flip
        A.Compose(base_transform + [A.VerticalFlip(p=1.0)] + normalize),  # 3. V-flip
        A.Compose(base_transform + [A.RandomRotate90(p=1.0)] + normalize),  # 4. Rotate 90
        A.Compose(base_transform + [A.Transpose(p=1.0)] + normalize),  # 5. Transpose
        A.Compose(base_transform + [A.Rotate(limit=5, p=1.0)] + normalize),  # 6. Rotate +5
        A.Compose(base_transform + [A.Rotate(limit=-5, p=1.0)] + normalize),  # 7. Rotate -5
        A.Compose(base_transform + [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + normalize),  # 8. Bright+
        A.Compose(base_transform + [A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=-0.1, p=1.0)] + normalize),  # 9. Bright-
        A.Compose(base_transform + [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=1.0)] + normalize),  # 10. Shift
        A.Compose(base_transform + [A.GaussNoise(var_limit=(5.0, 10.0), p=1.0)] + normalize),  # 11. Noise
        A.Compose(base_transform + [A.GaussianBlur(blur_limit=3, p=1.0)] + normalize),  # 12. Blur
        A.Compose(base_transform + [A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1.0)] + normalize),  # 13. Color+
        A.Compose(base_transform + [A.HueSaturationValue(hue_shift_limit=-5, sat_shift_limit=-5, val_shift_limit=-5, p=1.0)] + normalize),  # 14. Color-
        A.Compose(base_transform + [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)] + normalize),  # 15. CLAHE
    ]

