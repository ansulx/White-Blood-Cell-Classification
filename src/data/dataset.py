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

class WBCDataset(Dataset):
    """Dataset class for WBC image classification"""
    
    def __init__(self, csv_path, img_dir, transform=None, is_train=True):
        """
        Args:
            csv_path: Path to CSV file with image IDs and labels
            img_dir: Directory containing images
            transform: Transform to apply to images
            is_train: Whether this is training data (has labels)
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        self.transform = transform
        
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
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Torchvision transform
                image = self.transform(image)
        
        if self.is_train:
            label = self.df.iloc[idx]['labels']
            label_idx = self.class_to_idx[label]
            return image, label_idx, img_name
        else:
            return image, img_name

def get_train_transforms(img_size=384):
    """Get training data augmentation transforms"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(img_size=384):
    """Get validation/test transforms (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_tta_transforms(img_size=384):
    """Get Test Time Augmentation transforms"""
    return [
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.Transpose(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

