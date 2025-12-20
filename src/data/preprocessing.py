"""
Data preprocessing utilities for WBC-Bench-2026
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


def validate_image(image_path: str) -> bool:
    """
    Validate if image can be loaded and is valid.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def get_image_stats(image_path: str) -> Optional[dict]:
    """
    Get statistics about an image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with image statistics or None if invalid
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        return {
            'size': img.size,
            'mode': img.mode,
            'shape': img_array.shape,
            'dtype': str(img_array.dtype),
            'min': float(img_array.min()),
            'max': float(img_array.max()),
            'mean': float(img_array.mean()),
            'std': float(img_array.std()),
        }
    except Exception:
        return None


def normalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """
    Normalize image with mean and std.
    
    Args:
        image: Image array (H, W, C)
        mean: Mean values for each channel
        std: Std values for each channel
    
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    image = (image - mean) / std
    return image

