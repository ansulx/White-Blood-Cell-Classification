"""
Model architectures for WBC-Bench-2026
"""

import timm
from typing import Optional


def get_model(
    model_name: str = 'efficientnet_b4',
    num_classes: int = 13,
    pretrained: bool = True,
    drop_rate: Optional[float] = None,
    drop_path_rate: Optional[float] = None
):
    """
    Get model architecture from timm library.
    
    Supported architectures:
    - EfficientNet: efficientnet_b0 through efficientnet_b7
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
    - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    - Swin Transformer: swin_tiny_patch4_window7_224, swin_base_patch4_window7_224, etc.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        drop_rate: Dropout rate (model-specific default if None)
        drop_path_rate: Drop path rate for regularization (model-specific default if None)
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_name is not supported
    """
    model_kwargs = {
        'pretrained': pretrained,
        'num_classes': num_classes,
    }
    
    if model_name.startswith('efficientnet'):
        # EfficientNet models
        model_kwargs['drop_rate'] = drop_rate if drop_rate is not None else 0.3
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('convnext'):
        # ConvNeXt models
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('resnet'):
        # ResNet models
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('swin'):
        # Swin Transformer models
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: EfficientNet, ConvNeXt, ResNet, Swin Transformer"
        )
    
    return model

