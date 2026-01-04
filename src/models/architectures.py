"""
Model architectures for WBC-Bench-2026

Research-grade modular architecture module supporting multiple state-of-the-art models.
Primary architecture: ConvNeXt-Base (recommended for medical imaging tasks).
"""

import timm
from typing import Optional


def get_model(
    model_name: str = 'convnext_base',
    num_classes: int = 13,
    pretrained: bool = True,
    drop_rate: Optional[float] = None,
    drop_path_rate: Optional[float] = None
):
    """
    Get model architecture from timm library.
    
    Supported architectures:
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
      * Modern CNN architecture, outperforms EfficientNet
      * Better for medical imaging tasks
      * Uses drop_path_rate only (no drop_rate)
    
    - ConvNeXt V2 (RECOMMENDED): convnextv2_base, convnextv2_large
      * Improved version with better training dynamics (FCMAE pretraining)
      * convnextv2_large recommended for best accuracy
      * Uses drop_path_rate only (no drop_rate)
    
    - EfficientNet: efficientnet_b0 through efficientnet_b7
      * Legacy architecture, still effective
      * Uses both drop_rate and drop_path_rate
    
    - Swin Transformer: swin_tiny_patch4_window7_224, swin_base_patch4_window7_224, etc.
      * Vision transformer variant
      * Good for global feature learning
      * Uses drop_path_rate only
    
    - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
      * Classic architecture
      * Baseline comparison
    
    Args:
        model_name: Name of the model architecture (default: 'convnext_base')
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        drop_rate: Dropout rate (EfficientNet only, ignored for ConvNeXt/Swin)
        drop_path_rate: Drop path rate for regularization (model-specific default if None)
    
    Returns:
        Model instance ready for training
    
    Raises:
        ValueError: If model_name is not supported
    
    Example:
        >>> model = get_model('convnext_base', num_classes=13, pretrained=True)
        >>> model = get_model('efficientnet_b5', num_classes=13, drop_rate=0.4)
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
        # ConvNeXt models - optimized defaults for medical imaging
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.3
        # ConvNeXt doesn't use drop_rate, only drop_path_rate
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('resnet'):
        # ResNet models
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('swin'):
        # Swin Transformer models
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('swinv2') or model_name.startswith('swin_v2'):
        # Swin Transformer V2 models
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('maxvit'):
        # MaxViT models - Multi-Axis Vision Transformer
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.2
        model = timm.create_model(model_name, **model_kwargs)
    
    elif model_name.startswith('convnextv2') or model_name.startswith('convnext_v2'):
        # ConvNeXt V2 models
        model_kwargs['drop_path_rate'] = drop_path_rate if drop_path_rate is not None else 0.3
        model = timm.create_model(model_name, **model_kwargs)
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: EfficientNet, ConvNeXt, ConvNeXt V2, ResNet, Swin Transformer, Swin V2, MaxViT"
        )
    
    return model

