"""
Model definitions for WBC-Bench-2026
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm

def get_model(model_name='efficientnet_b4', num_classes=13, pretrained=True):
    """
    Get model architecture
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    if model_name.startswith('efficientnet'):
        # EfficientNet models
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
    elif model_name.startswith('convnext'):
        # ConvNeXt models
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=0.2
        )
    elif model_name.startswith('resnet'):
        # ResNet models
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_name.startswith('swin'):
        # Swin Transformer models
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=0.2
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_fn(loss_type='ce', class_weights=None, device='cuda'):
    """
    Get loss function
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'label_smoothing')
        class_weights: Class weights for weighted loss
        device: Device to put weights on
    
    Returns:
        Loss function
    """
    if loss_type == 'ce':
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=1, gamma=2)
    elif loss_type == 'label_smoothing':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

