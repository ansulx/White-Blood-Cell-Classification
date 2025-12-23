"""
Loss functions for WBC-Bench-2026
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for rare class (default: 1)
        gamma: Focusing parameter (default: 2)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    
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


def get_loss_fn(loss_type='ce', class_weights=None, device='cuda', label_smoothing=0.0):
    """
    Get loss function based on type.
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'label_smoothing')
        class_weights: Class weights for weighted loss (list or tensor)
        device: Device to put weights on
        label_smoothing: Label smoothing factor (0.0 to 1.0)
    
    Returns:
        Loss function instance
    """
    if loss_type == 'ce':
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    elif loss_type == 'focal':
        # Focal loss with label smoothing support
        if label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return FocalLoss(alpha=1, gamma=2)
    
    elif loss_type == 'label_smoothing':
        smoothing = label_smoothing if label_smoothing > 0 else 0.1
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: 'ce', 'focal', 'label_smoothing'")

