"""
Evaluation metrics for WBC-Bench-2026 competition
Implements primary metric (Macro-F1) and tie-breaking metrics
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                     labels: List[str] = None) -> float:
    """
    Compute macro-averaged F1 score (Primary Metric)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (optional)
    
    Returns:
        Macro-averaged F1 score
    """
    return f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy (Tie-breaker #1)
    
    Balanced accuracy = (Sensitivity + Specificity) / 2
    For multi-class: average of recall for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Balanced accuracy
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0)


def compute_macro_precision(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: List[str] = None) -> float:
    """
    Compute macro-averaged precision (Tie-breaker #2)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (optional)
    
    Returns:
        Macro-averaged precision
    """
    return precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)


def compute_macro_specificity(y_true: np.ndarray, y_pred: np.ndarray,
                             labels: List[str] = None) -> float:
    """
    Compute macro-averaged specificity (Tie-breaker #3)
    
    Specificity = TN / (TN + FP)
    For multi-class: computed per class and averaged
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (optional)
    
    Returns:
        Macro-averaged specificity
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        n_classes = len(np.unique(y_true))
    else:
        n_classes = len(labels)
    
    specificities = []
    for i in range(n_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    return np.mean(specificities)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: List[str] = None) -> Dict[str, float]:
    """
    Compute all competition metrics
    
    Args:
        y_true: True labels (can be indices or class names)
        y_pred: Predicted labels (can be indices or class names)
        class_names: List of class names (if using indices, provide mapping)
    
    Returns:
        Dictionary with all metrics
    """
    # Convert to indices if class names provided
    if class_names is not None:
        if isinstance(y_true[0], str):
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            y_true_idx = np.array([class_to_idx[label] for label in y_true])
            y_pred_idx = np.array([class_to_idx[label] for label in y_pred])
        else:
            y_true_idx = y_true
            y_pred_idx = y_pred
    else:
        y_true_idx = y_true
        y_pred_idx = y_pred
    
    metrics = {
        'macro_f1': compute_macro_f1(y_true_idx, y_pred_idx),
        'balanced_accuracy': compute_balanced_accuracy(y_true_idx, y_pred_idx),
        'macro_precision': compute_macro_precision(y_true_idx, y_pred_idx),
        'macro_specificity': compute_macro_specificity(y_true_idx, y_pred_idx),
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
    }
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    per_class_precision = precision_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    per_class_recall = recall_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    
    if class_names:
        metrics['per_class'] = {
            name: {
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec)
            }
            for name, f1, prec, rec in zip(class_names, per_class_f1, per_class_precision, per_class_recall)
        }
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str] = None) -> None:
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is not None and isinstance(y_true[0], str):
        # Convert to indices
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        y_true_idx = np.array([class_to_idx[label] for label in y_true])
        y_pred_idx = np.array([class_to_idx[label] for label in y_pred])
        target_names = class_names
    else:
        y_true_idx = y_true
        y_pred_idx = y_pred
        target_names = class_names if class_names else None
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true_idx, y_pred_idx, target_names=target_names, zero_division=0))
    
    # Print competition metrics
    all_metrics = compute_all_metrics(y_true_idx, y_pred_idx, class_names)
    print("\n" + "="*60)
    print("Competition Metrics")
    print("="*60)
    print(f"Primary Metric - Macro-F1: {all_metrics['macro_f1']:.4f}")
    print(f"Tie-Breaker #1 - Balanced Accuracy: {all_metrics['balanced_accuracy']:.4f}")
    print(f"Tie-Breaker #2 - Macro Precision: {all_metrics['macro_precision']:.4f}")
    print(f"Tie-Breaker #3 - Macro Specificity: {all_metrics['macro_specificity']:.4f}")
    print(f"Overall Accuracy: {all_metrics['accuracy']:.4f}")
    print("="*60)

