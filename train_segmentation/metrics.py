"""
Metrics module for cell segmentation evaluation.
Includes: DICE, HD95, Jaccard Index (IoU), Sensitivity, PPV
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple, Optional


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Binary prediction mask (H, W)
        target: Binary ground truth mask (H, W)
    
    Returns:
        Dice coefficient [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union


def compute_jaccard(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Jaccard Index (IoU).
    
    Args:
        pred: Binary prediction mask (H, W)
        target: Binary ground truth mask (H, W)
    
    Returns:
        Jaccard Index [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Sensitivity (Recall, True Positive Rate).
    Sen = TP / (TP + FN)
    
    Args:
        pred: Binary prediction mask (H, W)
        target: Binary ground truth mask (H, W)
    
    Returns:
        Sensitivity [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    
    if tp + fn == 0:
        return 1.0
    
    return tp / (tp + fn)


def compute_ppv(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Positive Predictive Value (Precision).
    PPV = TP / (TP + FP)
    
    Args:
        pred: Binary prediction mask (H, W)
        target: Binary ground truth mask (H, W)
    
    Returns:
        PPV [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    
    if tp + fp == 0:
        return 1.0
    
    return tp / (tp + fp)


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute 95% Hausdorff Distance.
    
    Args:
        pred: Binary prediction mask (H, W)
        target: Binary ground truth mask (H, W)
    
    Returns:
        HD95 distance in pixels
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Handle edge cases
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)  # Max possible distance
    
    # Get surface points (boundary)
    pred_boundary = _get_boundary(pred)
    target_boundary = _get_boundary(target)
    
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_boundary)
    target_dist = distance_transform_edt(~target_boundary)
    
    # Get distances from pred boundary to target and vice versa
    pred_to_target = pred_dist[target_boundary]
    target_to_pred = target_dist[pred_boundary]
    
    # Combine distances and compute 95th percentile
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    
    return np.percentile(all_distances, 95)


def _get_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels from a binary mask."""
    from scipy.ndimage import binary_erosion
    
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return boundary


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute all segmentation metrics.
    
    Args:
        pred: Binary prediction mask (H, W) or (N, H, W)
        target: Binary ground truth mask (H, W) or (N, H, W)
    
    Returns:
        Dictionary with all metrics
    """
    # Handle batch dimension
    if pred.ndim == 3:
        metrics_list = [compute_all_metrics(p, t) for p, t in zip(pred, target)]
        return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    
    return {
        'dice': compute_dice(pred, target),
        'hd95': compute_hd95(pred, target),
        'jaccard': compute_jaccard(pred, target),
        'sensitivity': compute_sensitivity(pred, target),
        'ppv': compute_ppv(pred, target),
    }


class MetricTracker:
    """Track and average metrics over multiple batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics_sum = {}
        self.count = 0
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        """Update with new metrics."""
        for k, v in metrics.items():
            if k not in self.metrics_sum:
                self.metrics_sum[k] = 0.0
            self.metrics_sum[k] += v * n
        self.count += n
    
    def get_average(self) -> Dict[str, float]:
        """Get averaged metrics."""
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.metrics_sum.items()}
    
    def __str__(self) -> str:
        avg = self.get_average()
        return ' | '.join([f'{k}: {v:.4f}' for k, v in avg.items()])
