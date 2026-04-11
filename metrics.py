"""Metrics computation."""

import numpy as np
import torch
from data import invert_target


def evaluate_mape(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error (MAPE)"""
    mape = np.mean(np.abs((targets - preds) / (np.abs(targets) + eps)))
    return float(mape)


def compute_metrics(output, target, target_max=None):
    """
    Compute all metrics. Expected detached output and target.
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
        
    if target_max is not None:
        preds = invert_target(output, target_max)
        targets = invert_target(target, target_max)
    else:
        preds = output
        targets = target
        
    return {
        'mape': evaluate_mape(preds, targets)
    }
