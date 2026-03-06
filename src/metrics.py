import numpy as np
import torch

from .preprocessor import invert_target


def get_preds_targets(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    
    """Get predictions and true targets"""
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            preds = model(data)
            preds_list.append(preds.detach().cpu())
            targets_list.append(target.detach().cpu())

    preds = torch.cat(preds_list).flatten().numpy()
    targets = torch.cat(targets_list).flatten().numpy()

    return preds, targets


def evaluate_mape(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error (MAPE)"""
    mape = np.mean(np.abs((targets - preds) / (np.abs(targets) + eps)))
    return float(mape)


def compute_metrics(
    preds: np.ndarray, targets: np.ndarray, target_max: np.ndarray, eps: float = 1e-8
):
    """Compute all metrics."""
    preds = invert_target(preds, target_max)
    targets = invert_target(targets, target_max)
    return evaluate_mape(preds, targets, eps)


