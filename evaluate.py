"""Evaluation script."""

import argparse
import numpy as np
import torch

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device


def evaluate(model, dataloader, criterion, device, target_max=None, return_preds=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            preds_list.append(output.cpu())
            targets_list.append(target.cpu())
    
    preds_all = torch.cat(preds_list).flatten().numpy()
    targets_all = torch.cat(targets_list).flatten().numpy()
    
    metrics = compute_metrics(preds_all, targets_all, target_max)
    result = {
        'loss': total_loss / len(dataloader),
        'mape': metrics['mape']
    }
    
    if return_preds:
        return preds_all, targets_all
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)
    
    # During standalone evaluation, target_max will be calculated inside get_dataloader by executing prepare_data.
    test_loader = get_dataloader(config, 'test')
    target_max = config.target_max
    
    model = create_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = get_loss_fn(config)
    results = evaluate(model, test_loader, criterion, device, target_max)
    
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test MAPE: {results['mape']:.4f}")


if __name__ == "__main__":
    main()
