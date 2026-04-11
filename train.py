"""Training script."""

import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device
from evaluate import evaluate


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    preds_list = []
    targets_list = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return {
        'loss': total_loss / len(dataloader)
    }


def train(config):
    """Main training function."""
    wandb.init(project="entity-embeddings", name=config.name, config=vars(config))
    device = get_device(config.device)
    
    # Track ensemble predictions
    ensemble_preds = []
    
    # We will get global dataloaders from the first seed, though the original logic varied seed inside the loop.
    # To keep exact behavior, we should create loaders per ensemble iter if shuffle_loader depends on seed,
    # but the original made loaders outside the ensemble loop and just re-seeded model initialization.
    set_seed(config.seed)
    train_loader = get_dataloader(config, 'train')
    test_loader = get_dataloader(config, 'test')
    
    target_max = config.target_max
    
    for i in range(config.ensemble_size):
        # We vary seed per ensemble model to get different initializations
        current_seed = config.seed + i
        set_seed(current_seed)
        
        model = create_model(config).to(device)
        criterion = get_loss_fn(config)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        print(f"\\n--- Training Ensemble Member {i+1}/{config.ensemble_size} ---")
        for epoch in range(config.num_epochs):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, test_loader, criterion, device, target_max)
            
            # Log to wandb
            wandb.log({
                f'member_{i}/epoch': epoch,
                f'member_{i}/train_loss': train_metrics['loss'],
                f'member_{i}/val_loss': val_metrics['loss'],
                f'member_{i}/val_mape': val_metrics['mape']
            })
            
            print(f"Epoch {epoch+1}/{config.num_epochs}: Train Loss={train_metrics['loss']:.8f}, Val MAPE={val_metrics['mape']:.8f}")
            
        folder_path = os.path.join('models', config.name)
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"{config.name}_{i+1}.pth")
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        print(f"Saved model to {save_path}")
        
        # Collect test predictions from this member
        member_preds, _ = evaluate(model, test_loader, criterion, device, target_max, return_preds=True)
        ensemble_preds.append(member_preds)

    # Compute final ensemble metrics
    mean_preds = np.mean(ensemble_preds, axis=0)
    
    # Retrieve true test targets to compute final metric
    _, true_targets = evaluate(model, test_loader, criterion, device, target_max, return_preds=True)
    
    ensemble_metrics = compute_metrics(mean_preds, true_targets, target_max)
    wandb.log({
        'ensemble_test_mape': ensemble_metrics['mape']
    })
    
    print(f"\\nEnsemble Final Result {config.name}: Test MAPE = {ensemble_metrics['mape']:.8f}")
    
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', f'{config.name}_result.json'), 'w') as f:
        json.dump({'ensemble_mape': ensemble_metrics['mape']}, f, indent=4)
        
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                       choices=['baseline', 'ablation_time_based', 'ablation_embedding', 'ablation_bn', 'ablation_dec_lr', 'ablation_inc_lr'],
                       help='Configuration to use')
    args = parser.parse_args()
    
    config = get_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
