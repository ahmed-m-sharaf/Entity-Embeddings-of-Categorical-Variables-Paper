# Import Libraries
import numpy as np
import torch
from torch import nn, optim

from src import *

# Configuration 
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_single_model(
    criterion_type,
    train_loader, 
    test_loader, 
    encoder, 
    emp_dims,
    target_max, 
    hidden_sizes,
    batch_norm,
    epochs,
    learning_rate,
    seed: int):
    
    """Train a single model and return train/test MAPE and training history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RossmannEmbeddingModel(
        encoder=encoder,
        emb_dims=emp_dims,
        hidden_sizes=hidden_sizes,
        output_sigmoid=OUTPUT_SIGMOID,
        batch_norm=batch_norm,
    ).to(device)

    criterion = nn.MSELoss() if criterion_type.lower() == 'MSE' else nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=epochs,
        device=device,
    )

    trainer.train()

    # Evaluate train MAPE
    train_preds, train_targets = get_preds_targets(model, train_loader, device)
    train_mape = compute_metrics(train_preds, train_targets, target_max)

    # Evaluate test MAPE
    test_preds, test_targets = get_preds_targets(model, test_loader, device)
    test_mape = compute_metrics(test_preds, test_targets, target_max)

    return train_mape, test_mape, test_preds, trainer.history, model




def Experiment(
    split_type,
    shuffle_loader,
    criterion_type,
    hidden_sizes,
    emp_dims,
    batch_norm,
    epochs,
    learning_rate,
    seed,
    model_name
):
    # Prepare data
    train_df, test_df, encoder, target_max = prepare_data(split_type)
    ds_train, ds_test, train_loader, test_loader = make_loaders(train_df, test_df, shuffle=shuffle_loader)

    ensemble_preds = []
    all_results = []

    for i in range(ENSEMBLE_SIZE):
        seed = 42 + i
        train_mape, test_mape, test_preds, history, model = train_single_model(
            criterion_type,
            train_loader, 
            test_loader, 
            encoder, 
            emp_dims,
            target_max, 
            hidden_sizes,
            batch_norm,
            epochs,
            learning_rate,
            seed
        )
        folder_path = os.path.join('models', model_name)
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"{model_name}_{i+1}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, save_path)
        print(f"Model state saved to {save_path}")
        
        label = None
        embedding_space_vis(model, model_name, f'{model_name}_{i+1}', label)

        ensemble_preds.append(test_preds)
        all_results.append(
            {
                "model": i + 1,
                "train_mape": train_mape,
                "test_mape": test_mape,
                "history": history,
            }
        )

        print(
            f"Model {i + 1} finished | Train MAPE={train_mape:.8f}, Test MAPE={test_mape:.8f}"
        )

    mean_preds = np.mean(ensemble_preds, axis=0)
    y_test_true = ds_test.y.numpy().flatten()
    ensemble_mape = compute_metrics(mean_preds, y_test_true, target_max)

    print("\nEnsemble Result")
    print(f"Ensemble Test MAPE: {ensemble_mape:.8f}")

    return all_results, ensemble_mape
