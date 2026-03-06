import json
import os
import matplotlib.pyplot as plt


def plot_losses(json_path, name):
    with open(json_path, "r") as f:
        data = json.load(f)

    models = data["models"]

    plt.figure(figsize=(16, 8))

    # Train Loss
    plt.subplot(2, 1, 1)
    for model in models:
        train_losses = model["history"]["train_losses"]
        plt.plot(train_losses, marker="o", label=f"Model {model['model']}")

    plt.ylabel("Train Loss")
    plt.title("Training Loss per Model")
    plt.legend()
    plt.grid(True)
    plt.xlim((0, 10))

    # Test Loss
    plt.subplot(2, 1, 2)
    for model in models:
        val_losses = model["history"]["val_losses"]
        plt.plot(val_losses, marker="o", label=f"Model {model['model']}")

    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss per Model")
    plt.legend()
    plt.grid(True)
    plt.xlim((0, 10))
    
    filepath = os.path.join('results', f'{name}.png')
    plt.savefig(filepath)
    


def plot_mape(json_path, name):
    with open(json_path, "r") as f:
        data = json.load(f)

    models = data["models"]
    ensemble_mape = data["ensemble_mape"]

    model_ids = [m["model"] for m in models]
    train_mape = [m["train_mape"] for m in models]
    val_mape = [m["test_mape"] for m in models]

    plt.figure(figsize=(16, 5))

    # Train MAPE
    plt.subplot(1, 2, 1)
    plt.bar(model_ids, train_mape)

    plt.xlabel("Model")
    plt.ylabel("Train MAPE")
    plt.title("Train MAPE per Model")

    # Test MAPE
    plt.subplot(1, 2, 2)
    plt.bar(model_ids, val_mape)
    plt.axhline(
        ensemble_mape, linestyle="-", color="black", linewidth=2, label="Ensemble MAPE"
    )

    plt.xlabel("Model")
    plt.ylabel("Test MAPE")
    plt.title("Test MAPE per Model")
    plt.legend()

    filepath = os.path.join('results', f'{name}.png')
    plt.savefig(filepath)
