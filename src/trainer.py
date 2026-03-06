import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        criterion,
        n_epochs: int = 100,
        device: str = "cpu",
    ):
        self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.criterion = criterion

        self.n_epochs = n_epochs
        self.history = {"train_losses": [], "val_losses": []}

    def train_epoch(self):
        self.model.train()
        train_loss = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.n_epochs):
            print(f"\nEpoch [{epoch + 1}/{self.n_epochs}]")

            # Train
            train_loss = self.train_epoch()
            self.history["train_losses"].append(train_loss)

            # Validate
            val_loss = self.validate()
            self.history["val_losses"].append(val_loss)

            log_msg = f"Epoch {epoch + 1:3d}/{self.n_epochs} {int((epoch + 1) / self.n_epochs * 100):3d}% [Loss (Train={train_loss:.8f}, Val={val_loss:.8f})]"
            print(log_msg)

        return self.model, self.history
