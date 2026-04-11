"""Model architecture."""

import torch
import torch.nn as nn


class RossmannEmbeddingModel(nn.Module):
    """Rossmann embedding model."""
    
    def __init__(self, config):
        super().__init__()
        
        self.output_sigmoid = config.output_sigmoid
        self.batch_norm = config.batch_norm
        
        # Expecting encoder directly passed via config from dataloader process
        encoder = config.encoder
        emb_dims = config.emp_dims
        hidden_sizes = config.hidden_sizes

        n_store = len(encoder.categories_[0])
        n_dow   = len(encoder.categories_[1])
        n_day   = len(encoder.categories_[2])
        n_month = len(encoder.categories_[3])
        n_year  = len(encoder.categories_[4])
        n_promo = len(encoder.categories_[5])
        n_state = len(encoder.categories_[6])

        self.store_embed = nn.Embedding(n_store, emb_dims["Store"])
        self.dow_embed   = nn.Embedding(n_dow, emb_dims["DayOfWeek"])
        self.day_embed   = nn.Embedding(n_day, emb_dims["day"])
        self.month_embed = nn.Embedding(n_month, emb_dims["month"])
        self.year_embed  = nn.Embedding(n_year, emb_dims["year"])
        self.promo_embed = nn.Embedding(n_promo, emb_dims["Promo"])
        self.state_embed = nn.Embedding(n_state, emb_dims["State"])

        self.total_emb_dim = (
            emb_dims["Store"]
            + emb_dims["DayOfWeek"]
            + emb_dims["day"]
            + emb_dims["month"]
            + emb_dims["year"]
            + emb_dims["Promo"]
            + emb_dims["State"]
        )

        self.fc1 = nn.Linear(self.total_emb_dim, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0]) if self.batch_norm else None
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1]) if self.batch_norm else None
        self.out = nn.Linear(hidden_sizes[1], 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def get_embedding_space(self, x_cat):
        x_cat = x_cat.long()

        store = self.store_embed(x_cat[:, 0])
        dow   = self.dow_embed(x_cat[:, 1])
        day   = self.day_embed(x_cat[:, 2])
        month = self.month_embed(x_cat[:, 3])
        year  = self.year_embed(x_cat[:, 4])
        promo = self.promo_embed(x_cat[:, 5])
        state = self.state_embed(x_cat[:, 6])

        embeddings = torch.cat([store, dow, day, month, year, promo, state], dim=1)
        return embeddings

    def forward(self, x_cat, x_cont=None):
        x = self.get_embedding_space(x_cat)

        x = self.relu(self.fc1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(self.fc2(x))
        if self.batch_norm:
            x = self.bn2(x)
        x = self.out(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


def create_model(config):
    """Create model from config."""
    return RossmannEmbeddingModel(config)
