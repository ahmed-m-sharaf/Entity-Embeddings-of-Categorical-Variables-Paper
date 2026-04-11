"""Data loading script."""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


DATA_PATH = "data/train.csv"
STORE_STATE = "data/store_states.csv"
TARGET_COL = "Sales"
DATE_COL = "Date"
CATEGORICAL_COLS = ["Store", "DayOfWeek", "day", "month", "year", "Promo", "State"]


def load_df(train_path, store_path):
    train_df = pd.read_csv(train_path, low_memory=False)
    store_state_df = pd.read_csv(store_path, low_memory=False)
    df = pd.merge(train_df, store_state_df, on="Store", how="left")
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()


def feature_engineering(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    return df


def transform_target(df, target_col, target_max=None):
    df[target_col] = np.log(df[target_col])
    if target_max is None:
        target_max = df[target_col].max()
    df[target_col] = df[target_col] / target_max
    return df, target_max


def encode_categorical(df, categorical_cols, encoder=None):
    oe = encoder
    if oe is not None:
        df[categorical_cols] = encoder.transform(df[categorical_cols])
    else:
        oe = OrdinalEncoder(dtype="int")
        df[categorical_cols] = oe.fit_transform(df[categorical_cols])
    return df, oe


def invert_target(normalized, target_max):
    if isinstance(normalized, torch.Tensor):
        normalized = normalized.cpu().numpy()
    return np.exp(normalized * target_max)


def split_data(df, config, seed):
    shuffle = False if config.split_type.lower() != 'random' else True
    if not shuffle:
        df = df.sort_values(by=DATE_COL, ascending=True).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=config.test_size, random_state=seed, shuffle=shuffle)
    return train_df, test_df


def prepare_data(config, seed):
    df = load_df(DATA_PATH, STORE_STATE)
    df = feature_engineering(df)
    
    train_df, test_df = split_data(df, config, seed)
    train_df, target_max = transform_target(train_df, TARGET_COL)
    test_df, _ = transform_target(test_df, TARGET_COL, target_max)
    train_df, encoder = encode_categorical(train_df, CATEGORICAL_COLS)
    test_df, _ = encode_categorical(test_df, CATEGORICAL_COLS, encoder)
    
    # Save encoder and target_max to use during inference/evaluation later
    joblib.dump(encoder, "encoder.pkl")
    with open('target_max.json', 'w') as f:
        json.dump({'target_max': target_max}, f, indent=4)
        
    return train_df, test_df, encoder, target_max


class RossmannDataset(Dataset):
    """Rossmann Dataset."""

    def __init__(self, df, categorical_cols, target_col):
        self.X = torch.tensor(df[categorical_cols].values, dtype=torch.long)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(config, split='train'):
    """Create dataloader."""
    # Note: For multiple models in an ensemble, the seed changes. 
    # For a generic get_dataloader, it typically depends on the config seed.
    seed = config.seed
    
    # Because prepare_data processes both train and test exactly together to share scalers and encoders
    # We call it once and return the desired loader
    train_df, test_df, encoder, target_max = prepare_data(config, seed)
    
    df = train_df if split == 'train' else test_df
    dataset = RossmannDataset(df, CATEGORICAL_COLS, TARGET_COL)
    
    # We provide the global configs target_max attribute so it can be extracted if needed later
    config.target_max = target_max
    config.encoder = encoder
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train' and config.shuffle_loader)
    )
    
    return dataloader
