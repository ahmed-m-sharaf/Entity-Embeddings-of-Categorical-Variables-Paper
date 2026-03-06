import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder


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


def invert_target(normalized, target_max) :
    if isinstance(normalized, torch.Tensor):
        normalized = normalized.cpu().numpy()
    return np.exp(normalized * target_max)