from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .rossmann_store import RossmannDataset
from .config import *
from .preprocessor import (
    encode_categorical,
    feature_engineering,
    load_df,
    transform_target,
)
import joblib
import json 

def split_data(df, split_type = SPLIT_TYPE):
    shuffle = False if split_type.lower() != 'random' else True
    if not shuffle:
        df = df.sort_values(by = DATE_COL, ascending=True).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=shuffle)
    return train_df, test_df

def prepare_data(split_type = SPLIT_TYPE):
    df = load_df(DATA_PATH, STORE_STATE)
    # for col in df.columns:
        # print(f"Column {col}: ")
        # print(df[col].value_counts())
    df = feature_engineering(df) # to get day, year, month
    
    train_df, test_df = split_data(df, split_type)
    train_df, target_max = transform_target(train_df, TARGET_COL)
    test_df, _ = transform_target(test_df, TARGET_COL, target_max)
    train_df, encoder = encode_categorical(train_df, CATEGORICAL_COLS)
    test_df, _ = encode_categorical(test_df, CATEGORICAL_COLS, encoder)
    joblib.dump(encoder, "encoder.pkl")
    json.dump({
        'target_max': target_max
    }, open('target_max.json', 'w'), indent=4)
    return train_df, test_df, encoder, target_max

def make_loaders(train_df, test_df, shuffle=True):
    ds_train = RossmannDataset(train_df, CATEGORICAL_COLS, TARGET_COL)
    ds_test = RossmannDataset(test_df, CATEGORICAL_COLS, TARGET_COL)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=shuffle)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    return ds_train, ds_test, loader_train, loader_test


if __name__ == '__main__':
    train_df, test_df, encoder, target_max = prepare_data('time')
    print(train_df)
    print('-'*50)
    print(test_df)
    print('-'*50)
    for i in range(len(CATEGORICAL_COLS)):
        print(encoder.categories_[i])
    print('-'*50)
    print(target_max)
    print()
    print()
    
    ds_train, ds_test, train_loader, test_loader = make_loaders(train_df, test_df, shuffle=True)
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        break