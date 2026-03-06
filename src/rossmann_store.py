import torch
from torch.utils.data import Dataset


class RossmannDataset(Dataset):
    def __init__(self, df, categorical_cols, target_col, num_feature = False):
        self.num_feat = num_feature
        
        self.X = torch.tensor(df[categorical_cols].values, dtype=torch.long)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float).view(-1, 1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]