import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]
