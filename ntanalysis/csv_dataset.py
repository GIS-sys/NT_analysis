import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_length=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.max_length = max_length
        if max_length is None:
            self.max_length = 1

    def __len__(self):
        return int(len(self.df) * self.max_length)

    def __getitem__(self, index):
        sensors = self.df.iloc[index, 1:]
        X = np.asarray(sensors[8:9], dtype=np.float32)
        y = np.asarray(sensors[8:9], dtype=np.float32)
        return X, y
