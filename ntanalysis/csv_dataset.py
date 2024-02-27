import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, halfinterval, max_length=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.max_length = max_length
        if max_length is None:
            self.max_length = 1
        self.halfinterval = halfinterval

    def __len__(self):
        return int(len(self.df) * self.max_length) - 2 * self.halfinterval

    def __getitem__(self, index):
        index += self.halfinterval
        sensors = self.df.iloc[
            index - self.halfinterval : index + self.halfinterval + 1, :
        ]
        X = np.asarray(sensors.iloc[:, 7], dtype=np.float32)
        y = np.asarray(sensors.iloc[self.halfinterval, 1:2], dtype=np.float32)
        # print (X,y)
        return X, y
