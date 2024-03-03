import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        input_size,
        input_gap,
        prediction_distance,
        prediction_size,
        max_length=None,
        transform=None,
    ):
        raw_df = pd.read_csv(csv_path)
        if max_length is None:
            max_length = 1
        dim_in = input_size * (10 + 7)
        dim_out = prediction_size * 7  # * (10 + 7)
        N = int(len(raw_df) * max_length)
        # N>=length+input_size*input_gap; N>=length+prediction_size+prediction_distance
        length = (
            N - max(input_size * input_gap, prediction_size + prediction_distance) - 1
        )
        self.data_in = np.zeros(shape=(length, dim_in), dtype=np.float32)
        self.data_out = np.zeros(shape=(length, dim_out), dtype=np.float32)
        for x in range(length):
            for i in range(input_size):
                row_source = x + i * input_gap
                y_bias = (10 + 7) * i
                for y in range(0, 10):
                    self.data_in[x][y_bias + y] = raw_df.iloc[row_source, 1 + y]
                self.data_in[x][y_bias + 10 + raw_df["TIME_dow"][row_source]] = 1
            for i in range(prediction_size):
                row_source = x + prediction_distance + i
                y_bias = 7 * i
                self.data_out[x][y_bias + raw_df["TIME_dow"][row_source]] = 1

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, index):
        X = self.data_in[index, :]
        y = self.data_out[index, :]
        return X, y
