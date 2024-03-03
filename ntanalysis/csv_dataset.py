import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    BAD_POINTS = [
        1643359260,
        1650396180,
        1652526540,
        1656152220,
        1662825660,
        1663951860,
        1664093760,
        1666003140,
        1679421000,
        1689845700,
        1665014400,
        1665705600,
    ]

    @staticmethod
    def target_function(times):
        result = np.zeros(shape=times.shape)
        for point in CsvDataset.BAD_POINTS:
            attempt = np.exp(-0.00014 * (point - times) / 60)
            attempt[attempt > 1] = 0
            result += attempt
        return result.astype(np.float32)

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
        print("Reading dataset from csv...")
        raw_df = pd.read_csv(csv_path)
        # main variables
        N = int(len(raw_df) * max_length)
        raw_df = raw_df.iloc[:N, :]
        # convert day of weeks to one-hot
        one_hot = pd.get_dummies(raw_df["TIME_dow"])
        raw_df = raw_df.iloc[:, 1:12].join(one_hot)
        raw_np = raw_df.to_numpy().astype(np.float32)
        # stack inputs
        inputs = []
        for i in range(input_size):
            start = i * input_gap
            rest = (input_size - 1) * input_gap - start + 1
            inputs.append(raw_np[start:-rest, 0:10])
        self.data_in = np.concatenate(inputs, axis=1)
        # stack outputs
        outputs = []
        for i in range(prediction_size):
            start = i
            rest = prediction_size - start
            outputs.append(
                CsvDataset.target_function(
                    raw_np[prediction_distance + start : -rest, 10:11]
                )
            )
        self.data_out = np.concatenate(outputs, axis=1)
        # trim to match sizes
        length = min(self.data_out.shape[0], self.data_in.shape[0])
        self.data_in = self.data_in[:length, :]
        self.data_out = self.data_out[:length, :]
        print("min and max times in dataset:", raw_np[0, 10], raw_np[-1, 10])

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, index):
        X = self.data_in[index, :]
        y = self.data_out[index, :]
        return X, y

    def __getitems__(self, ids):
        X = self.data_in[ids, :]
        y = self.data_out[ids, :]
        return X, y
