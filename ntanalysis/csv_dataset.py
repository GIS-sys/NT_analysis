import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    BAD_POINTS = [
        1699233044,
        1699252654,
        1700359390,
        1710494630,
    ]

    @staticmethod
    def target_function(times):
        result = np.zeros(shape=times.shape)
        for point in CsvDataset.BAD_POINTS:
            attempt = np.exp(-0.00004 * (point - times) / 60)
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
        # TODO
        drop_columns = ["TIME"]
        categorial_columns = ["TIME_dow"]
        # TODO
        print("Reading dataset from csv...")
        raw_df = pd.read_csv(csv_path)
        # remove columns
        tmp_df = raw_df.drop(drop_columns, axis=1)
        # convert to one-hot
        for col in categorial_columns:
            one_hot = pd.get_dummies(tmp_df[col])
            tmp_df = tmp_df.drop([col], axis=1).join(one_hot)
        # to numpy
        raw_np = tmp_df.to_numpy().astype(np.float32)
        # stack inputs
        inputs = []
        for i in range(input_size):
            start = i * input_gap
            rest = (input_size - 1) * input_gap - start + 1
            inputs.append(raw_np[start:-rest, :])
        # stack outputs
        outputs = []
        for i in range(prediction_size):
            start = i
            rest = prediction_size - start
            outputs.append(
                CsvDataset.target_function(raw_np[prediction_distance:-rest, 2:3])
            )
        inputs = [raw_np[prediction_distance:-prediction_size, 2:3]] + inputs
        # trim to match sizes
        length = min(inputs[0].shape[0], outputs[0].shape[0])
        inputs = [x[:length, :] for x in inputs]
        outputs = [x[:length, :] for x in outputs]
        # stack
        self.data_in = np.concatenate(inputs, axis=1)
        self.data_out = np.concatenate(outputs, axis=1)
        print("min and max times in dataset:", raw_df.iloc[0, 0], raw_df.iloc[-1, 0])

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
