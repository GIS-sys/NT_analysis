import numpy as np
import pandas as pd
import torch


class CsvDataset(torch.utils.data.Dataset):
    BAD_POINTS = [
        1699463348.0,
        1699765999.0,
        1699784446.0,
        1700265724.0,
        1700292586.0,
        1700512353.0,
        1700576158.0,
        1700703304.0,
        1701717047.0,
        1701724737.0,
        1701865266.0,
        1702440528.0,
        1702796419.0,
        1702826017.0,
        1703228951.0,
    ]

    @staticmethod
    def target_function(times):
        result = np.zeros(shape=times.shape)
        for point in CsvDataset.BAD_POINTS:
            attempt = np.exp(-0.0005 * (point - times) / 60)
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
        max_length=1,
        transform=None,
    ):
        # TODO
        drop_columns = ["TIME", "TIME_seconds"]
        categorial_columns = ["TIME_dow"]
        timestamp_column = "TIME_seconds"
        # TODO
        print("Reading dataset from csv...")
        raw_df = pd.read_csv(csv_path)
        # less data if needed
        N = int(len(raw_df) * max_length)
        raw_df = raw_df.iloc[:N, :]
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
            timestamps_df = raw_df[[timestamp_column]][prediction_distance:-rest]
            outputs.append(
                CsvDataset.target_function(timestamps_df.to_numpy().astype(np.float32))
            )
        timestamps_df = raw_df[[timestamp_column]][prediction_distance:-prediction_size]
        inputs = [timestamps_df.to_numpy().astype(np.float32)] + inputs
        # trim to match sizes
        length = min(inputs[0].shape[0], outputs[0].shape[0])
        inputs = [x[:length, :] for x in inputs]
        outputs = [x[:length, :] for x in outputs]
        # stack
        self.data_in = np.concatenate(inputs, axis=1)
        self.data_out = np.concatenate(outputs, axis=1)
        print("min and max times in dataset:", raw_df.iloc[0, 0], raw_df.iloc[-1, 0])
        if raw_df.shape[0] > 3600:
            print("3600th time:", raw_df.iloc[3600, 0])

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
