from typing import Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from ntanalysis.csv_dataset import CsvDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.dataloader_num_wokers = cfg.data.dataloader_num_wokers
        self.val_size = cfg.data.val_size
        self.test_size = cfg.data.test_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.full_dataset = CsvDataset(
            input_size=self.cfg.model.input_size,
            input_gap=self.cfg.data.input_gap,
            prediction_distance=self.cfg.data.prediction_distance,
            prediction_size=self.cfg.model.prediction_size,
            csv_path=self.cfg.data.csv_path,
            max_length=self.cfg.data.max_dataset_length,
        )
        N = len(self.full_dataset)
        # get shuffled indexes
        test_size = int(N * self.test_size)
        val_size = int(N * self.val_size)
        indexes = np.arange(N)
        # test_start_ind = np.random.randint(0, N - test_size)
        test_start_ind = (N - test_size) // 2
        test_end_ind = test_start_ind + test_size
        test_indexes = indexes[test_start_ind:test_end_ind]
        indexes = np.concatenate((indexes[0:test_start_ind], indexes[test_end_ind:]))
        np.random.shuffle(indexes)
        val_indexes = indexes[:val_size]
        train_indexes = indexes[val_size:]
        # subset from full dataset
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indexes)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, val_indexes)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indexes)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, tuple):
            # Tuple[Numpy, Numpy], (2, batch_size, features)
            X, y = batch
            X, y = torch.from_numpy(X), torch.from_numpy(y)
            return X, y
        elif isinstance(batch, list):
            # List[Tuple[Numpy, Numpy]], (batch_size, 2, features)
            X, y = zip(*batch, strict=True)
            X, y = torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(y))
            return X, y
        else:
            raise Exception("Unexpected type inn collate_fn")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
            persistent_workers=True,
            collate_fn=MyDataModule.collate_fn,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_wokers,
            persistent_workers=True,
            collate_fn=MyDataModule.collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
            persistent_workers=True,
            collate_fn=MyDataModule.collate_fn,
        )
