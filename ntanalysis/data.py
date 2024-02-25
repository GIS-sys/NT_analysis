from typing import Optional

import lightning.pytorch as pl
import torch
from ntanalysis.csv_dataset import CsvDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        halfinterval,
        csv_path,
        batch_size,
        dataloader_num_wokers,
        val_size,
        test_size,
        max_dataset_length,
    ):
        super().__init__()
        self.halfinterval = halfinterval
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.dataloader_num_wokers = dataloader_num_wokers
        self.val_size = val_size
        self.test_size = test_size
        self.max_dataset_length = max_dataset_length

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.full_dataset = CsvDataset(
            halfinterval=self.halfinterval,
            csv_path=self.csv_path,
            max_length=self.max_dataset_length,
        )
        N = len(self.full_dataset)
        # TODO shuffle these indexes
        val_indexes = list(range(0, int(N * self.val_size)))
        test_indexes = list(range(int(N * (1 - self.test_size)), len(self.full_dataset)))
        train_indexes = list(range(int(N * self.val_size), int(N * (1 - self.test_size))))
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indexes)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, val_indexes)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indexes)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_wokers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )
