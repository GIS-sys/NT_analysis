from typing import Optional

import lightning.pytorch as pl
import torch
import torchvision


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataloader_num_wokers, val_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataloader_num_wokers = dataloader_num_wokers
        self.val_size = val_size

    def prepare_data(self):
        torchvision.datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
        )
        torchvision.datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_val_dataset = torchvision.datasets.FashionMNIST(
            root="data",
            train=True,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root="data",
            train=False,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )
        train_indexes = list(
            range(0, int(len(self.train_val_dataset) * (1 - self.val_size)))
        )
        val_indexes = list(
            range(
                int(len(self.train_val_dataset) * (1 - self.val_size)),
                len(self.train_val_dataset),
            )
        )
        self.train_dataset = torch.utils.data.Subset(
            self.train_val_dataset, train_indexes
        )
        self.val_dataset = torch.utils.data.Subset(self.train_val_dataset, val_indexes)

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
