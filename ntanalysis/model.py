import math
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn


class MyModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(cfg.model.input_size * (10 + 7), cfg.model.layer_scale),
            nn.LeakyReLU(),
            nn.Linear(cfg.model.layer_scale, 2 * cfg.model.layer_scale),
            nn.LeakyReLU(),
            nn.Linear(2 * cfg.model.layer_scale, cfg.model.layer_scale),
            nn.LeakyReLU(),
            nn.Linear(cfg.model.layer_scale, cfg.model.prediction_size * 7),
        )
        self.loss_fn = nn.HuberLoss()

    def forward(self, x):
        # x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred

    def _base_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        return X, y, pred, loss

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        X, y, pred, loss = self._base_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y, pred, loss = self._base_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y, pred, loss = self._base_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "a": pred}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, pred, _ = self._base_step(batch)
        return np.concatenate((x, y, pred), axis=1)

    @staticmethod
    def lr_warmup_wrapper(warmup_steps: int, training_steps: int):
        def lr_warmup(current_step: int):
            if current_step < warmup_steps:
                return float(current_step / warmup_steps)
            else:
                return max(
                    0.0,
                    math.cos(
                        math.pi / 2 * float(current_step - warmup_steps) / training_steps
                    ),
                )

        return lr_warmup

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=type(self).lr_warmup_wrapper(
                warmup_steps=self.cfg.train.num_warmup_steps,
                training_steps=self.cfg.train.num_training_steps,
            ),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
