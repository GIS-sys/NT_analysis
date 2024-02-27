import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import torch
from ntanalysis.data import MyDataModule
from ntanalysis.model import MyModel
from ntanalysis.utils import get_default_trainer
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    cfg.artifacts.enable_logger = False
    dm = MyDataModule(
        halfinterval=cfg.model.halfinterval,
        csv_path=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        max_dataset_length=cfg.data.max_dataset_length,
    )
    model = MyModel(cfg)
    model.load_state_dict(
        torch.load(cfg.artifacts.model.path + cfg.artifacts.model.name + ".pth")
    )

    cfg.callbacks.swa.use = False
    cfg.artifacts.checkpoint.use = False
    trainer = get_default_trainer(cfg)

    trainer.test(model, datamodule=dm)
    answers = trainer.predict(model, datamodule=dm)
    answers = np.concatenate(answers)

    t = np.linspace(0, 1, answers.shape[0])
    plt.plot(t, answers[:, cfg.model.halfinterval], color="blue")
    plt.plot(t, answers[:, -2], color="green")
    plt.plot(t, answers[:, -1], color="red")
    plt.show()

    return answers


if __name__ == "__main__":
    infer()
