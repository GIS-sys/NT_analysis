import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from gismlops.data import MyDataModule
from gismlops.model import MyModel
from gismlops.utils import get_default_trainer
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    dm = MyDataModule(
        batch_size=cfg.data.batch_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        val_size=cfg.data.val_size,
    )
    model = MyModel(cfg)
    model.load_state_dict(
        torch.load(cfg.artifacts.model.path + cfg.artifacts.model.name + ".pth")
    )

    cfg.callbacks.swa.use = False
    cfg.artifacts.checkpoint.use = False
    trainer = get_default_trainer(cfg)

    trainer.test(model, datamodule=dm)
    answers = np.concatenate(trainer.predict(model, datamodule=dm), axis=1).T

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    answersDataFrame = pd.DataFrame(answers, columns=["target_index", "predicted_index"])
    answersDataFrame["target_label"] = answersDataFrame["target_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame["predicted_label"] = answersDataFrame["predicted_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame.to_csv("data/test.csv", index=False)
    return answersDataFrame


if __name__ == "__main__":
    infer()
