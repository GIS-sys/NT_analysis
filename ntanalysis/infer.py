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
    dm = MyDataModule(cfg)
    model = MyModel(cfg)
    model.load_state_dict(
        torch.load(cfg.artifacts.model.path + cfg.artifacts.model.name + ".pth")
    )

    cfg.callbacks.swa.use = False
    cfg.artifacts.checkpoint.use = False
    trainer = get_default_trainer(cfg)

    # trainer.test(model, datamodule=dm)
    answers = trainer.predict(model, datamodule=dm)
    answers = np.concatenate(answers)

    t = np.linspace(0, 1, answers.shape[0])
    print(answers)
    x = np.argmax(answers[:, 10 : 10 + 7], axis=1)
    y = np.argmax(answers[:, 10 + 7 : 10 + 14], axis=1)
    pred = np.argmax(answers[:, 10 + 14 :], axis=1)
    print(x)
    print(y)
    print(pred)
    plt.plot(t, x, color="blue")
    plt.plot(t, y, color="green")
    plt.plot(t, pred, color="red")
    plt.show()

    return answers


if __name__ == "__main__":
    infer()
