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
    # TODO
    target_columns_amount = 347
    # TODO
    pl.seed_everything(cfg.general.seed)
    cfg.data.val_size = 0.01
    cfg.data.test_size = 0.98
    cfg.data.max_dataset_length = 0.75
    cfg.data.batch_size = 4096
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
    print(answers)
    print(answers.shape)

    t = answers[:, 0]
    answers = answers[:, 1:]
    data_plot = []

    input_end = cfg.model.input_size * target_columns_amount
    output_end = input_end + cfg.model.prediction_size * 1
    data_plot.append(("output", answers[:, input_end]))
    data_plot.append(("prediction", answers[:, output_end]))
    # data_plot.append(("input_1", answers[:, 0]))
    for label, datum in data_plot:
        plt.plot(t, datum, label=label)
    plt.legend()
    plt.show()

    return answers


if __name__ == "__main__":
    infer()
