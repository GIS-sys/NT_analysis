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
    print(answers)

    t = np.linspace(0, 1, answers.shape[0])
    data_plot = []
    # for i in range(cfg.model.input_size):
    #    data_plot.append(
    #        (f"input {i}", np.argmax(answers[:, 17 * i + 10 : 17 * i + 10 + 7], axis=1))
    #    )
    # input_end = cfg.model.input_size * (10 + 7)
    # for i in range(cfg.model.prediction_size):
    #    data_plot.append(
    #        (
    #            f"output {i}",
    #            np.argmax(answers[:, input_end + 10 * i : input_end + 10 * i + 7], axis=1),
    #        )
    #    )
    # output_end = input_end + cfg.model.prediction_size * 7
    # for i in range(cfg.model.prediction_size):
    #    data_plot.append(
    #        (
    #            f"prediction {i}",
    #            np.argmax(
    #                answers[:, output_end + 7 * i : output_end + 7 * i + 7], axis=1
    #            ),
    #        )
    #    )
    input_end = cfg.model.input_size * (10 + 7)
    output_end = input_end + cfg.model.prediction_size * 10
    data_plot.append(("output", answers[:, input_end + 3]))
    data_plot.append(("output1", answers[:, input_end + 4]))
    # data_plot.append((f"prediction 1", answers[:, output_end + K-2]))
    # data_plot.append((f"prediction 2", answers[:, output_end + K-1]))
    data_plot.append(("prediction", answers[:, output_end + 3]))
    data_plot.append(("prediction1", answers[:, output_end + 4]))
    for label, datum in data_plot:
        plt.plot(t, datum, label=label)
    plt.legend()
    plt.show()

    return answers


if __name__ == "__main__":
    infer()
