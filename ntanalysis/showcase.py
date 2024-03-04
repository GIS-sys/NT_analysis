from datetime import datetime

import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from ntanalysis.data import MyDataModule
from ntanalysis.model import MyModel
from ntanalysis.utils import get_default_trainer
from omegaconf import DictConfig


BAD_POINT_THRESHOLD = 0.9
BAD_POINT_HEIGHT = 2
Y_AXIS = [0, 1.1]
TOTAL_FRAMES = 720
FPS = 24
TRAILING_PLOT = 0.1
TRAILING_CUMMEAN = 1000
ZIGZAG_THRESHOLDS = [(1, 0.4), (1, 0.8), (-1, 0.32)]
ARROW_PROPS = {"width": 1}


def animate_plot(t, data_plot, zigzag_base_data):
    # Main vars
    SLIDER_LENGTH = t.shape[0]
    # Calculate thresholds
    zigzag_index = 0
    arr = zigzag_base_data
    zigzag_positions = [0]
    while True:
        sign, thresh = ZIGZAG_THRESHOLDS[zigzag_index]
        found = np.where(arr * sign > thresh * sign)[0]
        if found.size == 0:
            break
        pos = found[0]
        arr = arr[pos:]
        zigzag_positions.append(zigzag_positions[-1] + pos)
        zigzag_index = (zigzag_index + 1) % len(ZIGZAG_THRESHOLDS)
    print(zigzag_positions)
    # Create plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # Initial plot
    for label, datum in data_plot:
        ax.plot(t, datum, label=label)
    for i, pos in enumerate(zigzag_positions):
        condition = ZIGZAG_THRESHOLDS[i % len(ZIGZAG_THRESHOLDS)]
        ax.annotate(
            str(condition), (t[pos], zigzag_base_data[pos]), arrowprops=ARROW_PROPS
        )
    ax.set_xlim(t[0], t[1])
    ax.set_ylim(Y_AXIS)
    # Slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, "X Max", 0, SLIDER_LENGTH, valinit=0)

    # Update function for the plot based on slider value
    def update(val):
        x_max = slider.val
        x_min = max(0, x_max - int(SLIDER_LENGTH * TRAILING_PLOT))
        ax.set_xlim(t[x_min], t[x_max])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Animation setting the slider values from 0 to 10 over time
    def update_animation(frame):
        slider.set_val(frame)
        return slider

    _ = FuncAnimation(
        fig,
        update_animation,
        frames=np.arange(0, SLIDER_LENGTH + 1, SLIDER_LENGTH // TOTAL_FRAMES),
        interval=1000 // FPS,
        blit=False,
    )
    # Plot
    plt.legend()
    plt.show()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def showcase(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    cfg.data.val_size = 0.01
    cfg.data.test_size = 0.98
    cfg.data.max_dataset_length = 0.76
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

    answers = trainer.predict(model, datamodule=dm)
    answers = np.concatenate(answers)
    print(answers)

    # TODO FIX TIME AXIS INTERVALS IF NEEDED
    # t = np.linspace(0, 1, answers.shape[0])
    t = pd.date_range(
        datetime.fromtimestamp(1633824000),
        datetime.fromtimestamp(1664613100),
        answers.shape[0],
    )
    data_plot = []
    input_end = cfg.model.input_size * 10
    output_end = input_end + cfg.model.prediction_size * 1
    data_plot.append(("output", answers[:, input_end]))
    data_plot.append(
        (
            "bad point",
            (answers[:, input_end] > BAD_POINT_THRESHOLD).astype(int) * BAD_POINT_HEIGHT,
        )
    )
    # data_plot.append(("prediction", answers[:, output_end]))
    pred_cumsum = np.cumsum(answers[:, output_end])
    pred_cummean_trailing = (
        pred_cumsum[TRAILING_CUMMEAN:] - pred_cumsum[:-TRAILING_CUMMEAN]
    ) / TRAILING_CUMMEAN
    pred_plot = np.concatenate(
        (answers[:TRAILING_CUMMEAN, output_end], pred_cummean_trailing)
    )
    data_plot.append(("prediction", pred_plot))
    animate_plot(t, data_plot, pred_plot)


if __name__ == "__main__":
    showcase()
