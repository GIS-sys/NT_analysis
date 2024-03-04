import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from ntanalysis.data import MyDataModule
from ntanalysis.model import MyModel
from ntanalysis.utils import get_default_trainer
from omegaconf import DictConfig


BAD_POINT_THRESHOLD = 0.9
TOTAL_FRAMES = 360
FPS = 24
TRAILING = 0.1


def animate_plot(t, data_plot):
    # Main vars
    SLIDER_LENGTH = t.shape[0]
    MAX_LIM = t[-1]
    # Create plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # Initial plot
    for label, datum in data_plot:
        ax.plot(t, datum, label=label)
    ax.set_xlim(0, MAX_LIM)
    # Slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, "X Max", 0, SLIDER_LENGTH, valinit=0)

    # Update function for the plot based on slider value
    def update(val):
        x_max = slider.val / SLIDER_LENGTH * MAX_LIM
        x_min = max(0, x_max - MAX_LIM * TRAILING)
        ax.set_xlim(x_min, x_max)
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
    cfg.data.max_dataset_length = 0.73
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

    t = np.linspace(0, 1, answers.shape[0])
    data_plot = []
    input_end = cfg.model.input_size * 10
    output_end = input_end + cfg.model.prediction_size * 1
    data_plot.append(("output", answers[:, input_end]))
    data_plot.append(
        ("bad point", (answers[:, input_end] > BAD_POINT_THRESHOLD).astype(int))
    )
    data_plot.append(("prediction", answers[:, output_end]))
    # TODO plot mean
    animate_plot(t, data_plot)


if __name__ == "__main__":
    showcase()
