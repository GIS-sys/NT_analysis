import hydra
import lightning.pytorch as pl
import torch
from gismlops.data import MyDataModule
from gismlops.model import MyModel
from gismlops.utils import get_default_trainer
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    dm = MyDataModule(
        batch_size=cfg.data.batch_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        val_size=cfg.data.val_size,
    )
    model = MyModel(cfg)

    trainer = get_default_trainer(cfg)

    if cfg.train.batch_size_finder:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)
    torch.save(
        model.state_dict(), cfg.artifacts.model.path + cfg.artifacts.model.name + ".pth"
    )
    dummy_input_batch = next(iter(dm.val_dataloader()))[0]
    dummy_input = torch.unsqueeze(dummy_input_batch[0], 0)
    torch.onnx.export(
        model,
        dummy_input,
        cfg.artifacts.model.path + cfg.artifacts.model.name + ".onnx",
        export_params=True,
        input_names=["inputs"],
        output_names=["predictions"],
        dynamic_axes={
            "inputs": {0: "BATCH_SIZE"},
            "predictions": {0: "BATCH_SIZE"},
        },
    )


if __name__ == "__main__":
    train()
