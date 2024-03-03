import hydra
import lightning.pytorch as pl
import torch
from ntanalysis.data import MyDataModule
from ntanalysis.model import MyModel
from ntanalysis.utils import get_default_trainer
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    dm = MyDataModule(cfg)
    model = MyModel(cfg)

    trainer = get_default_trainer(cfg)

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
