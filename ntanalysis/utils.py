import os

import lightning.pytorch as pl
from gismlops.git_manager import git_version


def configure_loggers_and_callbacks(cfg):
    if not cfg.artifacts.enable_logger:
        return [], []
    os.makedirs("./.logs/my-wandb-logs", exist_ok=True)
    mlflow_tracking_uri = cfg.artifacts.mlflow_tracking_uri
    loggers = [
        pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=mlflow_tracking_uri,
            tags={"commit": git_version()},
        ),
        pl.loggers.WandbLogger(
            project="mlops-logging-demo",
            name=cfg.artifacts.experiment_name,
            save_dir="./.logs/my-wandb-logs",
        ),
    ]
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]
    return loggers, callbacks


def get_default_trainer(cfg):
    loggers, callbacks = configure_loggers_and_callbacks(cfg)

    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        max_steps=cfg.train.num_warmup_steps + cfg.train.num_training_steps,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    return trainer
