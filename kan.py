import torch
from omegaconf import OmegaConf

from lkan.datamodule import TestDataModule
from lkan.loggers import CustomLogger
from lkan.models import KAN
from lkan.trainers import BasicKANTrainer

# from tmp.pykan.kan import KAN as KAN_real

if __name__ == "__main__":

    model = KAN(layers_dims=[2, 5, 1])

    name = "basickan"
    version = "0.1"
    save_dir = f"./.experiments/{name}/{version}"

    datamodule = TestDataModule(batch_size=16, split_ratio=0.8)
    datamodule.setup()
    logger = CustomLogger(
        save_dir=save_dir, name=name, version=version, cfg=OmegaConf.create({})
    )

    trainer = BasicKANTrainer(
        model=model,
        lr=0.01,
        update_grid=False,
        grid_update_freq=63,
        stop_grid_update_step=10000,
        logger=logger,
        lr_scheduler=None,
        lr_scheduler_params={},
        lr_step=None,
        clip_grad_norm=0.5,
        accumulate_grad_batches=1,
        device="cuda",
    )

    trainer.fit(
        max_epochs=10,
        max_steps=1000,
        validation_every_n_steps=10,
        save_every_n_steps=10,
        datamodule=datamodule,
    )
