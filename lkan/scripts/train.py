from omegaconf import OmegaConf

from lkan.loggers import CustomLogger
from lkan.utils import custom_import


def run(cfg: OmegaConf, logger: CustomLogger):
    model = custom_import(cfg.model)(**cfg.model_params)

    datamodule = custom_import(cfg.datamodule)(**cfg.datamodule_params)
    datamodule.setup()

    lr_scheduler = custom_import(cfg.lr_scheduler)

    trainer = custom_import(cfg.trainer)(
        model=model,
        logger=logger,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=cfg.lr_scheduler_params,
        **cfg.trainer_params
    )

    trainer.fit(datamodule=datamodule, **cfg.train_params)
