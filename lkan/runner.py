import os

from omegaconf import OmegaConf

from lkan.loggers import CustomLogger
from lkan.utils.base import custom_import


class Runner:
    def __init__(self):
        pass

    def load_config(self, config_path: str) -> OmegaConf:
        cfg = OmegaConf.load(config_path)
        return cfg

    def run(self, cfg: OmegaConf) -> None:
        logger = CustomLogger(
            save_dir=cfg.save_dir, name=cfg.name, version=cfg.version, cfg=cfg
        )
        cfg.save_dir = logger.save_dir

        script = custom_import(cfg.script)

        try:
            script.run(cfg, logger)
        except KeyboardInterrupt:
            logger.finalize()
