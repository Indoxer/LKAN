import os

from omegaconf import OmegaConf

from lkan.utils.base import custom_import


class Runner:
    def __init__(self):
        pass

    def load_config(self, config_path: str) -> OmegaConf:
        cfg = OmegaConf.load(config_path)
        return cfg

    def run(self, cfg: OmegaConf) -> None:
        logger = custom_import(cfg.logger)(**cfg.logger_params, cfg=cfg)

        # Logger can change save_dir
        if hasattr(logger, "save_dir"):
            cfg.save_dir = logger.save_dir

        script = custom_import(cfg.script)

        try:
            script.run(cfg, logger)
        except KeyboardInterrupt:
            logger.finalize()
