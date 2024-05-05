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
        save_dir = cfg.save_dir

        new_save_dir = save_dir

        i = 0

        if os.path.exists(save_dir) is False:
            new_save_dir = f"{save_dir}/run0"

        while os.path.exists(new_save_dir):
            new_save_dir = f"{save_dir}/run{str(i)}"
            i += 1

        save_dir = new_save_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(cfg, f"{save_dir}/config.yaml")

        cfg.save_dir = new_save_dir

        script = custom_import(cfg.script)

        logger = CustomLogger(
            save_dir=cfg.save_dir, name=cfg.name, version=cfg.version, cfg=cfg
        )
        try:
            script.run(cfg, logger)
        except KeyboardInterrupt:
            logger.finalize()
