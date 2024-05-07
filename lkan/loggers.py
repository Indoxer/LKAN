import os

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter, summary


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class BaseLogger:
    def __init__(self, *args, cfg=None, **kwargs):
        pass

    def update_hyperparams(self, hparams: dict, metrics: dict):
        pass

    def log_dict(self, params: dict, step: int):
        pass

    def add_images(self, tag: str, images: torch.Tensor, step: int):
        pass

    def save_model(self, model, step):
        pass

    def finalize(self):
        pass

    def setup(self, trainer):
        self.trainer = trainer


class PrintLogger(BaseLogger):
    def __init__(
        self,
        print_every_n_train_calls: int = 100,
        print_every_n_val_calls: int = 10,
        *args,
        cfg=None,
        **kwargs,
    ):
        self.limits = {
            True: print_every_n_train_calls,
            False: print_every_n_val_calls,
        }
        self.counters = {
            True: 0,
            False: 0,
        }
        super().__init__()

    def log_dict(self, params: dict):
        step = self.trainer.global_step

        training = self.trainer.training

        self.counters[training] += 1

        if (
            self.counters[training] == self.limits[training]
        ):  # TODO: print after last step
            print(f"Step: {step} - {'Train' if training else 'Validation'} - Metrics:")
            for key, value in params.items():
                print(f"{key}: {value}")
            print("\n")
            self.counters[training] = 0

    def finalize(self):
        print("End...")


class TensorBoardLogger(BaseLogger):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        cfg = self.folder_structure(cfg)

        self.save_dir = cfg.save_dir
        self._version = cfg.version

        self.writer = SummaryWriter(
            log_dir=self.save_dir,
        )

        hparams = flatten_dict(cfg)

        for key, value in hparams.items():
            if not isinstance(value, str):
                hparams[key] = str(value)

        self.hparams = hparams
        self.metrics = {}

    def folder_structure(self, cfg: OmegaConf):
        new_save_dir = f"{cfg.save_dir}/run0"

        i = 1

        while os.path.exists(new_save_dir):
            new_save_dir = f"{cfg.save_dir}/run{str(i)}"
            i += 1

        os.makedirs(new_save_dir, exist_ok=True)
        os.makedirs(f"{new_save_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(cfg, f"{new_save_dir}/config.yaml")

        cfg.save_dir = new_save_dir

        return cfg

    def update_hyperparams(self, hparams: dict, metrics: dict):
        self.hparams.update(hparams)
        self.metrics.update(metrics)

    def log_dict(
        self, params: dict
    ):  # TODO: What if validation_every_n_steps is "epoch"? it create average of all steps?
        step = self.trainer.global_step
        for key, value in params.items():
            self.writer.add_scalar(key, value, step)

    def add_images(self, tag: str, images: torch.Tensor, step: int):
        self.writer.add_images(tag, images, step)

    # def add_audio(self, tag: str, audio: torch.Tensor, step: int, sr: int = 48000):
    #     audio = audio.flatten()
    #     self.writer.add_audio(tag, audio, step, sample_rate=sr)

    def save_model(self, model, step):
        torch.save(model.state_dict(), f"{self.save_dir}/checkpoints/model_{step}.pt")

    def finalize(self):
        exp, ssi, sei = summary.hparams(self.hparams, self.metrics, {})
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)
        for k, v in self.metrics.items():
            self.writer.add_scalar(k, v)
        self.writer.close()
