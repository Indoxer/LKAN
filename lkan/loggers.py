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


class CustomLogger:
    def __init__(self, save_dir: str, name: str, version: str, cfg: OmegaConf):
        super().__init__()

        self.save_dir = save_dir
        self._version = version

        self.writer = SummaryWriter(
            log_dir=save_dir,
        )

        hparams = flatten_dict(cfg)

        for key, value in hparams.items():
            if not isinstance(value, str):
                hparams[key] = str(value)

        self.hparams = hparams
        self.metrics = {}

    @property
    def name(self):
        return "CustomLogger"

    @property
    def version(self):
        return self._version

    def update_hyperparams(self, hparams: dict, metrics: dict):
        self.hparams.update(hparams)
        self.metrics.update(metrics)

    def log_dict(self, params: dict, step: int):
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
