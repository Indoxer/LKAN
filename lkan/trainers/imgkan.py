import torch
import torch.nn.functional as F

from lkan.trainers.base import BaseTrainer

from .basickan import BasicKANTrainer


class ImgKANTrainer(BasicKANTrainer):
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if (
            self.global_step % self.grid_update_freq == 0
            and self.global_step < self.stop_grid_update_step
            and self.update_grid
        ):
            self.model.update_grid(batch[0].flatten(1))

        BaseTrainer.training_step(self, batch, batch_idx)

    def step(self, batch, batch_idx):
        x, y = batch

        x = x.flatten(1)

        y_pred = self.model(x)
        y_pred = F.softmax(y_pred, dim=1)

        loss = torch.nn.CrossEntropyLoss()(y_pred, y)
        accuracy = (y_pred.argmax(dim=1) == y).float().mean()

        logs = {
            "metrics/loss": loss,
            "metrics/accuracy": accuracy,
        }
        return loss, logs
