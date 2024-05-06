import torch.nn.functional as F

from .base import BaseTrainer


class BasicMLPTrainer(BaseTrainer):
    def step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)

        loss = F.mse_loss(y_pred, y)

        logs = {
            "metrics/loss": loss,
        }
        return loss, logs
