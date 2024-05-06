import torch
import torch.nn.functional as F

from .basicmlp import BasicMLPTrainer


class ImgMLPTrainer(BasicMLPTrainer):
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
