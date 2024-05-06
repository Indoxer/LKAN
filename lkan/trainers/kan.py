import torch
import torch.nn.functional as F

from lkan.loggers import CustomLogger
from lkan.models import KAN

from .base import BaseTrainer


class BasicKANTrainer(BaseTrainer):
    def __init__(
        self,
        model: KAN,
        lr: float,
        update_grid: bool,
        grid_update_freq: int,
        stop_grid_update_step: int,
        logger: CustomLogger,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        lr_scheduler_params: dict,
        lr_step: str,
        clip_grad_norm: float,
        accumulate_grad_batches: int,
        device: str,
    ) -> None:
        super().__init__(
            model=model,
            lr=lr,
            logger=logger,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            lr_step=lr_step,
            clip_grad_norm=clip_grad_norm,
            accumulate_grad_batches=accumulate_grad_batches,
            device=device,
        )
        self.update_grid = update_grid
        self.grid_update_freq = grid_update_freq
        self.stop_grid_update_step = stop_grid_update_step

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # if (
        #     self.global_step % self.grid_update_freq == 0
        #     and self.global_step < self.stop_grid_update_step
        #     and self.update_grid
        # ):
        #     self.model.update_grid(batch[0])
        #     print("grid updated")

        return super().training_step(batch, batch_idx)

    def step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)

        loss = F.mse_loss(y_pred, y)

        logs = {
            "metrics/loss": loss,
        }
        return loss, logs

    def validation_log(self, batch, batch_idx, loss, logs):
        self.logger.log_dict({f"val/{k}": v for k, v in logs.items()}, self.global_step)
