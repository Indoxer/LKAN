import torch
from cv2 import accumulate
from msm.loggers import CustomLogger
from msm.loss import vae_loss
from msm.models import VAE

from .base import BaseTrainer

# structure for trainer from my older project.


class VAETrainer(BaseTrainer):
    def __init__(
        self,
        model: VAE,
        lr: float,
        mse_coeff: float,
        kl_coeff: float,
        logger: CustomLogger,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        lr_scheduler_params: dict,
        lr_step: str,
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
            accumulate_grad_batches=accumulate_grad_batches,
            device=device,
        )

        self.mse_coeff = mse_coeff
        self.kl_coeff = kl_coeff

    def forward(self, x):
        y, _, _ = self.model(x)
        return y

    def step(self, batch, batch_idx):
        x, y = batch

        y, p, q = self.model(x)

        recon_loss, kl_loss, loss = vae_loss(x, y, q, p, self.mse_coeff, self.kl_coeff)

        logs = {
            "metrics/recon_loss": recon_loss,
            "metrics/kl_loss": kl_loss,
            "metrics/loss": loss,
        }
        return loss, logs

    def validation_log(self, batch, batch_idx, loss, logs):
        x = batch[0][0:5]
        y = self.forward(x)

        self.logger.update_hyperparams(
            {}, {"recon_loss_metric": logs["metrics/recon_loss"].unsqueeze(0)}
        )

        self.logger.add_images("image/original", x, self.global_step)
        self.logger.add_images("image/reconstructed", y, self.global_step)
        self.logger.log_dict({f"val/{k}": v for k, v in logs.items()}, self.global_step)
