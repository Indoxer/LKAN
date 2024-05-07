import itertools

import numpy as np
import torch
from tqdm import tqdm

from lkan.datamodule.base import BaseDataModule
from lkan.loggers import BaseLogger


class BaseTrainer:
    def __init__(
        self,
        model,
        lr: float = 0.01,
        logger: BaseLogger = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_params: dict = None,
        lr_step: str = None,
        clip_grad_norm: float = 0.5,
        accumulate_grad_batches: int = 1,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.lr = lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.opt, **lr_scheduler_params)
        self.lr_step = lr_step
        self.clip_grad_norm = clip_grad_norm
        if logger is None:
            logger = BaseLogger()
        self.logger = logger
        self.logger.setup(self)
        self.accumulate_grad_batches = accumulate_grad_batches

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, logs = self.step(batch, batch_idx)

        self.opt.zero_grad()
        loss.backward()
        if self.global_step % self.accumulate_grad_batches == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.opt.step()

        self.train_log(batch, batch_idx, loss, logs)

    def train_log(self, batch, batch_idx, loss, logs):
        logs["lr"] = self.opt.param_groups[0]["lr"]
        logs = {f"train/{k}": v for k, v in logs.items()}

        self.logger.log_dict({**logs})

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx)

            self.validation_log(batch, batch_idx, loss, logs)

    def validation_log(self, batch, batch_idx, loss, logs):
        self.logger.log_dict({f"val/{k}": v for k, v in logs.items()})

    def fit(
        self,
        datamodule: BaseDataModule,
        max_epochs: int = 10,
        max_steps: int = np.inf,
        validation_every_n_steps: int | str = 100,
        save_every_n_steps: int = np.inf,
    ):
        self.global_step = 0
        self.stop = False

        # For logger access
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.validation_every_n_steps = validation_every_n_steps
        self.save_every_n_steps = save_every_n_steps

        if validation_every_n_steps != "epoch":
            validation_dataloader = iter(itertools.cycle(datamodule.val_dataloader()))
        for epoch in range(max_epochs):
            print(f"Epoch: {epoch} / {max_epochs}")
            print("Training...")
            self.training = True
            self.model.train()
            for batch_idx, batch in enumerate(tqdm(datamodule.train_dataloader())):
                for i, el in enumerate(batch):
                    if isinstance(el, torch.Tensor):
                        batch[i] = el.to(self.device)

                self.training_step(batch, batch_idx)
                self.global_step += 1

                if (
                    self.lr_step not in ("epoch", None)
                    and self.global_step % self.lr_step == 0
                    and self.lr_scheduler is not None
                ):
                    self.lr_scheduler.step()

                if self.global_step > max_steps:
                    self.stop = True
                    break

                if self.global_step % save_every_n_steps == 0:
                    self.logger.save_model(self.model, self.global_step)

                if (
                    validation_every_n_steps != "epoch"
                    and batch_idx % validation_every_n_steps == 0
                ):
                    self.training = False
                    self.model.eval()
                    batch = next(validation_dataloader)
                    for i, el in enumerate(batch):
                        if isinstance(el, torch.Tensor):
                            batch[i] = el.to(self.device)
                    self.validation_step(batch, batch_idx)
                    self.model.train()

            self.training = False
            self.model.eval()
            if validation_every_n_steps == "epoch":

                print("Validating... ")
                for batch_idx, batch in enumerate(tqdm(datamodule.val_dataloader())):
                    for i, el in enumerate(batch):
                        if isinstance(el, torch.Tensor):
                            batch[i] = el.to(self.device)
                    self.validation_step(batch, batch_idx)
            self.model.train()

            if self.lr_step == "epoch" and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.stop:
                self.logger.finalize()
                break
