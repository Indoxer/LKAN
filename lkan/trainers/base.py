import itertools

import torch
from tqdm import tqdm

from lkan.loggers import CustomLogger


class BaseTrainer:
    def __init__(
        self,
        model,
        lr: float,
        logger: CustomLogger,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        lr_scheduler_params: dict,
        lr_step: str,
        clip_grad_norm: float,
        accumulate_grad_batches: int,
        device: str,
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.lr = lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = lr_scheduler(optimizer=self.opt, **lr_scheduler_params)
        self.lr_step = lr_step
        self.clip_grad_norm = clip_grad_norm
        self.logger = logger
        self.accumulate_grad_batches = accumulate_grad_batches

    def forward(self, x):
        raise NotImplementedError

    def step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, logs = self.step(batch, batch_idx)

        self.opt.zero_grad()
        loss.backward()
        if self.global_step % self.accumulate_grad_batches == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.opt.step()

        logs["lr"] = self.opt.param_groups[0]["lr"]
        logs = {f"train/{k}": v for k, v in logs.items()}

        self.logger.log_dict(
            {**logs},
            self.global_step,
        )

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx)

            self.validation_log(batch, batch_idx, loss, logs)

        self.model.train()

    def validation_log(self, batch, batch_idx, loss, logs):
        raise NotImplementedError

    def fit(
        self,
        max_epochs: int,
        max_steps: int,
        validation_every_n_batches: int,
        save_every_n_steps: int,
        datamodule,
    ):
        self.global_step = 0
        stop = False
        validation_dataloader = iter(itertools.cycle(datamodule.val_dataloader()))
        for epoch in range(max_epochs):
            print(f"Epoch: {epoch} / {max_epochs}")
            print("Training...")
            for batch_idx, batch in enumerate(tqdm(datamodule.train_dataloader())):
                for i, el in enumerate(batch):
                    if isinstance(el, torch.Tensor):
                        batch[i] = el.to(self.device)

                self.training_step(batch, batch_idx)
                self.global_step += 1

                if self.lr_step != "epoch" and self.global_step % self.lr_step == 0:
                    self.lr_scheduler.step()

                if self.global_step > max_steps:
                    stop = True
                    break

                if self.global_step % save_every_n_steps == 0:
                    self.logger.save_model(self.model, self.global_step)

                if batch_idx % validation_every_n_batches == 0:
                    batch = next(validation_dataloader)
                    for i, el in enumerate(batch):
                        if isinstance(el, torch.Tensor):
                            batch[i] = el.to(self.device)
                    self.validation_step(batch, batch_idx)

            if self.lr_step == "epoch":
                self.lr_scheduler.step()

            if stop:
                break
