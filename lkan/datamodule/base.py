import torch
from torch.utils.data import random_split


class BaseDataModule:
    def __init__(self, batch_size: int, split_ratio: float):
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def split_dataset(self, dataset):
        train_size = int(self.split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
