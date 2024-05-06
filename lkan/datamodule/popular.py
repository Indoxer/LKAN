import torch
import torchvision
from torchvision import transforms

from .base import BaseDataModule


class TestDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        split_ratio: float,
        n_var: int = 2,
        f: callable = None,
        n_samples: int = 1000,
    ):
        super().__init__(batch_size, split_ratio)
        if f is None:
            self.f = lambda x: torch.exp(
                torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2
            )
        else:
            self.f = f

        self.n_var = n_var
        self.n_samples = n_samples

    def setup(self):
        x = torch.rand(self.n_samples, self.n_var)
        y = self.f(x)

        dataset = torch.utils.data.TensorDataset(x, y)

        self.train, self.val = self.split_dataset(dataset)


class ImageDataModule(BaseDataModule):
    def __init__(self, root: str, split_ratio: float, batch_size: int, input_size):
        super().__init__(batch_size, split_ratio)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize(input_size, antialias=True),
                # transforms.RandomCrop(input_size),
            ]
        )
        self.root = root


class MNISTDataModule(ImageDataModule):

    def setup(self):
        mnist = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            transform=self.transform,
            download=True,
        )
        self.train, self.val = self.split_dataset(mnist)


class FashionMNISTDataModule(ImageDataModule):
    def setup(self):
        fashion_mnist = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=self.transform,
            download=True,
        )
        self.train, self.val = self.split_dataset(fashion_mnist)


class CIFAR10DataModule(ImageDataModule):
    def setup(self):
        cifar10 = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            transform=self.transform,
            download=True,
        )
        self.train, self.val = self.split_dataset(cifar10)
