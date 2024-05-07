import torch
import torchvision
from torchvision import transforms

from .base import BaseDataModule


class OwnDataModule(BaseDataModule):
    def __init__(
        self, batch_size: int, split_ratio: float, x: torch.Tensor, y: torch.Tensor
    ):
        super().__init__(batch_size, split_ratio)
        dataset = torch.utils.data.TensorDataset(x, y)
        self.train, self.val = self.split_dataset(dataset)

    def setup(self):
        pass


class OwnDataModule2(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ):
        self.batch_size = batch_size
        self.train = torch.utils.data.TensorDataset(x_train, y_train)
        self.val = torch.utils.data.TensorDataset(x_val, y_val)

    def setup(self):
        pass


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
    def __init__(
        self,
        root: str,
        batch_size: int,
        input_size: int,
        split_ratio: float = 0.8,
        transform=None,
    ):
        super().__init__(batch_size, split_ratio)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(input_size, antialias=True),
                ]
            )
        else:
            self.transform = transform
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
