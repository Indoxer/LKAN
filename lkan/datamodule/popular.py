import torchvision
from torchvision import transforms

from .base import BaseDataModule


class ImageDataModule(BaseDataModule):
    def __init__(self, root: str, split_ratio: float, batch_size: int, input_size):
        super().__init__(batch_size, split_ratio)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(input_size, antialias=True),
                transforms.RandomCrop(input_size),
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
