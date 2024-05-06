import torch

from lkan.models import KANLinear


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear1 = KANLinear(2, 5)
        self.linear2 = KANLinear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model = Model()

y = model.forward(torch.randn(5, 2))
