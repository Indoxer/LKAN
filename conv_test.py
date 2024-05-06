import torch

from lkan.models import KANConv2d

layer = KANConv2d(1, 3, 3, padding=0, dilation=2, stride=2)

x = torch.randn(3, 1, 28, 28)

print(layer(x).shape)
