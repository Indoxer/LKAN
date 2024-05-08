import torch

from lkan.models.layers import KANConv2d

x = torch.rand(20, 3, 100, 100, device="cuda")

l = KANConv2d(3, 64, 3, device="cuda")

with torch.autograd.profiler.profile(with_stack=True, profile_memory=True) as prof:
    l(x)
