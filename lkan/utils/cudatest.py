import numpy as np
import torch
from torch.utils.cpp_extension import load

from kan import fftkan

cudakan = load(
    name="cudakan",
    sources=["./extension/kan.cpp", "./extension/kan_cuda.cu"],
    verbose=True,
)


class FFTKAN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, S, C, B, G, I, O):
        return cudakan.fftkan_forward(X, W, S, C, B, G, I, O)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return cudakan.b_sigmoid(grad_output)


B = 200
G = 1
I = 1
O = 1

#   // X [B, I]
#   // W [O, I]
#   // S [O, I]
#   // C [2, O, I, G]
#   // -> Y [B, O]

# X = torch.tensor([[0.5], [1.0]], device="cuda")
# W = torch.tensor([[1.0]], device="cuda")
# S = torch.tensor([[0.0]], device="cuda")
# C = torch.tensor([[[[0.0]]], [[[0.0]]]], device="cuda")

X = torch.randn(B, I, device="cuda")
W = torch.randn(O, I, device="cuda")
S = torch.randn(O, I, device="cuda") * 0.0
C = torch.randn(2, O, I, G, device="cuda") * 0.0

print(X, W, S, C)

y = FFTKAN.apply(X, W, S, C, B, G, I, O)

print(X, W, S, C)

y2 = fftkan(X, W, S, C, B, G, I, O)

print(y, y2)

print(torch.abs(y - y2).mean().item())
