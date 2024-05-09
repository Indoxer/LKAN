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
    def forward(ctx, X, W, S, C, B, I, O, G):
        return cudakan.fftkan_forward(X, W, S, C, B, I, O, G)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return cudakan.b_sigmoid(grad_output)


B = 1
G = 1
I = 1
O = 2

#   // X [B, I]
#   // W [O, I]
#   // S [O, I]
#   // C [2, O, I, G]
#   // -> Y [B, O]

# X = torch.tensor([[0.5]], device="cuda")
# W = torch.tensor([[1.0], [1.0]], device="cuda")
# S = torch.tensor([[0.0], [0.0]], device="cuda")
# C = torch.tensor([[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]], device="cuda")

X = torch.randn(B, I, device="cuda")
W = torch.randn(O, I, device="cuda") * 0.0
S = torch.randn(O, I, device="cuda")
C = torch.randn(2, O, I, G, device="cuda")

print(X, W, S, C)

y = FFTKAN.apply(X, W, S, C, B, I, O, G)
# y2 = FFTKAN.apply(X, W, S, C, B, G, I, O)

y2 = fftkan(X, W, S, C, B, I, O, G)

print("out: ", y, y2)

print(torch.abs(y - y2).mean().item())
