import os

import numpy as np
import torch
from torch.utils.cpp_extension import load

from kan import fftkan2

path = os.path.dirname(__file__)

sources = [
    os.path.join(path, "src", f)
    for f in os.listdir(os.path.join(path, "src"))
    if f.endswith(".cpp") or f.endswith(".cu")
]

cudakan = load(
    name="cudakan",
    sources=sources,
    verbose=True,
)


class FFTKAN(torch.autograd.Function):
    @staticmethod
    def forward(X, W, S, C, B, I, O, G):
        return cudakan.fftkan_forward(X, W, S, C, B, I, O, G)

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, S, C, B, I, O, G = inputs
        ctx.vars = (B, I, O, G)
        ctx.save_for_backward(X, W, S, C)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dY):
        X, W, S, C = ctx.saved_tensors
        B, I, O, G = ctx.vars
        dX, dW, dS, dC = cudakan.fftkan_backward(dY, X, W, S, C, B, I, O, G)

        return dX, dW, dS, dC, None, None, None, None


B = 5
G = 5
I = 5
O = 5

X = torch.randn(B, I, device="cuda", requires_grad=True)
W = torch.randn(O, I, device="cuda", requires_grad=True)
S = torch.randn(O, I, device="cuda", requires_grad=True)
C = torch.randn(O, I, 2, G, device="cuda", requires_grad=True)

y = FFTKAN.apply(X, W, S, C, B, I, O, G).mean()
y.backward()

g_X = X.grad.clone()
g_W = W.grad.clone()
g_S = S.grad.clone()
g_C = C.grad.clone()

X.grad = None
W.grad = None
S.grad = None
C.grad = None

y2 = fftkan2(X, W, S, C, B, I, O, G).mean()
y2.backward()

print(torch.abs(y - y2).max())
print(torch.abs(g_X - X.grad).max())  # Invalid
print(torch.abs(g_W - W.grad).max())
print(torch.abs(g_S - S.grad).max())
print(torch.abs(g_C - C.grad).max())

print("out: ", y, y2)

print(torch.abs(y - y2).max().item())
