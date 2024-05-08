import torch
from torch.utils.cpp_extension import load

cudakan = load(name="cudakan", sources=["./extension/kan.cpp"], verbose=True)


class FFTKAN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return cudakan.f_sigmoid(input)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return cudakan.b_sigmoid(grad_output)


y = FFTKAN.apply(torch.randn(2, 3, device="cuda"))
print(y)
