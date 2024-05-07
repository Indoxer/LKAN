import math

import torch
import torch.nn.functional as F

from .kan_linear_2 import KANLinear2


class KANConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        device="cpu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.kernels = torch.nn.ModuleList()
        for _ in range(in_channels):
            self.kernels.append(
                KANLinear2(kernel_size * kernel_size, out_channels, device=device)
            )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_channels), requires_grad=True
            )
        else:
            self.bias = bias

        self.unfold = torch.nn.Unfold(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[-3], shape[-2], shape[-1])

        x = (
            self.unfold(x)
            .permute(0, 2, 1)
            .view(x.shape[0], -1, self.in_channels, self.kernel_size**2)
        )

        x = torch.stack(
            [
                kernel(x[:, :, i, :].contiguous())
                for i, kernel in enumerate(self.kernels)
            ],
            dim=2,
        ).sum(dim=2)

        if self.bias is not False:
            x = x + self.bias[None, None, :]

        h = math.floor(
            (shape[-2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )
        w = math.floor(
            (shape[-1] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )

        x = x.permute(0, 2, 1).view(*shape[:-3], self.out_channels, h, w)

        return x
