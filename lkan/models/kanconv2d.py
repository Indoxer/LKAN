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
        bias=False,
        device="cpu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.kernel = KANLinear2(
            in_channels * kernel_size * kernel_size, out_channels, device=device
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

        x = self.unfold(x).permute(0, 2, 1).contiguous()

        x = self.kernel(x)
        if self.bias:
            x = x + self.bias[None, :, None]

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
