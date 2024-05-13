import math

import numpy as np
import torch
from torch.nn import functional as F

from lkan.utils.kan import conv2d_efficient_fftkan

from .kan_linear import KANLinear
from .kan_linear_fft import KANLinearFFT


class KANConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        grid_size=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=1.0,
        base_fun=torch.nn.SiLU(),
        scale_spline_trainable=True,
        scale_base_trainable=True,
        chunk_size=None,
        device="cpu",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "kernel_size must be an integer or a tuple of two integers"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.grid_size = grid_size
        self.base_fun = base_fun
        self.device = device
        self.chunk_size = chunk_size

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        self.out_channels,
                        self.in_channels // self.groups,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=scale_spline_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        self.coeff = torch.nn.Parameter(
            torch.rand(
                2,
                self.out_channels,
                (self.in_channels // self.groups),
                self.kernel_size[0],
                self.kernel_size[1],
                grid_size,
                device=device,
            )
            * noise_scale
            / (np.sqrt(kernel_size**2) * np.sqrt(grid_size)),
        )  # [2, out_channels, kernel_size**2, grid_size]

        self.scale_base = torch.nn.Parameter(
            (
                1 / (kernel_size**2**0.5)
                + (
                    torch.randn(
                        self.out_channels,
                        self.in_channels // self.groups,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        device=device,
                    )
                    * 2
                    - 1
                )
                * noise_scale_base
            ),
            requires_grad=scale_base_trainable,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_channels, device=device), requires_grad=True
            )
        else:
            self.bias = bias

        self.conv_fftkan = conv2d_efficient_fftkan

    def forward(self, x):
        x = self.conv_fftkan(
            x,
            self.scale_base,
            self.scale_spline,
            self.coeff,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x
