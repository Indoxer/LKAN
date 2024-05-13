import math
import os

import kancpp
import torch
import torch.nn.functional as F
from matplotlib import scale
from numpy import shape
from torch.utils.cpp_extension import load


def b_splines(x, grid, k):
    """Generate B-splines.

    Args:
        x (torch.Tensor): [batch_size_size, in_dim]
        grid (torch.Tensor): [in_dim, grid_size + 2*k + 1]
        k (int): degree of B-splines

    Returns:
        (torch.Tenosr): [batch_size, in_dim, grid_size + k]
    """

    x = x.unsqueeze(-1)

    value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    for p in range(1, k + 1):
        value = (x - grid[:, : -(p + 1)]) / (
            grid[:, p:-1] - grid[:, : -(p + 1)]
        ) * value[:, :, :-1] + (grid[:, p + 1 :] - x) / (
            grid[:, p + 1 :] - grid[:, 1:(-p)]
        ) * value[
            :, :, 1:
        ]

    return value


def curve2coeff(x, y, grid, k, eps=1e-8):
    """Calculate coefficients of B-splines.

    Args:
        x_eval (torch.Tensor): [batch_size, in_dim]
        y_eval (torch.Tensor): [batch_size, in_dim, out_dim]
        grid (torch.Tensor): [in_dim, grid_size + 2*k + 1]
        k (int): degree of B-splines

    Returns:
        (torch.Tensor): [out_dim, in_dim, grid_size + k]
    """

    splines = b_splines(x, grid, k).transpose(0, 1)

    device = splines.device
    if device != "cpu":
        splines = splines.cpu()
        y = y.cpu()

    # [in_dim, batch_size, grid_size + k] @ [in_dim, grid_size + k, out_dim] - [in_dim, batch_size, out_dim] = 0
    value = torch.linalg.lstsq(splines, y.transpose(0, 1)).solution

    value = value.to(device)

    # [in_dim, grid_size + k, out_dim] -> [out_dim, in_dim, grid_size + k]
    value = value.permute(2, 0, 1)

    return value


class FFTKAN(torch.autograd.Function):
    @staticmethod
    def forward(
        X,
        scale_base,
        scale_spline,
        coeff,
    ):
        batch_size = X.size(0)
        in_dim = X.size(1)
        out_dim = scale_base.size(0)
        grid_size = coeff.size(3)

        assert list(scale_base.shape) == [out_dim, in_dim]
        assert list(scale_spline.shape) == [out_dim, in_dim]
        assert list(coeff.shape) == [2, out_dim, in_dim, grid_size]

        return kancpp.fftkan_forward(
            X,
            scale_base,
            scale_spline,
            coeff,
            batch_size,
            in_dim,
            out_dim,
            grid_size,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            X,
            scale_base,
            scale_spline,
            coeff,
        ) = inputs
        ctx.vars = (
            X.size(0),
            X.size(1),
            scale_base.size(0),
            coeff.size(3),
        )
        ctx.save_for_backward(X, scale_base, scale_spline, coeff)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dY):
        dX, d_scale_base, d_scale_spline, d_coeff_weight = kancpp.fftkan_backward(
            dY, *ctx.saved_tensors, *ctx.vars
        )

        return dX, d_scale_base, d_scale_spline, d_coeff_weight, None, None, None, None


fftkan = FFTKAN.apply


class Conv2dFFTKAN(torch.autograd.Function):
    def forward(
        X,
        scale_base,
        scale_spline,
        coeff,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        raise NotImplementedError

    def setup_context(ctx, inputs, output):
        (
            X,
            scale_base,
            scale_spline,
            coeff,
            bias,
            stride,
            padding,
            dilation,
            groups,
        ) = inputs

        ctx.vars = (stride, padding, dilation, groups)

        ctx.save_for_backward(X, scale_base, scale_spline, coeff, bias)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dY):
        dX, d_scale_base, d_scale_spline, d_coeff_weight, d_bias = (
            kancpp.conv2d_fftkan_backward(dY, *ctx.saved_tensors, *ctx.vars)
        )

        return (
            dX,
            d_scale_base,
            d_scale_spline,
            d_coeff_weight,
            d_bias,
            None,
            None,
            None,
            None,
        )


def efficient_fftkan(
    X,
    scale_base,
    scale_spline,
    coeff,
):
    in_dim = X.size(-1)
    shape = X.shape[:-1]
    X = X.reshape(-1, in_dim)

    batch_size = X.size(0)
    out_dim = scale_base.size(0)
    grid_size = coeff.size(3)

    assert list(scale_base.shape) == [out_dim, in_dim]
    assert list(scale_spline.shape) == [out_dim, in_dim]
    assert list(coeff.shape) == [2, out_dim, in_dim, grid_size]

    K = torch.arange(1, grid_size + 1, device=X.device).view(1, 1, grid_size)

    base_fun = torch.nn.SiLU()

    splines_cos = torch.cos(X.unsqueeze(-1) * K)
    splines_sin = torch.sin(X.unsqueeze(-1) * K)

    # [batch, in_dim, grid_size * 2]
    splines = torch.cat([splines_cos, splines_sin], dim=-1)

    ####### Efficient KAN forward #########

    batch_size = X.shape[0]
    y_b = F.linear(base_fun(X), scale_base)
    # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

    y_spline = F.linear(
        splines.view(batch_size, -1),
        (coeff.permute(1, 2, 0, 3) * scale_spline.unsqueeze(-1).unsqueeze(-1)).reshape(
            out_dim, -1
        ),
    )  # [batch_size, in_dim * grid_size * 2] @ [out_dim, in_dim * grid_size * 2]^T = [batch, out_dim]

    y = y_b + y_spline

    y = y.view(*shape, out_dim)

    return y.contiguous()


# TODO: How to test this?
def conv2d_efficient_fftkan(
    x, scale_base, scale_spline, coeff, bias, stride=1, padding=0, dilation=1, groups=1
):
    # x [batch_size, in_channels, h, w]
    # kernel_size [height, width]
    # scale_base [out_channels, in_channels/groups, kernel_size[0], kernel_size[1]]
    # scale_spline [out_channels, in_channels/groups, kernel_size[0], kernel_size[1]]
    # coeff [2, out_channels, in_channels/groups, kernel_size[0], kernel_size[1], grid_size]

    shape = x.shape
    batch_size = shape[0]
    in_channels = shape[1]

    grid_size = coeff.shape[-1]

    out_channels = scale_base.shape[0]
    kernel_size = (scale_base.shape[2], scale_base.shape[3])

    base_fun = torch.nn.SiLU()
    K = torch.arange(1, grid_size + 1, device=x.device).view(1, 1, 1, grid_size)

    x = (
        torch.nn.functional.unfold(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        .transpose(-1, -2)
        .reshape(-1, groups, (in_channels // groups) * kernel_size[0] * kernel_size[1])
        .permute(1, 0, 2)
    )

    y_b = (
        base_fun(x)
        @ scale_base.view(groups, out_channels // groups, -1).transpose(-1, -2)
    ).permute(1, 0, 2)

    splines_cos = torch.cos(x.unsqueeze(-1) * K)
    splines_sin = torch.sin(x.unsqueeze(-1) * K)
    splines = torch.cat([splines_cos, splines_sin], dim=-1)

    y_spline = (
        splines.flatten(2)
        @ (coeff.permute(1, 2, 3, 4, 0, 5) * scale_spline.unsqueeze(-1).unsqueeze(-1))
        .reshape(groups, out_channels // groups, -1)
        .transpose(-1, -2)
    ).permute(1, 0, 2)

    y = y_b + y_spline + bias

    y = y.reshape(batch_size, -1, out_channels)

    h = math.floor(
        (shape[-2] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
    )
    w = math.floor(
        (shape[-1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
    )

    y = y.permute(0, 2, 1).view(batch_size, out_channels, h, w)

    return y.contiguous()
