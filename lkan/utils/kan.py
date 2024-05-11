import math
import os

import kancpp
import torch
import torch.nn.functional as F
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
        (
            X,
            scale_base,
            scale_spline,
            coeff,
        ) = ctx.saved_tensors
        (
            batch_size,
            in_dim,
            out_dim,
            grid_size,
        ) = ctx.vars
        dX, d_scale_base, d_scale_spline, d_coeff_weight = kancpp.fftkan_backward(
            dY,
            X,
            scale_base,
            scale_spline,
            coeff,
            batch_size,
            in_dim,
            out_dim,
            grid_size,
        )

        return dX, d_scale_base, d_scale_spline, d_coeff_weight, None, None, None, None


fftkan = FFTKAN.apply


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

    # [batch, in_dim, 2*grid_size]
    splines = X.view(*X.shape, 1).expand(-1, -1, 2 * grid_size)

    splines_cos = torch.cos(splines[:, :, :grid_size] * K)
    splines_sin = torch.sin(splines[:, :, grid_size:] * K)

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


def conv_efficient_fftkan(
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
    # X = [batch, in_channels, h, w]

    shape = X.shape

    batch_size = shape[0]
    in_channels = shape[1]
    out_channels = scale_base.size(1)

    kernel_size = coeff.shape[3] ** (0.5)

    X = (
        torch.nn.functional.unfold(
            X,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )  # [batch, patches, in_channels * kernel_size**2]
        .permute(0, 2, 1)  # [batch, in_channels * kernel_size**2, patches]
        .view(
            batch_size, -1, in_channels, kernel_size**2
        )  # [batch, patches, in_channels, kernel_size**2]
    ).contiguous()

    X = torch.vmap(efficient_fftkan, (2, 0, 0, 0), 2, chunk_size=4)(
        X,
        scale_base,
        scale_spline,
        coeff,
    ).sum(dim=2)

    h = math.floor(
        (shape[-2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    w = math.floor(
        (shape[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )

    X = X.permute(0, 2, 1).view(*shape[:-3], out_channels, h, w)

    return x
