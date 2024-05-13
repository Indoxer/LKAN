import math

import torch


def conv2d(x, weights, bias, stride=1, padding=0, dilation=1, groups=1):
    # x [batch_size, in_channels, h, w]
    # kernel_size [height, width]
    # weights [out_channels, in_channels/groups, kernel_size[0], kernel_size[1]]
    # bias [out_channels]

    shape = x.shape
    batch_size = shape[0]
    in_channels = shape[1]

    out_channels = weights.shape[0]
    kernel_size = (weights.shape[2], weights.shape[3])

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

    # y = torch.einsum(
    #     "gbk,gko->gbo",
    #     x,
    #     weights.view(groups, out_channels // groups, -1).permute(0, 2, 1),
    # ).permute(1, 0, 2)

    # [groups, batch_size*patches, (in_channels // groups) * kernel_size[0] * kernel_size[1]]
    # @ [groups, batch_size*patches, out_channels // groups]
    # = [groups, batch_size*patches, out_channels // groups]
    # -> [batch_size*patches, groups, out_channels // groups]
    y = (
        x @ weights.view(groups, out_channels // groups, -1).transpose(-1, -2)
    ).permute(1, 0, 2)

    y = y.reshape(batch_size, -1, out_channels) + bias

    h = math.floor(
        (shape[-2] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
    )
    w = math.floor(
        (shape[-1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
    )

    y = y.permute(0, 2, 1).view(batch_size, out_channels, h, w)

    return y.contiguous()


def conv2d_kan(
    x, scale_base, scale_spline, coeff, stride=1, padding=0, dilation=1, groups=1
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

    y = y_b + y_spline

    y = y.reshape(batch_size, -1, out_channels)

    h = math.floor(
        (shape[-2] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride + 1
    )
    w = math.floor(
        (shape[-1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride + 1
    )

    y = y.permute(0, 2, 1).view(batch_size, out_channels, h, w)

    return y.contiguous()


batch_size = 3
in_channels = 8
out_channels = 6
h = 10
w = 8
kernel_size = (2, 3)
if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
stride = 2
padding = 2
dilation = 1
groups = 2
grid_size = 4

# x = torch.arange(batch_size * in_channels * h * w, dtype=torch.float32).view(
#     batch_size, in_channels, h, w
# )
# weights = torch.ones(
#     out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1],
#     dtype=torch.float32,
# ).view(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
# bias = torch.ones(out_channels, dtype=torch.float32).view(out_channels)

# x = torch.rand(batch_size, in_channels, h, w)
# weights = torch.rand(
#     out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
# )
# bias = torch.rand(out_channels)

# y1 = torch.nn.functional.conv2d(x, weights, bias, stride, padding, dilation, groups)
# y2 = conv2d(x, weights, bias, stride, padding, dilation, groups)
# print(torch.abs(y1 - y2).max())

x = torch.rand(batch_size, in_channels, h, w)
scale_base = torch.rand(
    out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
)
scale_spline = torch.rand(
    out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
)
coeff = torch.rand(
    2, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], grid_size
)

y = conv2d_kan(x, scale_base, scale_spline, coeff, stride, padding, dilation, groups)
