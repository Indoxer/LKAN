import torch

from lkan.models.layers import KANLinear, KANLinear0, KANLinearFFT
from lkan.utils import b_splines


def plot_linear(
    self,
    layer,
    range=[-1.0, 1.0],
    n_points=100,
    draw_together=[["bias", "spline", "silu"]],
):
    raise NotImplementedError("plot_linear is not implemented yet")
    x = torch.linspace(layer.range[0], range[1], n_points).repeat(1, layer.in_features)

    y_bias = layer.bias

    if isinstance(layer, KANLinear0):
        raise NotImplementedError("KANLinear0 not supported")
    elif isinstance(layer, KANLinearFFT):
        pass
    elif isinstance(layer, KANLinear):
        y_base = layer.scale_base * layer.base_fun(x).unsqueeze(1)
        splines = b_splines(x, layer.grid, layer.k)
        y_splines = (splines.permute(1, 0, 2) @ layer.coeff.permute(1, 2, 0)).permute(
            1, 2, 0
        )  # multiply by scale_spline
    else:
        raise NotImplementedError(f"Unknown layer type: {type(layer).__name__}")

    alias = dict(
        bias=y_bias,
        silu=y_base,
        spline=y_splines,
    )

    for add_together in draw_together:
        y = sum(alias[key] for key in add_together)
        # plot
