from typing import Callable, List

import torch

from .layers.kan_linear import KANLinear
from .layers.kan_linear_2 import KANLinear2
from .layers.kan_linear_fft import KANLinearFFT


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_dims: List[int] = [4, 3, 2],
        grid_size: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        noise_scale_base: float = 0.1,
        scale_spline: float = 1.0,
        base_fun: str | Callable = "silu",
        bias: bool = True,
        grid_eps: float = 1.0,
        grid_range: List[float] = [-1.0, 1.0],
        bias_trainable: bool = True,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        kan_layer_version: str = "fft",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if isinstance(base_fun, str):
            if base_fun == "silu":
                base_fun = torch.nn.SiLU()

        kan_layer_version = str(kan_layer_version)

        for in_dim, out_dim in zip(layers_dims, layers_dims[1:]):
            if kan_layer_version == "2":
                layer = KANLinear2(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=grid_size,
                    k=k,
                    noise_scale=noise_scale,
                    noise_scale_base=noise_scale_base,
                    scale_spline=scale_spline,
                    base_fun=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    bias=bias,
                    sp_trainable=sp_trainable,
                    sb_trainable=sb_trainable,
                    device=device,
                )
            elif kan_layer_version == "1":
                layer = KANLinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=grid_size,
                    k=k,
                    noise_scale=noise_scale,
                    noise_scale_base=noise_scale_base,
                    scale_spline=scale_spline,
                    base_fun=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    bias=bias,
                    bias_trainable=bias_trainable,
                    sp_trainable=sp_trainable,
                    sb_trainable=sb_trainable,
                    device=device,
                )
            elif kan_layer_version == "fft":
                layer = KANLinearFFT(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=grid_size,
                    noise_scale=noise_scale,
                    noise_scale_base=noise_scale_base,
                    scale_spline=scale_spline,
                    base_fun=base_fun,
                    bias=bias,
                    bias_trainable=bias_trainable,
                    sp_trainable=sp_trainable,
                    sb_trainable=sb_trainable,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown kan_layer_version: {kan_layer_version}")
            self.layers.append(layer)

    def update_grid(self, x):
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
