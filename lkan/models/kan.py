import enum

import torch

from .kan_layer import KANLayer
from .kan_layer_b import KANLayerB


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_dims=[4, 3, 2],
        grid_size=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=1.0,
        grid_range=[-1, 1],
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        kan_layer_version="normal",
        device="cpu",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if isinstance(base_fun, str):
            if base_fun == "silu":
                base_fun = torch.nn.SiLU()

        for in_dim, out_dim in zip(layers_dims, layers_dims[1:]):
            if kan_layer_version == "b":
                layer = KANLayerB(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=grid_size,
                    k=k,
                    noise_scale=noise_scale,
                    noise_scale_base=noise_scale_base,
                    base_fun=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    sb_trainable=sb_trainable,
                    device=device,
                )
            else:
                layer = KANLayer(
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
                    bias_trainable=bias_trainable,
                    sp_trainable=sp_trainable,
                    sb_trainable=sb_trainable,
                    device=device,
                )
            self.layers.append(layer)

    def update_grid(self, x):
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
