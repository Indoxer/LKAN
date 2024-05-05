import enum

import torch

from .kan_layer import KANLayer


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_dims=[4, 3, 2],
        grid_number=5,
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
        device="cpu",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for in_dim, out_dim in zip(layers_dims, layers_dims[1:]):
            self.layers.append(
                KANLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_number=grid_number,
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
            )

    def update_grid(self, x):
        for i, layer in enumerate(self.layers):
            y = x.clone()
            for l in self.layers[:i]:
                y = l(y)
            layer.update_grid(y)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
