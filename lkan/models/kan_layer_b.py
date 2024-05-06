import numpy as np
import torch
import torch.nn.functional as F

from lkan.utils.kan import b_splines, curve2coeff

from .kan_layer import KANLayer


class KANLayerB(KANLayer):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, +1],
        sb_trainable=True,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-k, grid_size + k + 1, device=device) * step + grid_range[0]
        ).repeat(self.in_dim, 1)
        self.register_buffer("grid", grid)  # grid [in_dim, grid_size + 2*k + 1]

        noise = (
            (torch.rand(grid_size + 1, in_dim, out_dim, device=device) - 1 / 2)
            * noise_scale
            / self.grid_size
        )
        self.coeff = torch.nn.Parameter(
            curve2coeff(
                x=self.grid.T[k:-k],  # [grid_size + 1, in_dim]
                y=noise,  # [grid_size + 1, in_dim, out_dim]
                grid=self.grid,
                k=k,
            ).contiguous()
        )  # [out_dim, in_dim, grid_size + k]

        self.scale_base = torch.nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=sb_trainable,
        )

    def forward(self, x):
        # x [batch, in_dim]

        splines = b_splines(x, self.grid, self.k)  # [batch_size, in_dim, grid_size + k]

        ####### Efficient KAN forward, without mask and scale_spline #########

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)
        # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

        y_spline = F.linear(
            splines.view(batch_size, -1), self.coeff.view(self.out_dim, -1)
        )  # [batch_size, in_dim * (grid_size + k)] @ [out_dim, in_dim * (grid_size + k)]^T = [batch, out_dim]

        y = y_b + y_spline

        #######################################################################

        return y