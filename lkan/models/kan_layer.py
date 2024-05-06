import numpy as np
import torch

from lkan.utils.kan import coeff2curve, curve2coeff


class KANLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_number=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,  # Not used now
        grid_range=[-1, +1],
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        device="cpu",
    ):

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_number = grid_number
        self.device = device

        step = (grid_range[1] - grid_range[0]) / grid_number
        grid = (
            torch.arange(grid_number + 2 * k + 1, device=device) * step
            + grid_range[0]
            - step * k
        )
        grid = grid.repeat(self.size, 1)
        self.register_buffer("grid", grid)

        weight_sharing = torch.arange(self.size, device=device)
        self.register_buffer("weight_sharing", weight_sharing)

        self.coeff = torch.nn.Parameter(
            curve2coeff(
                self.grid,
                (
                    (torch.rand(self.size, self.grid.shape[1], device=device) - 1 / 2)
                    * noise_scale
                    / grid_number
                ),
                self.grid,
                k,
            )
        )
        self.mask = torch.nn.Parameter(torch.ones(self.size, device=device))
        self.scale_base = torch.nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(in_dim * out_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=sb_trainable,
        )
        self.scale_spline = torch.nn.Parameter(
            torch.full((self.size,), fill_value=scale_spline, device=device),
            requires_grad=sp_trainable,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(out_dim, device=device), requires_grad=bias_trainable
        )

    # TODO: Refactor this function
    @torch.no_grad()
    def update_grid(self, x, margin=0.01):

        batch_size = x.shape[0]
        batch = x.shape[0]

        x = x.repeat(1, self.out_dim).reshape(batch_size, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coeff2curve(x_pos, self.grid, self.coeff, self.k)
        ids = [int(batch / self.grid_number * i) for i in range(self.grid_number)] + [
            -1
        ]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat(
            [
                grid_adaptive[:, [0]]
                - margin
                + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a
                for a in np.linspace(0, 1, num=self.grid_number + 1)
            ],
            dim=1,
        )
        self.grid.data = (
            self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        )

        h = (self.grid[:, [-1]] - self.grid[:, [0]]) / (self.grid.shape[1] - 1)

        for i in range(self.k):
            self.grid = torch.cat([self.grid[:, [0]] - h, self.grid], dim=1)
            self.grid = torch.cat([self.grid, self.grid[:, [-1]] + h], dim=1)

        self.coeff.data = curve2coeff(x_pos, y_eval, self.grid, self.k)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.repeat(1, self.out_dim).permute(1, 0)

        y = coeff2curve(
            x_eval=x,
            grid=self.grid[self.weight_sharing],
            coeff=self.coeff[self.weight_sharing],
            k=self.k,
        ).permute(1, 0)

        y = (
            self.scale_base.unsqueeze(dim=0) * self.base_fun(x).permute(1, 0)
            + self.scale_spline.unsqueeze(dim=0) * y
        )

        y = self.mask[None, :] * y

        y = torch.sum(y.reshape(batch_size, self.out_dim, self.in_dim), dim=2)

        return y + self.bias
