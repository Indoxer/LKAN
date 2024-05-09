import numpy as np
import torch
import torch.nn.functional as F


class KANLinearFFT(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=1.0,
        base_fun=torch.nn.SiLU(),
        bias=False,
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_fun = base_fun
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        k = torch.arange(1, self.grid_size + 1, device=device).view(
            1, 1, self.grid_size
        )
        self.register_buffer("k", k)

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        out_dim,
                        in_dim,
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=sp_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        self.coeff = torch.nn.Parameter(
            torch.rand(out_dim, in_dim, 2, grid_size, device=device)
            * noise_scale
            / (np.sqrt(in_dim) * np.sqrt(grid_size)),
        )  # [out_dim, in_dim,2, grid_size]

        self.scale_base = torch.nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=sb_trainable,
        )

        if bias is True:
            self.bias = torch.nn.Parameter(
                torch.rand(out_dim), requires_grad=bias_trainable
            )
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)
        # x [batch, in_dim]

        # [batch, in_dim, 2*grid_size]

        splines = x.view(*x.shape, 1).expand(-1, -1, 2 * self.grid_size)

        splines_cos = torch.cos(splines[:, :, : self.grid_size] * self.k)
        splines_sin = torch.sin(splines[:, :, self.grid_size :] * self.k)

        splines = torch.cat([splines_cos, splines_sin], dim=-1)

        ####### Efficient KAN forward #########

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)
        # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

        y_spline = F.linear(
            splines.view(batch_size, -1),
            (self.coeff * self.scale_spline.unsqueeze(-1).unsqueeze(-1)).view(
                self.out_dim, -1
            ),
        )  # [batch_size, in_dim * grid_size * 2] @ [out_dim, in_dim * grid_size * 2]^T = [batch, out_dim]

        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias

        #######################################################################

        y = y.view(*shape, self.out_dim)

        return y

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.coeff.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
