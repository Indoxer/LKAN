import torch
import torch.nn.functional as F


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


# placeholder for cuda version


def fftkan(X, W, S, C, G, I, O):
    """Notation used in the code: (I have equasion that match this notation (easiest to write code in cuda))

    Args:
        X (torch.Tensor): input tensor of shape [batch, in_dim]
        W (torch.Tensor): silu(x) weights of shape [in_dim, out_dim]
        S (torch.Tensor): spline scale of shape [in_dim, out_dim]
        C (torch.Tensor): spline coefficients of shape [in_dim, 2, out_dim, grid_size]
        G (int): grid size
        I (int): in_dim
        O (int): out_dim

    Returns:
        Y: output tensor of shape [batch, out_dim]
    """
    K = torch.arange(1, G + 1, device=X.device).view(1, 1, G)

    base_fun = torch.nn.SiLU()  # TODO: Change in future to be customizable.
    # [batch, in_dim, 2*grid_size]
    splines = X.view(*X.shape, 1).expand(-1, -1, 2 * G)

    splines_cos = torch.cos(splines[:, :, :G] * K)
    splines_sin = torch.sin(splines[:, :, G:] * K)

    # [batch, in_dim, grid_size * 2]
    splines = torch.cat([splines_cos, splines_sin], dim=-1)

    ####### Efficient KAN forward #########

    batch_size = X.shape[0]
    y_b = F.linear(base_fun(X), W)
    # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

    y_spline = F.linear(
        splines.view(batch_size, -1),
        (C * S.unsqueeze(-1)).view(O, -1),
    )  # [batch_size, in_dim * G * 2] @ [out_dim, in_dim * G * 2]^T = [batch, out_dim]

    y = y_b + y_spline

    return y
