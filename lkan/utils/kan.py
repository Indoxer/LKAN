import torch


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

    # [in_dim, batch_size, grid_size + k] @ [in_dim, grid_size + k, out_dim] - [in_dim, batch_size, out_dim] = 0
    value = torch.linalg.lstsq(
        splines,
        y.transpose(0, 1),
    ).solution

    # [in_dim, grid_size + k, out_dim] -> [out_dim, in_dim, grid_size + k]
    value = value.permute(2, 0, 1)

    return value
