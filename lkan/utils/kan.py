import torch


def b_splines(x, grid, k):
    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    for p in range(1, k + 1):
        value = (x - grid[:, : -(p + 1)]) / (
            grid[:, p:-1] - grid[:, : -(p + 1)]
        ) * value[:, :-1] + (grid[:, p + 1 :] - x) / (
            grid[:, p + 1 :] - grid[:, 1:(-p)]
        ) * value[
            :, 1:
        ]

    return value


def coeff2curve(x_eval, grid, coeff, k):
    return (coeff.unsqueeze(1) @ b_splines(x_eval, grid, k)).squeeze()


def curve2coeff(x_eval, y_eval, grid, k):
    value = torch.linalg.lstsq(
        b_splines(x_eval, grid, k).permute(0, 2, 1), y_eval.unsqueeze(dim=2)
    ).solution[:, :, 0]

    return value
