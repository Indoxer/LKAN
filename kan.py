import torch

from lkan.models import KAN
from tmp.pykan.kan import KAN as KAN_real
from tmp.pykan.kan import create_dataset

# testing the KAN model

# if __name__ == "__main__":
#     layers_dims = [4, 3, 2]

#     real_model = KAN_real(layers_dims, grid=5, k=3, seed=0)
#     model = KAN(layers_dims)

#     for layer, real_layer in zip(model.layers, real_model.act_fun):
#         layer.coeff.data = real_layer.coef.data
#         layer.scale_base.data = real_layer.scale_base.data

#     x_update = torch.rand(3, 4)
#     real_model.update_grid_from_samples(x_update)
#     model.update_grid(x_update)

#     inp = torch.rand(3, 4)
#     out = model.forward(inp)
#     real_out = real_model.forward(inp)
#     print(f"my: {out.detach().numpy()}")
#     print(f"real: {real_out.detach().numpy()}")
#     print(f"max diff: {torch.max(torch.abs(out - real_out))}")


if __name__ == "__main__":
    f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2)

    model = KAN([120, 30, 20])
