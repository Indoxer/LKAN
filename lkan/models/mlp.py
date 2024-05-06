from typing import List

import torch


class MLP(torch.nn.Module):
    def __init__(self, layers_dims: List[int], device: str) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for in_dim, out_dim in zip(layers_dims[:-1], layers_dims[1:-1]):
            self.layers += [torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU()]

        self.layers.append(torch.nn.Linear(layers_dims[-2], layers_dims[-1]))

        self.to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
