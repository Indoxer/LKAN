#pragma once
#include <torch/extension.h>

torch::Tensor fftkan_cuda_forward(torch::Tensor X, torch::Tensor scale_base,
                                  torch::Tensor scale_spline,
                                  torch::Tensor coeff, int batch_size,
                                  int in_dim, int out_dim, int grid_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fftkan_cuda_backward(torch::Tensor dY, torch::Tensor X,
                     torch::Tensor scale_base, torch::Tensor scale_spline,
                     torch::Tensor coeff, int batch_size, int in_dim,
                     int out_dim, int grid_size);