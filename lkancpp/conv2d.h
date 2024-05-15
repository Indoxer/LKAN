#pragma once
#include <torch/extension.h>

torch::Tensor conv2d_fftkan_cuda_forward(
    torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline,
    torch::Tensor coeff, torch::Tensor bias, int stride, int padding,
    int dilation, int groups, int batch_size, int in_channels,
    std::tuple<int, int> hw, int out_channels, std::tuple<int, int> kernel_size,
    int grid_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
conv2d_fftkan_cuda_backward(torch::Tensor dY, torch::Tensor X,
                            torch::Tensor scale_base,
                            torch::Tensor scale_spline, torch::Tensor coeff,
                            torch::Tensor bias, int stride, int padding,
                            int dilation, int groups, int batch_size,
                            int in_channels, std::tuple<int, int> hw,
                            int out_channels, std::tuple<int, int> kernel_size,
                            int grid_size);