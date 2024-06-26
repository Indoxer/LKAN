#pragma once
#include <torch/extension.h>

#include "conv2d.h"
#include "linear.h"

torch::Tensor fftkan_forward(torch::Tensor X, torch::Tensor scale_base,
                             torch::Tensor scale_spline, torch::Tensor coeff,
                             int batch_size, int in_dim, int out_dim,
                             int grid_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fftkan_backward(torch::Tensor dY, torch::Tensor X, torch::Tensor scale_base,
                torch::Tensor scale_spline, torch::Tensor coeff, int batch_size,
                int in_dim, int out_dim, int grid_size);

torch::Tensor conv2d_fftkan_forward(torch::Tensor X, torch::Tensor scale_base,
                                    torch::Tensor scale_spline,
                                    torch::Tensor coeff, torch::Tensor bias,
                                    int stride, int padding, int dilation,
                                    int groups, int batch_size, int in_channels,
                                    std::tuple<int, int> hw, int out_channels,
                                    std::tuple<int, int> kernel_size,
                                    int grid_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
conv2d_fftkan_backward(torch::Tensor dY, torch::Tensor X,
                       torch::Tensor scale_base, torch::Tensor scale_spline,
                       torch::Tensor coeff, torch::Tensor bias, int stride,
                       int padding, int dilation, int groups, int batch_size,
                       int in_channels, std::tuple<int, int> hw,
                       int out_channels, std::tuple<int, int> kernel_size,
                       int grid_size);

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SIZE(x, sizes) \
    TORCH_CHECK(x.sizes() == sizes, #x ".size(0) != " #sizes)

// cuda

// cpu