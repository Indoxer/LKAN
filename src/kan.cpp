#include "kan.hpp"

#include <torch/extension.h>

#include <iostream>

torch::Tensor fftkan_forward(torch::Tensor X, torch::Tensor scale_base,
                             torch::Tensor scale_spline, torch::Tensor coeff,
                             int batch_size, int in_dim, int out_dim,
                             int grid_size) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);

  if (X.is_cuda()) {
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);

    return fftkan_cuda_forward(X, scale_base, scale_spline, coeff, batch_size,
                               in_dim, out_dim, grid_size);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fftkan_backward(torch::Tensor dY, torch::Tensor X, torch::Tensor scale_base,
                torch::Tensor scale_spline, torch::Tensor coeff, int batch_size,
                int in_dim, int out_dim, int grid_size) {
  CHECK_CONTIGUOUS(dY);
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);

  if (X.is_cuda()) {
    CHECK_CUDA(dY);
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);

    return fftkan_cuda_backward(dY, X, scale_base, scale_spline, coeff,
                                batch_size, in_dim, out_dim, grid_size);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
}

torch::Tensor conv2d_fftkan_forward(torch::Tensor X, torch::Tensor scale_base,
                                    torch::Tensor scale_spline,
                                    torch::Tensor coeff, torch::Tensor bias,
                                    int stride, int padding, int dilation,
                                    int groups, int batch_size, int in_channels,
                                    std::tuple<int, int> hw, int out_channels,
                                    std::tuple<int, int> kernel_size,
                                    int grid_size) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);
  CHECK_CONTIGUOUS(bias);

  if (X.is_cuda()) {
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);
    CHECK_CUDA(bias);

    TORCH_CHECK(false, "CUDA version not implemented yet");
    // return conv2d_fftkan_cuda_forward(X, scale_base, scale_spline, coeff,
    // bias, stride, padding, dilation, groups);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
conv2d_fftkan_backward(torch::Tensor dY, torch::Tensor X,
                       torch::Tensor scale_base, torch::Tensor scale_spline,
                       torch::Tensor coeff, torch::Tensor bias, int stride,
                       int padding, int dilation, int groups, int batch_size,
                       int in_channels, std::tuple<int, int> hw,
                       int out_channels, std::tuple<int, int> kernel_size,
                       int grid_size) {
  CHECK_CONTIGUOUS(dY);
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);
  CHECK_CONTIGUOUS(bias);

  if (X.is_cuda()) {
    CHECK_CUDA(dY);
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);
    CHECK_CUDA(bias);
    TORCH_CHECK(false, "CUDA version not implemented yet");
    // return conv2d_fftkan_cuda_backward(dY, X, scale_base, scale_spline,
    // coeff, bias, stride, padding, dilation, groups);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
}