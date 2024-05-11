#include <torch/extension.h>

#include "kan.hpp"
#include <iostream>

torch::Tensor fftkan_forward(torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline, torch::Tensor coeff, int batch_size, int in_dim, int out_dim, int grid_size)
{
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);

  if (X.is_cuda())
  {
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);

    return fftkan_cuda_forward(X, scale_base, scale_spline, coeff, batch_size, in_dim, out_dim, grid_size);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
  // return fftkan_cpu_forward(X, scale_base, scale_spline, coeff, batch_size, in_dim, out_dim, grid_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fftkan_backward(torch::Tensor dY, torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline, torch::Tensor coeff, int batch_size, int in_dim, int out_dim, int grid_size)
{
  CHECK_CONTIGUOUS(dY);
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(scale_base);
  CHECK_CONTIGUOUS(scale_spline);
  CHECK_CONTIGUOUS(coeff);

  if (X.is_cuda())
  {
    CHECK_CUDA(dY);
    CHECK_CUDA(X);
    CHECK_CUDA(scale_base);
    CHECK_CUDA(scale_spline);
    CHECK_CUDA(coeff);

    return fftkan_cuda_backward(dY, X, scale_base, scale_spline, coeff, batch_size, in_dim, out_dim, grid_size);
  }
  TORCH_CHECK(false, "CPU version not implemented yet");
  // return fftkan_cpu_backward(dY, X, scale_base, scale_spline, coeff, batch_size, in_dim, out_dim, grid_size);
}
