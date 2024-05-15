#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor conv2d_fftkan_cuda_forward(
    torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline,
    torch::Tensor coeff, torch::Tensor bias, int stride, int padding,
    int dilation, int groups, int batch_size, int in_channels,
    std::tuple<int, int> hw, int out_channels, std::tuple<int, int> kernel_size,
    int grid_size) {
    auto Y = torch::empty({batch_size, out_channels, std::get<0>(hw), std::get<1>(hw)}, X.options());

    return Y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
conv2d_fftkan_cuda_backward(torch::Tensor dY, torch::Tensor X,
                            torch::Tensor scale_base,
                            torch::Tensor scale_spline, torch::Tensor coeff,
                            torch::Tensor bias, int stride, int padding,
                            int dilation, int groups, int batch_size,
                            int in_channels, std::tuple<int, int> hw,
                            int out_channels, std::tuple<int, int> kernel_size,
                            int grid_size) {
    auto dX = torch::empty_like(X);
    auto d_scale_base = torch::empty_like(scale_base);
    auto d_scale_spline = torch::empty_like(scale_spline);
    auto d_coeff = torch::empty_like(coeff);
    auto d_bias = torch::empty_like(bias);

    return {dX, d_scale_base, d_scale_spline, d_coeff, d_bias};
}
