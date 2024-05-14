#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void fftkan_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> scale_base,
    const torch::PackedTensorAccessor32<scalar_t, 2> scale_spline,
    const torch::PackedTensorAccessor32<scalar_t, 4> coeff,
    torch::PackedTensorAccessor32<scalar_t, 2> Y, const int batch_size,
    const int in_dim, const int out_dim, const int grid_size) {
    // X [batch_size, in_dim]
    // scale_base [out_dim, in_dim]
    // scale_spline [out_dim, in_dim]
    // coeff [2, out_dim, in_dim, grid_size]
    // -> Y [batch_size, out_dim]
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < out_dim) {
        scalar_t sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            scalar_t x = X[b][i];
            scalar_t g_sum = 0.0f;
            scalar_t sin_0, cos_0;
            sincos(x, &sin_0, &cos_0);
            scalar_t sin_g = 0.0f, cos_g = 1.0f, tmp_cos;
            for (int g = 0; g < grid_size; g++) {
                tmp_cos = cos_g * cos_0 - sin_g * sin_0;
                sin_g = sin_g * cos_0 + cos_g * sin_0;
                cos_g = tmp_cos;
                g_sum += coeff[0][o][i][g] * cos_g;
                g_sum += coeff[1][o][i][g] * sin_g;
            }
            sum += scale_spline[o][i] * g_sum;
            sum += scale_base[o][i] * x / (1.0f + expf(-x));
        }

        Y[b][o] = sum;
    }
}

template <typename scalar_t>
__global__ void fftkan_cuda_backward_kernel_WSC(
    const torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> scale_spline,
    const torch::PackedTensorAccessor32<scalar_t, 4> coeff,
    const torch::PackedTensorAccessor32<scalar_t, 2> dY,
    torch::PackedTensorAccessor32<scalar_t, 2> d_scale_base,
    torch::PackedTensorAccessor32<scalar_t, 2> d_scale_spline,
    torch::PackedTensorAccessor32<scalar_t, 4> d_coeff, const int batch_size,
    const int in_dim, const int out_dim, const int grid_size) {
    // dY [batch_size, out_dim]
    // d_scale_base [out_dim, in_dim]
    // d_scale_spline [out_dim, in_dim]
    // d_coeff [2, out_dim, in_dim, grid_size]

    // X [batch_size, in_dim]
    // scale_spline [out_dim, in_dim]
    // coeff [2, out_dim, in_dim, grid_size]

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (o < out_dim && i < in_dim) {
        scalar_t d_base_sum = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            scalar_t x = X[b][i];
            d_base_sum += dY[b][o] * x / (1.0f + expf(-x));
        }
        d_scale_base[o][i] = d_base_sum;

        scalar_t s = scale_spline[o][i];
        scalar_t d_spline_sum = 0.0f;
        scalar_t sin_g, cos_g;
        for (int g = 0; g < grid_size; g++) {
            scalar_t d_coeff_0_sum = 0.0f;
            scalar_t d_coeff_1_sum = 0.0f;
            scalar_t c0 = coeff[0][o][i][g];
            scalar_t c1 = coeff[1][o][i][g];
            for (int b = 0; b < batch_size; b++) {
                scalar_t x = X[b][i];
                scalar_t dy = dY[b][o];
                sincos(x * (g + 1), &sin_g, &cos_g);  // Unnecessary recalculation of
                                                      // sin and cos, can be optimized
                d_coeff_0_sum += dy * cos_g;
                d_coeff_1_sum += dy * sin_g;
                d_spline_sum += dy * (c0 * cos_g + c1 * sin_g);
            }
            d_coeff[0][o][i][g] = s * d_coeff_0_sum;
            d_coeff[1][o][i][g] = s * d_coeff_1_sum;
        }
        d_scale_spline[o][i] = d_spline_sum;
    }
}

// TODO: Backward need to be optimized
template <typename scalar_t>
__global__ void fftkan_cuda_backward_kernel_X(
    const torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> scale_base,
    const torch::PackedTensorAccessor32<scalar_t, 2> scale_spline,
    const torch::PackedTensorAccessor32<scalar_t, 4> coeff,
    const torch::PackedTensorAccessor32<scalar_t, 2> dY,
    torch::PackedTensorAccessor32<scalar_t, 2> dX, const int batch_size,
    const int in_dim, const int out_dim, const int grid_size) {
    // dY [batch_size, out_dim]
    // dX [batch_size, in_dim]

    // X [batch_size, in_dim]
    // scale_base [out_dim, in_dim]
    // scale_spline [out_dim, in_dim]
    // coeff [2, out_dim, in_dim, grid_size]

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && i < in_dim) {
        scalar_t sum = 0.0f;
        scalar_t x = X[b][i];
        scalar_t sin_0, cos_0;
        sincos(x, &sin_0, &cos_0);
        for (int o = 0; o < out_dim; o++) {
            scalar_t g_sum = 0.0f;
            scalar_t sin_g = 0.0f, cos_g = 1.0f, tmp_cos;
            scalar_t dy = dY[b][o];
            for (int g = 0; g < grid_size; g++) {
                tmp_cos = cos_g * cos_0 - sin_g * sin_0;
                sin_g = sin_g * cos_0 + cos_g * sin_0;
                cos_g = tmp_cos;
                g_sum -= coeff[0][o][i][g] * (g + 1) * sin_g;  // High memory access
                g_sum += coeff[1][o][i][g] * (g + 1) * cos_g;  // High memory access
            }
            sum += dy * scale_spline[o][i] * g_sum;
            scalar_t sigmoid_x = 1 / (1 + expf(-x));
            sum += dy * scale_base[o][i] * sigmoid_x * (1.0f + x * (1 - sigmoid_x));
        }
        dX[b][i] = sum;
    }
}

// TODO: Backward need to be optimized
torch::Tensor fftkan_cuda_forward(torch::Tensor X, torch::Tensor scale_base,
                                  torch::Tensor scale_spline,
                                  torch::Tensor coeff, int batch_size,
                                  int in_dim, int out_dim, int grid_size) {
    auto Y = torch::empty({batch_size, out_dim}, X.options());

    const dim3 threads(32, 32);
    const dim3 blocks((batch_size + threads.x - 1) / threads.x,
                      (out_dim + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "fftkan_cuda_forward_kernel", ([&]() {
            fftkan_cuda_forward_kernel<scalar_t>
                <<<blocks, threads>>>(X.packed_accessor32<scalar_t, 2>(),
                                      scale_base.packed_accessor32<scalar_t, 2>(),
                                      scale_spline.packed_accessor32<scalar_t, 2>(),
                                      coeff.packed_accessor32<scalar_t, 4>(),
                                      Y.packed_accessor32<scalar_t, 2>(),
                                      batch_size, in_dim, out_dim, grid_size);
        }));

    cudaDeviceSynchronize();

    return Y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fftkan_cuda_backward(torch::Tensor dY, torch::Tensor X,
                     torch::Tensor scale_base, torch::Tensor scale_spline,
                     torch::Tensor coeff, int batch_size, int in_dim,
                     int out_dim, int grid_size) {
    auto dX = torch::empty_like(X);
    auto d_scale_base = torch::empty_like(scale_base);
    auto d_scale_spline = torch::empty_like(scale_spline);
    auto d_coeff = torch::empty_like(coeff);

    const dim3 threads(16, 16);
    const dim3 blocks((out_dim + threads.x - 1) / threads.x,
                      (in_dim + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "fftkan_cuda_backward_WSC", ([&] {
            fftkan_cuda_backward_kernel_WSC<scalar_t><<<blocks, threads>>>(
                X.packed_accessor32<scalar_t, 2>(),
                scale_spline.packed_accessor32<scalar_t, 2>(),
                coeff.packed_accessor32<scalar_t, 4>(),
                dY.packed_accessor32<scalar_t, 2>(),
                d_scale_base.packed_accessor32<scalar_t, 2>(),
                d_scale_spline.packed_accessor32<scalar_t, 2>(),
                d_coeff.packed_accessor32<scalar_t, 4>(), batch_size, in_dim,
                out_dim, grid_size);
        }));

    const dim3 threads2(16, 16);
    const dim3 blocks2((batch_size + threads2.x - 1) / threads2.x,
                       (in_dim + threads2.y - 1) / threads2.y);

    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "fftkan_cuda_backward_X", ([&]() {
            fftkan_cuda_backward_kernel_X<scalar_t><<<blocks2, threads2>>>(
                X.packed_accessor32<scalar_t, 2>(),
                scale_base.packed_accessor32<scalar_t, 2>(),
                scale_spline.packed_accessor32<scalar_t, 2>(),
                coeff.packed_accessor32<scalar_t, 4>(),
                dY.packed_accessor32<scalar_t, 2>(),
                dX.packed_accessor32<scalar_t, 2>(), batch_size, in_dim, out_dim,
                grid_size);
        }));

    cudaDeviceSynchronize();

    return {dX, d_scale_base, d_scale_spline, d_coeff};
}

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
