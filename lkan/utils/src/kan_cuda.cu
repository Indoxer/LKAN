#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: Optimize kernels memory access

template <typename scalar_t>
__global__ void fftkan_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> W,
    const torch::PackedTensorAccessor32<scalar_t, 2> S,
    const torch::PackedTensorAccessor32<scalar_t, 4> C,
    torch::PackedTensorAccessor32<scalar_t, 2> Y,
    const int B, const int I, const int O, const int G)
{
    // X [B, I]
    // W [O, I]
    // S [O, I]
    // C [O, I, 2, G]
    // -> Y [B, O]
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && o < O)
    {
        scalar_t sum = 0.0f;
        for (int i = 0; i < I; i++)
        {
            scalar_t g_sum = 0.0f;
            for (int g = 0; g < G; g++)
            {
                scalar_t v_sin, v_cos;
                sincos((g + 1) * X[b][i], &v_sin, &v_cos);
                g_sum += C[o][i][0][g] * v_cos;
                g_sum += C[o][i][1][g] * v_sin;
            }
            sum += S[o][i] * g_sum;
            sum += W[o][i] * X[b][i] / (1.0f + expf(-X[b][i]));
        }

        Y[b][o] = sum;
    }
}

template <typename scalar_t>
__global__ void fftkan_cuda_backward_kernel_WSC(
    const torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> S,
    const torch::PackedTensorAccessor32<scalar_t, 4> C,
    const torch::PackedTensorAccessor32<scalar_t, 2> dY,
    torch::PackedTensorAccessor32<scalar_t, 2> dW,
    torch::PackedTensorAccessor32<scalar_t, 2> dS,
    torch::PackedTensorAccessor32<scalar_t, 4> dC,
    const int B, const int I, const int O, const int G)
{
    // dY [B, O]
    // dW [O, I]
    // dS [O, I]
    // dC [O, I, 2, G]

    // X [B, I]
    // S [O, I]
    // C [O, I, 2, G]

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (o < O && i < I)
    {
        scalar_t dS_sum = 0.0f;
        scalar_t dW_sum = 0.0f;
        for (int b = 0; b < B; b++)
        {
            dW_sum += dY[b][o] * X[b][i] / (1.0f + expf(-X[b][i]));
            scalar_t dS_g_sum = 0.0f;
            for (int g = 0; g < G; g++)
            {
                scalar_t v_sin, v_cos;
                sincos((g + 1) * X[b][i], &v_sin, &v_cos);
                dS_g_sum += C[o][i][0][g] * v_cos + C[o][i][1][g] * v_sin;
                dC[o][i][0][g] += dY[b][o] * S[o][i] * v_cos;
                dC[o][i][1][g] += dY[b][o] * S[o][i] * v_sin;
            }
            dS_sum += dY[b][o] * dS_g_sum;
        }
        dS[o][i] = dS_sum;
        dW[o][i] = dW_sum;
    }
}

template <typename scalar_t>
__global__ void fftkan_cuda_backward_kernel_X(
    const torch::PackedTensorAccessor32<scalar_t, 2> X,
    const torch::PackedTensorAccessor32<scalar_t, 2> W,
    const torch::PackedTensorAccessor32<scalar_t, 2> S,
    const torch::PackedTensorAccessor32<scalar_t, 4> C,
    const torch::PackedTensorAccessor32<scalar_t, 2> dY,
    torch::PackedTensorAccessor32<scalar_t, 2> dX,
    const int B, const int I, const int O, const int G)
{
    // dY [B, O]
    // dX [B, I]

    // X [B, I]
    // W [O, I]
    // S [O, I]
    // C [O, I, 2, G]

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && i < I)
    {
        scalar_t sum = 0.0f;
        for (int o = 0; o < O; o++)
        {
            scalar_t g_sum = 0.0f;
            for (int g = 0; g < G; g++)
            {
                scalar_t v_sin, v_cos;
                sincos((g + 1) * X[b][i], &v_sin, &v_cos);
                g_sum -= C[o][i][0][g] * (g + 1) * v_sin;
                g_sum += C[o][i][1][g] * (g + 1) * v_cos;
            }
            sum += dY[o][i] * S[o][i] * g_sum;
            scalar_t sigmoid_x = 1 / (1 + expf(-X[b][i]));
            sum += dY[b][o] * W[o][i] * sigmoid_x * (1.0f + X[b][i] * (1 - sigmoid_x));
        }
        dX[b][i] = sum;
    }
}

torch::Tensor fftkan_cuda_forward(torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int B, int I, int O, int G)
{
    auto Y = torch::zeros({B, O}, X.options());

    const dim3 threads(16, 16);
    const dim3 blocks((B + threads.x - 1) / threads.x, (O + threads.y - 1) / threads.x);

    cudaDeviceSynchronize(); 

    AT_DISPATCH_FLOATING_TYPES(
        X.type(),
        "fftkan_cuda_forward_kernel",
        ([&]
         { fftkan_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
               X.packed_accessor32<scalar_t, 2>(),
               W.packed_accessor32<scalar_t, 2>(),
               S.packed_accessor32<scalar_t, 2>(),
               C.packed_accessor32<scalar_t, 4>(),
               Y.packed_accessor32<scalar_t, 2>(),
               B, I, O, G); }));

    cudaDeviceSynchronize();

    return Y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fftkan_cuda_backward(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int B, int I, int O, int G)
{
    auto dX = torch::zeros_like(X);
    auto dW = torch::zeros_like(W);
    auto dS = torch::zeros_like(S);
    auto dC = torch::zeros_like(C);

    const dim3 threads(16, 16);
    const dim3 blocks((O + threads.x - 1) / threads.x, (I + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(),
        "fftkan_cuda_backward_WSC",
        ([&]
         { fftkan_cuda_backward_kernel_WSC<scalar_t><<<blocks, threads>>>(
               X.packed_accessor32<scalar_t, 2>(),
               S.packed_accessor32<scalar_t, 2>(),
               C.packed_accessor32<scalar_t, 4>(),
               dY.packed_accessor32<scalar_t, 2>(),
               dW.packed_accessor32<scalar_t, 2>(),
               dS.packed_accessor32<scalar_t, 2>(),
               dC.packed_accessor32<scalar_t, 4>(),
               B, I, O, G); }));


    const dim3 threads2(16, 16);
    const dim3 blocks2((B + threads2.x - 1) / threads2.x, (I + threads2.y - 1) / threads2.y);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(),
        "fftkan_cuda_backward_X",
        ([&]
         { fftkan_cuda_backward_kernel_X<scalar_t><<<blocks2, threads2>>>(
               X.packed_accessor32<scalar_t, 2>(),
               W.packed_accessor32<scalar_t, 2>(),
               S.packed_accessor32<scalar_t, 2>(),
               C.packed_accessor32<scalar_t, 4>(),
               dY.packed_accessor32<scalar_t, 2>(),
               dX.packed_accessor32<scalar_t, 2>(),
               B, I, O, G); }));

    cudaDeviceSynchronize();

    return {dX, dW, dS, dC};
}