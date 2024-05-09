#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fftkan_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> X,
    const torch::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> W,
    const torch::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> S,
    const torch::PackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, size_t> C,
    torch::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> Y,
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
                sincos((g+1) * X[b][i], &v_sin, &v_cos);
                g_sum += C[o][i][0][g]*v_cos;
                g_sum += C[o][i][1][g]*v_sin;
            }
            sum += S[o][i]*g_sum;
            sum += W[o][i] * X[b][i] / (1.0f + expf(-X[b][i]));
        }

        Y[b][o] = sum;
  }
}

torch::Tensor fftkan_cuda_forward(torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int B, int I, int O, int G)
{
    auto Y = torch::zeros({B, O}, X.options());

    const dim3 threads(16, 16);
    const dim3 blocks((B + threads.x - 1) / threads.x, (O + threads.y - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(),
        "fftkan_cuda_forward",
        ([&]{ 
            fftkan_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            X.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            W.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            S.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            C.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, size_t>(),
            Y.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
               B, I, O, G); 
        })
    );

    cudaDeviceSynchronize();

    return Y;
}