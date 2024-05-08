#include <torch/extension.h>

#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fftkan_cuda_forward(torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int B, int I, int O, int G);  

torch::Tensor fftkan_forward(torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int B, int I, int O, int G)
{
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    CHECK_INPUT(S);
    CHECK_INPUT(C);

    return fftkan_cuda_forward(X, W, S, C, B, I, O, G);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fftkan_forward", &fftkan_forward, "FFTKAN forward (CUDA)");
}


