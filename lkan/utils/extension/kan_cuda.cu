#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("lltm_forward", &lltm_cuda_forward);
  m.impl("lltm_backward", &lltm_cuda_backward);
}