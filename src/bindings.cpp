#include <torch/extension.h>
#include "kan.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fftkan_forward", &fftkan_forward, "FFTKAN forward (CUDA)");
  m.def("fftkan_backward", &fftkan_backward, "FFTKAN backward (CUDA)");
}