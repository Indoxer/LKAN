#include <torch/extension.h>

#include "kan.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fftkan_forward", &fftkan_forward, "FFTKAN forward");
  m.def("fftkan_backward", &fftkan_backward, "FFTKAN backward");
  m.def("conv2d_fftkan_forward", &conv2d_fftkan_forward,
        "Conv2d FFTKAN forward");
  m.def("conv2d_fftkan_backward", &conv2d_fftkan_backward,
        "Conv2d FFTKAN backward");
}