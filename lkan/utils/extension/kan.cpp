#include <torch/extension.h>

#include <iostream>

torch::Tensor f_sigmoid(torch::Tensor z)
{
    z = torch::exp(-z);
    z = 1 / (1 + z);
    return z;
}

torch::Tensor b_sigmoid(torch::Tensor z)
{
    auto s = f_sigmoid(z);
    return (1 - s) * s;
}
//X, W, S, C, G, I, O


// torch::Tensor fftkan_forward(torch::Tensor X, torch::Tensor W, torch::Tensor S, torch::Tensor C, int G, int I, int O){

// }


// // Defines the operators
// TORCH_LIBRARY(extension_cpp, m) {
//   m.impl_abstract_pystub("extension_cpp.ops");
// //   m.def("f_sigmoid(Tensor z) -> (Tensor)");
// //   m.def("b_sigmoid(Tensor z) -> (Tensor)");
// //   m.def("lltm_forward(Tensor input, Tensor weights, Tensor bias, Tensor old_h, Tensor old_cell) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
// //   m.def("lltm_backward(Tensor grad_h, Tensor grad_cell, Tensor new_cell, Tensor input_gate, Tensor output_gate, Tensor candidate_cell, Tensor X, Tensor gate_weights, Tensor weights) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
// }

// Registers CPU implementations for lltm_forward, lltm_backward
// TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
// {
//     m.impl("f_sigmoid", &f_sigmoid);
//     m.impl("b_sigmoid", &b_sigmoid);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f_sigmoid", &f_sigmoid, "sigmoid forward");
  m.def("b_sigmoid", &f_sigmoid, "sigmoid backward");
}


