#include <torch/extension.h>

// I am abandoning the CPU implementation for now.

// TODO: Implement MultiThreaded CPU version

// template <typename scalar_t>
// void fftkan_cpu_forward_(torch::TensorAccessor<scalar_t, 2> X, torch::TensorAccessor<scalar_t, 2> scale_base, torch::TensorAccessor<scalar_t, 2> scale_spline, torch::TensorAccessor<scalar_t, 4> coeff, torch::TensorAccessor<scalar_t, 2> Y, int batch_size, int in_dim, int out_dim, int grid_size)
// {

//   for (int b = 0; b < batch_size; b++)
//   {
//     for (int o = 0; o < out_dim; o++)
//     {
//       scalar_t sum = 0.0f;
//       for (int i = 0; i < in_dim; i++)
//       {
//         scalar_t x = X[b][i];
//         scalar_t g_sum = 0.0f;
//         scalar_t sin_0, cos_0;
//         sincos(x, &sin_0, &cos_0); // sincos calculates only for float?
//         scalar_t sin_g = 0.0f, cos_g = 1.0f, tmp_cos;
//         for (int g = 0; g < grid_size; g++)
//         {
//           tmp_cos = cos_g * cos_0 - sin_g * sin_0;
//           sin_g = sin_g * cos_0 + cos_g * sin_0;
//           cos_g = tmp_cos;
//           g_sum += coeff[0][o][i][g] * cos_g;
//           g_sum += coeff[1][o][i][g] * sin_g;
//         }
//         sum += scale_spline[o][i] * g_sum;
//         sum += scale_base[o][i] * x / (1.0f + expf(-x));
//       }

//       Y[b][o] = sum;
//     }
//   }
// }

// // TODO: Implement MultiThreaded CPU version and optimize for CPU
// template <typename scalar_t>
// void fftkan_cpu_backward_(torch::TensorAccessor<scalar_t, 2> dY, torch::TensorAccessor<scalar_t, 2> X, torch::TensorAccessor<scalar_t, 2> scale_base, torch::TensorAccessor<scalar_t, 2> scale_spline, torch::TensorAccessor<scalar_t, 4> coeff, torch::TensorAccessor<scalar_t, 2> dX, torch::TensorAccessor<scalar_t, 2> d_scale_base, torch::TensorAccessor<scalar_t, 2> d_scale_spline, torch::TensorAccessor<scalar_t, 4> d_coeff, int batch_size, int in_dim, int out_dim, int grid_size)
// {
//   scalar_t d_base_sum = 0.0f;

//   for (int i = 0; i < in_dim; i++)
//   {
//     for (int o = 0; o < out_dim; o++)
//     {
//       for (int b = 0; b < batch_size; b++)
//       {
//         scalar_t x = X[b][i];
//         d_base_sum += dY[b][o] * x / (1.0f + expf(-x));
//       }
//       d_scale_base[o][i] = d_base_sum;

//       scalar_t s = scale_spline[o][i];
//       scalar_t d_spline_sum = 0.0f;
//       scalar_t sin_g, cos_g;
//       for (int g = 0; g < grid_size; g++)
//       {
//         scalar_t d_coeff_0_sum = 0.0f;
//         scalar_t d_coeff_1_sum = 0.0f;
//         scalar_t c0 = coeff[0][o][i][g];
//         scalar_t c1 = coeff[1][o][i][g];
//         for (int b = 0; b < batch_size; b++)
//         {
//           scalar_t x = X[b][i];
//           scalar_t dy = dY[b][o];
//           sincos(x * (g + 1), &sin_g, &cos_g); // Unnecessary recalculation of sin and cos, can be optimized
//           d_coeff_0_sum += dy * cos_g;
//           d_coeff_1_sum += dy * sin_g;
//           d_spline_sum += dy * (c0 * cos_g + c1 * sin_g);
//         }
//         d_coeff[0][o][i][g] = s * d_coeff_0_sum;
//         d_coeff[1][o][i][g] = s * d_coeff_1_sum;
//       }
//       d_scale_spline[o][i] = d_spline_sum;
//     }
//   }

//   for (int b = 0; b < batch_size; b++)
//   {
//     for (int i = 0; i < in_dim; i++)
//     {
//       scalar_t sum = 0.0f;
//       scalar_t x = X[b][i];
//       scalar_t sin_0, cos_0;
//       sincos(x, &sin_0, &cos_0);
//       for (int o = 0; o < out_dim; o++)
//       {
//         scalar_t g_sum = 0.0f;
//         scalar_t sin_g = 0.0f, cos_g = 1.0f, tmp_cos;
//         scalar_t dy = dY[b][o];
//         for (int g = 0; g < grid_size; g++)
//         {
//           tmp_cos = cos_g * cos_0 - sin_g * sin_0;
//           sin_g = sin_g * cos_0 + cos_g * sin_0;
//           cos_g = tmp_cos;
//           g_sum -= coeff[0][o][i][g] * (g + 1) * sin_g;
//           g_sum += coeff[1][o][i][g] * (g + 1) * cos_g;
//         }
//         sum += dy * scale_spline[o][i] * g_sum;
//         scalar_t sigmoid_x = 1 / (1 + expf(-x));
//         sum += dy * scale_base[o][i] * sigmoid_x * (1.0f + x * (1 - sigmoid_x));
//       }
//       dX[b][i] = sum;
//     }
//   }

//   return {dX, d_scale_base, d_scale_spline, d_coeff};
// }

// torch::Tensor fftkan_cpu_forward(torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline, torch::Tensor coeff, int batch_size, int in_dim, int out_dim, int grid_size)
// {
//   auto Y = torch::zeros({batch_size, out_dim}, X.options());

//   AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "fftkan_cpu_forward", [&]
//                              { fftkan_cpu_forward_<scalar_t>(X.accessor<scalar_t, 2>(), scale_base.accessor<scalar_t, 2>(), scale_spline.accessor<scalar_t, 2>(), coeff.accessor<scalar_t, 4>(), Y.accessor<scalar_t, 2>(), batch_size, in_dim, out_dim, grid_size); });

//   return Y;
// }

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fftkan_cpu_backward(torch::Tensor dY, torch::Tensor X, torch::Tensor scale_base, torch::Tensor scale_spline, torch::Tensor coeff, int batch_size, int in_dim, int out_dim, int grid_size)
// {
//   auto dX = torch::zeros_like(X);
//   auto d_scale_base = torch::zeros_like(scale_base);
//   auto d_scale_spline = torch::zeros_like(scale_spline);
//   auto d_coeff = torch::zeros_like(coeff);

//   AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "fftkan_cpu_backward", [&]
//                              { fftkan_cpu_backward_<scalar_t>(dY.accessor<scalar_t, 2>(), X.accessor<scalar_t, 2>(), scale_base.accessor<scalar_t, 2>(), scale_spline.accessor<scalar_t, 2>(), coeff.accessor<scalar_t, 4>(), dX.accessor<scalar_t, 2>(), d_scale_base.accessor<scalar_t, 2>(), d_scale_spline.accessor<scalar_t, 2>(), d_coeff.accessor<scalar_t, 4>(), batch_size, in_dim, out_dim, grid_size); });

//   return {dX, d_scale_base, d_scale_spline, d_coeff};
// }
