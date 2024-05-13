import torch

from lkan.utils.kan import efficient_fftkan, fftkan

batch_size = 4
grid_size = 4
in_dim = 4
out_dim = 4

X = torch.rand((batch_size, in_dim), device="cuda", requires_grad=True)
scale_base = torch.rand((out_dim, in_dim), device="cuda", requires_grad=True)
scale_spline = torch.rand((out_dim, in_dim), device="cuda", requires_grad=True)
coeff = torch.rand((2, out_dim, in_dim, grid_size), device="cuda", requires_grad=True)

y1 = efficient_fftkan(X, scale_base, scale_spline, coeff).mean()
y1.backward()

dX = X.grad.clone()
d_scale_base = scale_base.grad.clone()
d_scale_spline = scale_spline.grad.clone()
d_coeff = coeff.grad.clone()

X.grad = None
scale_base.grad = None
scale_spline.grad = None
coeff.grad = None

y2 = fftkan(X, scale_base, scale_spline, coeff).mean()
y2.backward()

print(
    f"max diff: {torch.abs(y1 - y2).max()}",
)

print(f"grad diff X: {torch.abs(dX - X.grad).max()}")
print(f"grad diff scale_base: {torch.abs(d_scale_base - scale_base.grad).max()}")
print(f"grad diff scale_spline: {torch.abs(d_scale_spline - scale_spline.grad).max()}")
print(f"grad diff coeff: {torch.abs(d_coeff - coeff.grad).max()}")
