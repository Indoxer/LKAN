import torch

from lkan.utils.kan import efficient_fftkan, fftkan_cuda

B = 1
G = 1
I = 1
O = 1

X = torch.rand((B, I), device="cuda", requires_grad=True)
W = torch.rand((O, I), device="cuda", requires_grad=True)
S = torch.rand((O, I), device="cuda", requires_grad=True)
C = torch.rand((2, O, I, G), device="cuda", requires_grad=True)

y1 = efficient_fftkan(X, W, S, C, B, G, I, O).mean()
y1.backward()

dX = X.grad.clone()
dW = W.grad.clone()
dS = S.grad.clone()
dC = C.grad.clone()

X.grad = None
W.grad = None
S.grad = None
C.grad = None

y2 = fftkan_cuda(X, W, S, C, B, G, I, O).mean()
y2.backward()

print(
    f"max diff: {torch.abs(y1 - y2).max()}",
)

print(f"grad diff X: {torch.abs(dX - X.grad).max()}")
print(f"grad diff W: {torch.abs(dW - W.grad).max()}")
print(f"grad diff S: {torch.abs(dS - S.grad).max()}")
print(f"grad diff C: {torch.abs(dC - C.grad).max()}")
