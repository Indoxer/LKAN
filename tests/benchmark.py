import torch

from lkan.utils.kan import efficient_fftkan, fftkan_cuda

B = 200
G = 30
I = 200
O = 200

X = torch.rand((B, I), device="cuda", requires_grad=True)
Y = torch.rand((B, O), device="cuda", requires_grad=True)
W = torch.rand((O, I), device="cuda", requires_grad=True)
S = torch.rand((O, I), device="cuda", requires_grad=True)
C = torch.rand((2, O, I, G), device="cuda", requires_grad=True)


def benchmark(X, Y, layer, name, times, args):
    forwards_t = 0.0
    forwards_mem = 0.0
    backwards_t = 0.0
    backwards_mem = 0.0

    for n in range(times):

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = layer(X, *args)
        end.record()

        torch.cuda.synchronize()

        forwards_t += start.elapsed_time(end) / times
        forwards_mem += torch.cuda.max_memory_allocated() / times

        loss = torch.nn.functional.mse_loss(y, Y)

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()

        backwards_t += start.elapsed_time(end) / times
        backwards_mem += torch.cuda.max_memory_allocated() / times

    print(
        f"""
################# {name} #########################
    forward avg time: {forwards_t:.4G} ms
    backward avg time: {backwards_t:.4G} ms
    forward max memory peak: {forwards_mem/(10**6):.4G} MB
    backward max memory peak: {(backwards_mem/(10**6)):.4G} MB
#################################################
        """
    )


args = (W, S, C, B, I, O, G)

benchmark(X, Y, efficient_fftkan, "efficient kan", 10, args)
benchmark(X, Y, fftkan_cuda, "fftkan cuda", 10, args)
