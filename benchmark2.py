import torch

from lkan.models.layers import KANLinearFFT

l = KANLinearFFT(4000, 10000, device="cuda", cpp=True)

prof = torch.profiler.profile(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(".logs/kanconv"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)

prof.start()
for _ in range(20):
    x = torch.ones(100, 4000, device="cuda")
    prof.step()
    y = l(x)
prof.stop()


print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=8
    )
)
print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=8
    )
)
