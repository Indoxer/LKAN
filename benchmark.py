import torch

from lkan.models.layers import KANConv2d

l = KANConv2d(3, 64, 3, device="cuda")
l2 = torch.nn.Conv2d(3, 700, 3, device="cuda")

print("KANConv2d parameters: ", sum(p.numel() for p in l.parameters()))
print("Conv2d parameters: ", sum(p.numel() for p in l2.parameters()))

prof = torch.profiler.profile(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(".logs/conv"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)

prof.start()
for _ in range(20):
    x = torch.ones(100, 3, 50, 50, device="cuda")
    prof.step()
    y = l2(x)
prof.stop()

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=10
    )
)
print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=10
    )
)
prof = torch.profiler.profile(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(".logs/kanconv"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)

prof.start()
for _ in range(20):
    x = torch.ones(100, 3, 50, 50, device="cuda")
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
