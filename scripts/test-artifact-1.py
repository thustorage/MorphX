import torch
import torch.nn as nn
import time
import random
from smsched_api import fix_sm_for_stream

# Co-locate 4096x4096x4096 GEMM with 16384x16384x8 GEMM, combined in two streams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)

n = 30
n_ci = n
n_mi = n

workloads = []
workloads_be = []
random.seed(0)
for i in range(n):
    workloads.append(i & 1)
for i in range(100):
    workloads_be.append(i & 1)
print(workloads)

with torch.cuda.stream(s0):
    x_ci = torch.full((n_ci, 4096, 4096), 0.3).bfloat16().to(device)
    w_ci = torch.full((n_ci, 4096, 4096), 3.0).bfloat16().to(device)

    x_mi = torch.full((n_mi, 8, 16384), 0.3).bfloat16().to(device)
    w_mi = torch.full((n_mi, 16384, 16384), 3.0).bfloat16().to(device)
    # Warm up
    for i in range(n_ci):
        _ = nn.functional.linear(x_ci[i], w_ci[i])
    for i in range(n_mi):
        _ = nn.functional.linear(x_mi[i], w_mi[i])
s0.synchronize()
print("Warm up done", flush=True)

event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n):
        if workloads[i] == 0:
            ret_mi = nn.functional.linear(x_mi[i], w_mi[i])
        else:
            ret_ci = nn.functional.linear(x_ci[i], w_ci[i])
event_end.record(s0)
event_end.synchronize()
standalone = n / event_start.elapsed_time(event_end) * 1000

# fix_sm_for_stream(s0, 0, 76)
# fix_sm_for_stream(s1, 76, 108)
with torch.cuda.stream(s1):
    for i in range(100):
        if workloads_be[i] == 0:
            ret_ci = nn.functional.linear(x_ci[i % n_ci], w_ci[i % n_ci])
        else:
            ret_mi = nn.functional.linear(x_mi[i % n_mi], w_mi[i % n_mi])
event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n):
        if workloads[i] == 0:
            ret_mi = nn.functional.linear(x_mi[i], w_mi[i])
        else:
            ret_ci = nn.functional.linear(x_ci[i], w_ci[i])
event_end.record(s0)
event_end.synchronize()
s1.synchronize()
co_locate = n / event_start.elapsed_time(event_end) * 1000

print("standalone Throughput: ", standalone, flush=True)
print("co-locate Throughput: ", co_locate * 2, flush=True)
print("normalized co-locate Throughput: ", co_locate * 2 / standalone, flush=True)