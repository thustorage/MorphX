import torch
import torch.nn as nn
import time
from smsched_api import fix_sm_for_stream

# Co-locate 4096x4096x4096 GEMM with 16384x16384x8 GEMM, each in a separate stream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)

n_ci = 30
n_mi = 30

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
    for i in range(n_ci):
        ret_ci = nn.functional.linear(x_ci[i], w_ci[i])
event_end.record(s0)
event_end.synchronize()
CI_standalone = n_ci / event_start.elapsed_time(event_end) * 1000

event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n_mi):
        ret_mi = nn.functional.linear(x_mi[i], w_mi[i])
event_end.record(s0)
event_end.synchronize()
MI_standalone = n_mi / event_start.elapsed_time(event_end) * 1000

fix_sm_for_stream(s0, 0, 54)
fix_sm_for_stream(s1, 54, 108)
with torch.cuda.stream(s1):
    for i in range(100):
        ret_mi = nn.functional.linear(x_mi[i % n_mi], w_mi[i % n_mi])
event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n_ci):
        ret_ci = nn.functional.linear(x_ci[i], w_ci[i])
event_end.record(s0)
event_end.synchronize()
s1.synchronize()
CI_co_locate = n_ci / event_start.elapsed_time(event_end) * 1000

# fix_sm_for_stream(s1, 0, 76)
# fix_sm_for_stream(s0, 76, 108)
with torch.cuda.stream(s1):
    for i in range(100):
        ret_ci = nn.functional.linear(x_ci[i % n_ci], w_ci[i % n_ci])
event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n_mi):
        ret_mi = nn.functional.linear(x_mi[i], w_mi[i])
event_end.record(s0)
event_end.synchronize()
s1.synchronize()
MI_co_locate = n_mi / event_start.elapsed_time(event_end) * 1000

print("CI standalone Throughput: ", CI_standalone, flush=True)
print("MI standalone Throughput: ", MI_standalone, flush=True)
print("CI co-locate Throughput: ", CI_co_locate, flush=True)
print("MI co-locate Throughput: ", MI_co_locate, flush=True)
print("CI col-locate Normalized Throughput: ", CI_co_locate / CI_standalone, flush=True)
print("MI col-locate Normalized Throughput: ", MI_co_locate / MI_standalone, flush=True)