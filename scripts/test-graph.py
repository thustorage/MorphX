import torch
import emogi_graph as graph
import torch.nn as nn
import smsched_api
import time
import threading

filename = "/home/rtx/gpu/dataset/mawi/mawi.bel"
barrier = threading.Barrier(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
n = 30

pr = graph.PageRank(filename, s0.cuda_stream)
pr.compute(s0.cuda_stream)
pr.join()
s0.synchronize()
with torch.cuda.stream(s1):
    a = torch.full((n, 4096, 4096), 0.3).bfloat16().to(device)
    b = torch.full((n, 4096, 4096), 0.3).bfloat16().to(device)
    for i in range(10):
        _ = nn.functional.linear(a[i], b[i])
s1.synchronize()

smsched_api.fix_sm_for_stream(s0, 0, 108)
smsched_api.fix_sm_for_stream(s1, 0, 108)

# pr.compute(s0.cuda_stream)

t0 = time.time()
e0.record(s1)
with torch.cuda.stream(s1):
    for i in range(n):
        # print(f"Iteration {i} started", flush=True)
        _ = nn.functional.linear(a[i], b[i])
        # print(f"Iteration {i} finished", flush=True)
        # stream.synchronize() 
        # print(f"Launch time: {(time.time() - t0)*1000.0} ms", flush=True)
e1.record(s1)
e1.synchronize()
print(f"Gemm throughput: {n / e0.elapsed_time(e1)}", flush=True)
pr.join()


# with torch.cuda.stream(s1):
#     for i in range(100000):
#         _ = nn.functional.linear(a[i % 30], b[i % 30])
t0 = time.time()
pr.compute(s0.cuda_stream)
pr.join()
t1 = time.time()
print(f"PageRank throughput: {1 / (t1 - t0)}", flush=True)