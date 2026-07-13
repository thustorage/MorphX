import torch
from torch import nn
import flashinfer
import time
import os
from smsched_api import fix_sm_for_stream

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
# torch.manual_seed(0)

num_qo_heads = 32
num_kv_heads = 32
head_dim = 128
batch_size = 8
pages_per_req = 128
max_num_pages = pages_per_req * batch_size
page_size = 16
# allocate 128MB workspace buffer
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
event_start = torch.cuda.Event(enable_timing=True)
event_ci_end = torch.cuda.Event(enable_timing=True)
event_mi_end = torch.cuda.Event(enable_timing=True)
n_ci = 20
n_mi = 20
with torch.cuda.stream(s0):
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    kv_page_indices = torch.arange(max_num_pages).int().to("cuda")
    kv_page_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * pages_per_req
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device="cuda")
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * pages_per_req
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16, 
        kv_data_type=torch.bfloat16
    )
    prefill_wrapper.plan(
        qo_indptr, 
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE", 
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16
    )

    outputs = []
    decode_q = torch.empty(batch_size, num_qo_heads, head_dim).bfloat16().to("cuda")
    prefill_q = torch.empty(qo_indptr[-1], num_qo_heads, head_dim).bfloat16().to("cuda")
    kv_cache = torch.empty(max_num_pages, 2, page_size, num_kv_heads, head_dim).bfloat16().to("cuda")

    for _ in range(3):
        o = decode_wrapper.run(decode_q, kv_cache)
    for _ in range(3):
        o = prefill_wrapper.run(prefill_q, kv_cache)

with torch.cuda.stream(s0):
    x_ci = torch.full((4096, 4096), 0.3).bfloat16().to("cuda")
    w_ci = torch.full((4096, 4096), 3.0).bfloat16().to("cuda")

    x_mi = torch.full((4, 16384), 0.3).bfloat16().to("cuda")
    w_mi = torch.full((16384, 16384), 3.0).bfloat16().to("cuda")
    for i in range(n_ci):
        _ = nn.functional.linear(x_ci, w_ci)
    for i in range(n_mi):
        _ = nn.functional.linear(x_mi, w_mi)

print("Warm up done", flush=True)

s0.synchronize()
event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n_ci):
        o = prefill_wrapper.run(prefill_q, kv_cache)
event_ci_end.record(s0)
event_ci_end.synchronize()
CI_standalone_full = n_ci / event_start.elapsed_time(event_ci_end) * 1000
event_start.record(s0)
with torch.cuda.stream(s0):
    for i in range(n_mi):
        o = decode_wrapper.run(decode_q, kv_cache)
event_mi_end.record(s0)
event_mi_end.synchronize()
MI_standalone_full = n_mi / event_start.elapsed_time(event_mi_end) * 1000


CI_standalone = []
MI_standalone = []
CI_co_locate = []
MI_co_locate = []

CI_standalone.append([0, 0])
MI_standalone.append([0, 0])
CI_co_locate.append([0, 0])
MI_co_locate.append([0, 0])

for div in range(4, 108, 4): 
    fix_sm_for_stream(s0, 0, div)
    fix_sm_for_stream(s1, div, 108)
    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_ci):
            o = prefill_wrapper.run(prefill_q, kv_cache)
    event_ci_end.record(s0)
    event_ci_end.synchronize()
    CI_standalone.append([div, n_ci / event_start.elapsed_time(event_ci_end) * 1000])

    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_mi):
            o = decode_wrapper.run(decode_q, kv_cache)
    event_mi_end.record(s0)
    event_mi_end.synchronize()
    MI_standalone.append([div, n_mi / event_start.elapsed_time(event_mi_end) * 1000])

    with torch.cuda.stream(s1):
        for i in range(100):
            o = decode_wrapper.run(decode_q, kv_cache)
    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_ci):
            o = prefill_wrapper.run(prefill_q, kv_cache)
    event_ci_end.record(s0)
    event_ci_end.synchronize()
    s1.synchronize()
    CI_co_locate.append([div, n_ci / event_start.elapsed_time(event_ci_end) * 1000])

    with torch.cuda.stream(s1):
        for i in range(100):
            o = prefill_wrapper.run(prefill_q, kv_cache)
    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_mi):
            o = decode_wrapper.run(decode_q, kv_cache)
    event_mi_end.record(s0)
    event_mi_end.synchronize()
    s1.synchronize()
    MI_co_locate.append([div, n_mi / event_start.elapsed_time(event_mi_end) * 1000])

CI_standalone.append([108, CI_standalone_full])
MI_standalone.append([108, MI_standalone_full])
CI_co_locate.append([108, CI_standalone_full])
MI_co_locate.append([108, MI_standalone_full])

sum_co_locate = []
for i in range(len(CI_co_locate)):
    for j in range(len(MI_co_locate)):
        if CI_co_locate[i][0] + MI_co_locate[j][0] == 108:
            sum_co_locate.append([CI_co_locate[i][0], CI_co_locate[i][1] / CI_standalone_full + MI_co_locate[j][1] / MI_standalone_full])

print("", flush=True)
print("CI standalone Throughput: ", CI_standalone_full, flush=True)
print("MI standalone Throughput: ", MI_standalone_full, flush=True)
print("CI standalone: ", CI_standalone, flush=True)
print("MI standalone: ", MI_standalone, flush=True)
print("CI co-locate: ", CI_co_locate, flush=True)
print("MI co-locate: ", MI_co_locate, flush=True)
print("co-locate sum: ", sum_co_locate, flush=True)


