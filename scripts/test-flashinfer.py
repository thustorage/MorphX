import torch
from torch import nn
import flashinfer
import time
import os
from smsched_api import fix_sm_for_stream

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
# torch.manual_seed(0)

num_qo_heads = 96
num_kv_heads = 96
head_dim = 128
batch_size = 16
pages_per_req = 1024
max_num_pages = pages_per_req * batch_size
page_size = 16
# allocate 128MB workspace buffer
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)
decode_time = []
prefill_time = []
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
    print("Warm up done", flush=True)

    event_start.record(s0)
    for i in range(10):
        o = decode_wrapper.run(decode_q, kv_cache)
    event_end.record(s0)
    event_end.synchronize()

    event_start.record(s0)
    for i in range(10):
        o = prefill_wrapper.run(prefill_q, kv_cache)
    event_end.record(s0)
    event_end.synchronize()
print('Decode time:', decode_time)
print('Prefill time:', prefill_time)

# s0.synchronize()

# with torch.cuda.stream(s0):
#     for _ in range(3):
#         o = decode_wrapper.run(q, kv_cache)

# with torch.cuda.stream(s1):
#     for _ in range(8):
#         ret = nn.functional.linear(x, w)

# for _ in range(5):
#     _ = decode_wrapper.run(q, kv_cache)
# for _ in range(5):
#     _ = nn.functional.linear(x, w)
# torch.cuda.synchronize()

# s0 = torch.cuda.Stream()
# s1 = torch.cuda.Stream()

# s0_start = time.time()
# with torch.cuda.stream(s0):
#     for i in range(4):
#         o = decode_wrapper.run(q, kv_cache)
# time.sleep(0.002)
# start = time.time()
# with torch.cuda.stream(s1):
#     ret = nn.functional.linear(x, w)
# s1.synchronize()
# end = time.time()
# s0.synchronize()
# s0_end = time.time()
# time_taken = time_taken = end - start
# print(f"Gemm computation time: {time_taken * 1000000.0:.6f} us")
# print(f"attention computation time: {(s0_end - s0_start) * 1000000.0:.6f} us")

# s1_start = time.time()
# with torch.cuda.stream(s1):
#     for i in range(4):
#         ret = nn.functional.linear(x, w)
# time.sleep(0.002)
# start = time.time()
# with torch.cuda.stream(s0):
#     o = decode_wrapper.run(q, kv_cache)
# s0.synchronize()
# end = time.time()
# s1.synchronize()
# s1_end = time.time()
# print(f"Gemm computation time: {(s1_end - s1_start) * 1000000.0:.6f} us")
# print(f"attention computation time: {(end - start) * 1000000.0:.6f} us")

# torch.cuda.synchronize()
# start = time.time()
# for i in range(5):
#     ret = nn.functional.linear(x, w)
# torch.cuda.synchronize()
# end = time.time()
# time_taken = time_taken = end - start
# print(f"Gemm computation time: {time_taken / 5.0 * 1000000.0:.6f} us")

# torch.cuda.synchronize()
# start = time.time()
# o = decode_wrapper.run(q, kv_cache)
# torch.cuda.synchronize()
# end = time.time()
# time_taken = time_taken = end - start
# print(f"attention computation time: {time_taken * 1000000.0:.6f} us")
