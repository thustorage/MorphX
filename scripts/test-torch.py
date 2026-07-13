import torch
import torch.nn as nn
import time
from smsched_api import fix_sm_for_stream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
iter = 10
e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)

# fix_sm_for_stream(s0, 0, 32)

with torch.cuda.stream(s0):
    x = torch.full((4096, 4096), 0.3).bfloat16().to(device)
    w = torch.full((4096, 4096), 3.0).bfloat16().to(device)
    for _ in range(5):
        _ = nn.functional.linear(x, w)
    e0.record(s0)
    for _ in range(iter):
        ret = nn.functional.linear(x, w)
    e1.record(s0)
    e1.synchronize()
    print(ret)
    print(f"elapsed time: {e0.elapsed_time(e1) / iter * 1000} us")

    # x1 = torch.full((16, 32768), 0.3).bfloat16().to(device)
    # w1 = torch.full((32768, 32768), 3.0).bfloat16().to(device)
    # ret1 = nn.functional.linear(x1, w1)
    # print(ret1)
    # s0.synchronize()

    x2 = torch.full((2, 8192), 0.3).bfloat16().to(device)
    w2 = torch.full((8192, 8192), 3.0).bfloat16().to(device)
    ret2 = nn.functional.linear(x2, w2)
    s0.synchronize()
    print(ret2)

# --- Spawn two threads, each using a different CUDA stream to run 10 matmuls and record time ---
import threading

def worker(stream, n_iter, tid, start_barrier):
    # Each thread creates its own tensors on the device while using the provided stream.
    with torch.cuda.stream(stream):
        xs = torch.full((4096, 4096), 0.3, dtype=torch.bfloat16, device=device)
        ws = torch.full((4096, 4096), 3.0, dtype=torch.bfloat16, device=device)

        # warm-up to avoid including kernel launch/setup overhead
        for _ in range(2):
            _ = nn.functional.linear(xs, ws)

        # ensure warm-up kernels have finished on this stream before synchronizing threads
        stream.synchronize()

        # wait until all threads have finished warm-up
        start_barrier.wait()

        # create timing events after barrier so all threads start timing together
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        e_start.record(stream)
        for _ in range(n_iter):
            # perform matrix multiply
            out = torch.matmul(xs, ws)
        e_end.record(stream)

    # synchronize on the end event to ensure timing is finished
    e_end.synchronize()
    elapsed_ms = e_start.elapsed_time(e_end)
    avg_us = elapsed_ms / n_iter * 1000.0
    print(f"Thread {tid} (stream={stream}): avg per matmul = {avg_us:.2f} us", flush=True)


if __name__ == '__main__':
    n_iter = 10
    threads = []
    # barrier to synchronize threads after warm-up so timing starts together
    start_barrier = threading.Barrier(2)
    for idx, stream in enumerate([s0, s1]):
        t = threading.Thread(target=worker, args=(stream, n_iter, idx, start_barrier))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


