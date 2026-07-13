import torch
from flash_attn.flash_attn_interface import flash_attn_with_kvcache
import time

# Configuration
batch_size = 32
seq_len = 1024 * 16
num_heads = 32
head_dim = 128
dtype = torch.bfloat16
device = 'cuda'
s0 = torch.cuda.Stream()

with torch.cuda.stream(s0):
    Q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype, device=device)
    K_cache = torch.randn(batch_size, seq_len * 2, num_heads, head_dim, dtype=dtype, device=device)
    V_cache = torch.randn(batch_size, seq_len * 2, num_heads, head_dim, dtype=dtype, device=device)

    for _ in range(3):
        _ = flash_attn_with_kvcache(Q, K_cache, V_cache, cache_seqlens=seq_len)

    torch.cuda.synchronize()
    start = time.time()
    output = flash_attn_with_kvcache(Q, K_cache, V_cache, cache_seqlens=seq_len)
    torch.cuda.synchronize()
    end = time.time()
    time_taken = end - start
    print(f"Computation time: {time_taken * 1000000.0:.6f} us")
