import torch
from xformers.ops import memory_efficient_attention
import time

batch_size = 32
seq_len = 2048
num_heads = 96
head_dim = 128
dtype = torch.float16
device = 'cuda'

# Create input tensors (Q, K, V)
Q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype, device=device)
K_cache = torch.randn(batch_size, seq_len * 2, num_heads, head_dim, dtype=dtype, device=device)
V_cache = torch.randn(batch_size, seq_len * 2, num_heads, head_dim, dtype=dtype, device=device)

# Warm-up GPU to avoid timing outliers
for _ in range(3):
    _ = memory_efficient_attention(Q, K_cache, V_cache)

# Timing the attention computation
torch.cuda.synchronize()
start = time.time()
output = memory_efficient_attention(Q, K_cache, V_cache)
torch.cuda.synchronize()
end = time.time()
time_taken = end - start
print(f"Computation time: {time_taken * 1000000.0:.6f} us")
