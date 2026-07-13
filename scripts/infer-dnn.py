import torch
import torchvision
import time
import argparse
import numpy as np
from collections import deque
from torch.amp import autocast

# 参数解析
parser = argparse.ArgumentParser(description="Online Inference Simulation")
parser.add_argument("--rps", type=float, default=50.0, help="Requests per second (Poisson distribution)")
parser.add_argument("--duration", type=float, default=10.0, help="Total duration of simulation in seconds")
parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum batch size for inference")
args = parser.parse_args()

# 检查硬件是否支持 bf16
if not torch.cuda.is_bf16_supported():
    print("Warning: Your GPU does not support BF16 natively.")

# 加载模型
print("Loading model...", flush=True)
model = torchvision.models.resnet50().cuda()
model = model.to(memory_format=torch.channels_last)
model.eval()

# 预热
print("Warming up...", flush=True)
dummy_input = torch.randn(args.max_batch_size, 3, 256, 256).cuda().to(memory_format=torch.channels_last)
for _ in range(3):
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(dummy_input)
torch.cuda.synchronize()

# 生成请求时间戳 (泊松过程)
print(f"Generating requests for {args.duration}s at {args.rps} RPS...", flush=True)
# 泊松过程的到达间隔服从指数分布
# 生成足够多的请求，稍微多一点以防万一
num_requests_estimated = int(args.rps * args.duration * 1.2) + 100
inter_arrival_times = np.random.exponential(1.0 / args.rps, num_requests_estimated)
arrival_times = np.cumsum(inter_arrival_times)
# 截断超过 duration 的请求
arrival_times = arrival_times[arrival_times <= args.duration]
print(f"Total requests generated: {len(arrival_times)}", flush=True)

# 准备输入数据缓存 (避免每次分配显存)
max_input_data = torch.randn(args.max_batch_size, 3, 256, 256).cuda()
max_input_data = max_input_data.to(memory_format=torch.channels_last)

print("Starting online inference simulation...", flush=True)

request_queue = deque()
next_request_idx = 0
completed_requests = 0
inference_count = 0

start_time = time.time()

while True:
    current_time = time.time() - start_time
    
    while next_request_idx < len(arrival_times) and arrival_times[next_request_idx] <= current_time:
        request_queue.append(arrival_times[next_request_idx])
        next_request_idx += 1
    
    if current_time > args.duration and not request_queue:
        break
        
    if request_queue:
        batch_size = min(len(request_queue), args.max_batch_size)
        
        for _ in range(batch_size):
            request_queue.popleft()
        batch_input = max_input_data[:batch_size]
        
        inf_start = time.time()
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(batch_input)
        torch.cuda.synchronize()
        inf_end = time.time()
        
        inf_time = inf_end - inf_start
        completed_requests += batch_size
        inference_count += 1
        
        print(f"Inference {inference_count}: Batch Size: {batch_size}, Time: {inf_time:.4f}s, Pending: {len(request_queue)}", flush=True)
    else:
        # 避免空转占用 CPU
        if args.rps < 1000:
            time.sleep(0.0001) 

total_time = time.time() - start_time
print(f"\nSimulation finished.")
print(f"Total Duration: {total_time:.2f}s")
print(f"Total Inferences: {inference_count}")
print(f"Total Requests Processed: {completed_requests}")
print(f"Average Throughput: {completed_requests / total_time:.2f} req/s")
