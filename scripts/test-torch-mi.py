import torch
import argparse
import time

def run_benchmark(M, K, N, steps, warmup_steps):
    # 1. 检查设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Running on GPU: {device_name}")
        
        # 检查 GPU 是否支持 BF16 (Ampere 架构及以上，如 RTX 3090, A100, H100)
        if not torch.cuda.is_bf16_supported():
            print("Warning: Your GPU does not natively support BF16 acceleration. It may fall back to FP32.")
    else:
        device = torch.device("cpu")
        print("Running on CPU (Warning: CPU BF16 performance might be slow)")

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):

        # 2. 准备数据
        print(f"Preparing tensors: [{M}, {K}] @ [{K}, {N}] in bfloat16...")
        try:
            # 创建随机 Tensor 并转为 bf16
            a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
        except RuntimeError as e:
            print(f"Error allocating memory: {e}")
            return

        # 3. 预热 (Warm-up)
        # GPU 需要预热以达到最佳时钟频率，并初始化 CUDA context
        print(f"Warming up for {warmup_steps} steps...")
        for _ in range(warmup_steps):
            c = torch.matmul(a, b)
        
        # 确保预热完成
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 4. 正式测试
        print(f"Benchmarking for {steps} steps...")
        
        if device.type == 'cuda':
            # 使用 CUDA 事件进行更精确的计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(steps):
                c = torch.matmul(a, b)
            end_event.record()
            
            # 等待 GPU 完成所有任务
            torch.cuda.synchronize()
            
            # 计算时间 (毫秒)
            total_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = total_time_ms / steps
            avg_time_s = avg_time_ms / 1000.0
            
        else:
            # CPU 计时
            start_time = time.perf_counter()
            for _ in range(steps):
                c = torch.matmul(a, b)
            end_time = time.perf_counter()
            
            avg_time_s = (end_time - start_time) / steps
            avg_time_ms = avg_time_s * 1000.0

    # 5. 计算 TFLOPS
    # 矩阵乘法 FLOPs = 2 * M * K * N
    flops = 2 * M * K * N
    tflops = (flops / avg_time_s) / 1e12
    bandwidth = (M * K + K * N + M * N) * 2 / (avg_time_s * 1e9)  # GB/sz

    # 6. 输出结果
    print("-" * 40)
    print(f"Results for Matrix Size: {M}x{K} x {K}x{N}")
    print(f"Avg Latency: {avg_time_ms:.4f} ms")
    print(f"Throughput:  {tflops:.4f} TFLOPS")
    print(f"Memory Bandwidth: {bandwidth:.4f} GB/s")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch BF16 Matmul Benchmark")
    
    # 定义命令行参数
    parser.add_argument("--M", type=int, default=4096, help="Rows of Matrix A")
    parser.add_argument("--K", type=int, default=4096, help="Cols of A / Rows of B")
    parser.add_argument("--N", type=int, default=4096, help="Cols of Matrix B")
    parser.add_argument("--steps", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    run_benchmark(args.M, args.K, args.N, args.steps, args.warmup)