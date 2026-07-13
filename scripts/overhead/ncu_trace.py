import sys
import subprocess
import shutil
import re
import os
import argparse

# === 配置 ===
# 我们需要收集的 Metrics (适用于 Ampere, Hopper, Blackwell 等现代架构)
# dram__bytes_read.sum:  DRAM 读取总字节数
# dram__bytes_write.sum: DRAM 写入总字节数
# gpu__time_duration.sum: Kernel 执行耗时 (纳秒)
METRICS = "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
# ============

def check_ncu():
    """检查 ncu 是否可用"""
    if shutil.which("ncu") is None:
        print("Error: 'ncu' (Nsight Compute) command not found.")
        print("Please install NVIDIA Nsight Compute or check your PATH.")
        sys.exit(1)

def build_parser():
    parser = argparse.ArgumentParser(
        description="使用 Nsight Compute 对指定命令进行 DRAM 带宽 profile"
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="需要 profile 的命令及其参数，例如: python single_request.py --workload gemm ...",
    )
    parser.add_argument(
        "--metrics",
        default=METRICS,
        help="需要采集的 metrics 列表（逗号分隔）。默认值适用于大多数场景。",
    )
    return parser


def parse_and_run(command, metrics):
    command_str = " ".join(command)
    print(f"[*] Starting DRAM Bandwidth Trace for: {command_str}")
    print("[*] Note: Execution will be SLOW due to Kernel Replay.")
    print("-" * 100)
    print(f"{'Kernel Name':<40} | {'Dur (ms)':<10} | {'Read (MB)':<10} | {'Write (MB)':<10} | {'BW (GB/s)':<10}")
    print("-" * 100)

    # 构建 ncu 命令
    # --csv: 输出 CSV 格式方便解析
    # --log-file stdout: 将日志打印到标准输出流以便 Python 捕获
    # --target-processes all: 追踪所有子进程
    cmd = [
        "ncu",
        "--csv",
        "--metrics", metrics,
        "--target-processes", "all",
        *command,
    ]

    # 启动进程
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # CSV 解析状态机
    # ncu 的 CSV 格式通常是:
    # "Header", "Metric Name", "Metric Unit", "Metric Value" ...
    # 或者一行包含所有 Metrics，取决于版本。
    # 最稳健的方法是读取 header 映射索引，然后读取数据行。
    print(process.stdout)
    print(process.stderr)
    
    header_map = {}
    csv_mode = False

    metric_index_cache = {}

    def resolve_metric_index(metric_base):
        if metric_base in metric_index_cache:
            return metric_index_cache[metric_base]

        if metric_base in header_map:
            metric_index_cache[metric_base] = header_map[metric_base]
            return header_map[metric_base]

        # 兼容诸如 "metric.name (unit)" 或 "metric.name [unit]" 的列名
        for header, idx in header_map.items():
            normalized = re.split(r"\s*[\[(]", header, maxsplit=1)[0]
            if normalized == metric_base:
                metric_index_cache[metric_base] = idx
                return idx

        metric_index_cache[metric_base] = None
        return None

    def get_val(metric_base):
        idx = resolve_metric_index(metric_base)
        if idx is None or idx >= len(parts):
            return 0.0
        try:
            return float(parts[idx])
        except ValueError:
            return 0.0

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            clean_line = line.strip()
            
            # 过滤掉非 CSV 的普通输出（通常是 ncu 的初始化日志）
            if not csv_mode:
                if "ID," in clean_line and "Kernel Name," in clean_line:
                    csv_mode = True
                    # 解析 Header，找到我们需要的数据在哪一列
                    headers = [h.strip('"') for h in clean_line.split(',')]
                    for idx, h in enumerate(headers):
                        header_map[h] = idx
                    continue
                else:
                    # 打印程序的原始输出，或者是 ncu 的报错
                    # 可以在这里选择是否打印
                    if "Error" in clean_line or "Warning" in clean_line:
                        print(f"[NCU MSG] {clean_line}")
                    continue

            # 处理 CSV 数据行
            if csv_mode:
                # 简单的 CSV 分割 (注意：Kernel Name 可能包含逗号，这里做简单处理)
                # 严谨处理应该用 csv 模块，但为了流式处理简单起见：
                parts = [p.strip('"') for p in clean_line.split(',')]
                
                # 确保行数据完整
                if len(parts) < len(header_map):
                    continue

                try:
                    # 获取数据
                    kernel_name = parts[header_map["Kernel Name"]]

                    read_bytes = get_val("dram__bytes_read.sum")
                    write_bytes = get_val("dram__bytes_write.sum")
                    duration_ns = get_val("gpu__time_duration.sum")

                    # 计算
                    if duration_ns > 0:
                        total_bytes = read_bytes + write_bytes
                        duration_sec = duration_ns / 1e9
                        bw_gbs = (total_bytes / 1e9) / duration_sec
                        
                        read_mb = read_bytes / 1024 / 1024
                        write_mb = write_bytes / 1024 / 1024
                        duration_ms = duration_ns / 1e6
                        
                        # 缩短 Kernel 名字以便显示
                        short_name = kernel_name[:35] + "..." if len(kernel_name) > 35 else kernel_name

                        print(f"{short_name:<40} | {duration_ms:<10.2f} | {read_mb:<10.2f} | {write_mb:<10.2f} | \033[92m{bw_gbs:<10.2f}\033[0m")
                except (ValueError, KeyError):
                    pass

    except KeyboardInterrupt:
        print("\nStopping...")
        process.kill()
    
    process.wait()
    print("-" * 100)
    print("Done.")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.error("请提供需要 profile 的命令，例如: python ncu_trace.py python single_request.py --workload gemm --runs 10")

    check_ncu()
    parse_and_run(args.command, args.metrics)