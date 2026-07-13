
import re
import sys
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    kernel_times = defaultdict(float)
    kernel_single_times = defaultdict(float)
    kernel_counts = defaultdict(int)
    # 正则表达式匹配形如'kernel finished: %s, %d/%d, stream: 0x%p,elapsed: %lfus'的行
    pattern = re.compile(r'^\[smsched\] kernel finished: (.+?), (\d+)/(\d+), stream: (0x[\da-fA-F]+), elapsed: (\d+\.?\d*)us\s*$')
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                match = pattern.match(line)
                if match:
                    kernel_name = match.group(1)
                    elapsed_time = float(match.group(5))
                    kernel_times[kernel_name] += elapsed_time
                    kernel_counts[kernel_name] += 1
        
        for kernel_name, total_time in kernel_times.items():
            kernel_single_times[kernel_name] = total_time / kernel_counts[kernel_name]
        # 按总时间降序排序
        kernel_times = sorted(kernel_times.items(), key=lambda x: -x[1])
        kernel_single_times = sorted(kernel_single_times.items(), key=lambda x: -x[1])
        # calculate the sum of a list
        all_kernel_time = 0
        for kernel, total_time in kernel_times:
            all_kernel_time += total_time
        
        # 输出结果
        for kernel, total_time in kernel_times:
            print(f"{total_time:10.2f}us {total_time / all_kernel_time:.4f} {kernel}")
        print("\n")
        for kernel, single_time in kernel_single_times:
            print(f"{single_time:10.2f}us {kernel_counts[kernel]} {kernel}")
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()