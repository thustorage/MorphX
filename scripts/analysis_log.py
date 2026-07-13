import re

def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    kernel_data = {}
    current_waves = []
    
    # First pass: associate kernel names with their AvgWave data
    for i, line in enumerate(lines):
        if '[smsched] kernel' in line:
            kernel_name_match = re.search(r'\[smsched\] kernel (.*): CI:', line)
            if kernel_name_match:
                kernel_name = kernel_name_match.group(1).strip()
                
                # Look for AvgWave lines immediately following
                waves = []
                for j in range(i + 1, len(lines)):
                    if '[smsched] AvgWave' in lines[j]:
                        waves.append(lines[j].strip())
                    else:
                        break
                if waves:
                    kernel_data[kernel_name + '_pk'] = waves

    # Second pass: find "using patched kernel" that appears for the second time and print associated data
    patched_kernel_counts = {}
    processed_kernels = set()
    for line in lines:
        if '[smsched] using patched kernel' in line:
            patched_kernel_match = re.search(r'\[smsched\] using patched kernel (.*)', line)
            if patched_kernel_match:
                patched_kernel_name = patched_kernel_match.group(1).strip()
                
                patched_kernel_counts[patched_kernel_name] = patched_kernel_counts.get(patched_kernel_name, 0) + 1
                
                if patched_kernel_counts[patched_kernel_name] == 2 and patched_kernel_name not in processed_kernels:
                    processed_kernels.add(patched_kernel_name)
                    print(f"Function: {patched_kernel_name}")
                    if patched_kernel_name in kernel_data:
                        for wave in kernel_data[patched_kernel_name]:
                            print(wave)
                    else:
                        print("  No corresponding AvgWave list found.")
                    print("-" * 20)

if __name__ == "__main__":
    parse_log('scripts/logs/LLMEngine.log')
