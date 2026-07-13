import matplotlib.pyplot as plt
import ast
import sys

if len(sys.argv) < 3:
    print("Usage: python draw.py <input_file> <output_dir>")

file_path = sys.argv[1]
output_dir = sys.argv[2]

# 初始化数据字典
data = {
    "CI standalone": [],
    "MI standalone": [],
    "CI co-locate": [],
    "MI co-locate": [], 
    "co-locate sum": [], 
    "CI co-locate/standalone": [[0, 1]],
    "MI co-locate/standalone": [[0, 1]]
}

# 读取文件并提取数据
throughputs = {}
with open(file_path, "r") as file:
    for line in file:
        if "CI standalone Throughput:" in line:
            throughputs["CI standalone"] = float(line.split(":")[1].strip())
        elif "MI standalone Throughput:" in line:
            throughputs["MI standalone"] = float(line.split(":")[1].strip())
        elif "CI standalone:" in line:
            data["CI standalone"] = ast.literal_eval(line.split(":")[1].strip())
        elif "MI standalone:" in line:
            data["MI standalone"] = ast.literal_eval(line.split(":")[1].strip())
        elif "CI co-locate:" in line:
            data["CI co-locate"] = ast.literal_eval(line.split(":")[1].strip())
        elif "MI co-locate:" in line:
            data["MI co-locate"] = ast.literal_eval(line.split(":")[1].strip())
        elif "co-locate sum:" in line:
            data["co-locate sum"] = ast.literal_eval(line.split(":")[1].strip())

# 对数据进行归一化
for key in data:
    if key == "co-locate sum" or "/" in key:
        continue
    throughput_key = "CI standalone" if "CI" in key else "MI standalone"
    normalization_factor = throughputs[throughput_key]
    data[key] = [[x, y / normalization_factor] for x, y in data[key]]

for i in range(1, len(data["CI co-locate"])):
    data["CI co-locate/standalone"].append([data["CI co-locate"][i][0], data["CI co-locate"][i][1] / data["CI standalone"][i][1]])
for i in range(1, len(data["MI co-locate"])):
    data["MI co-locate/standalone"].append([data["MI co-locate"][i][0], data["MI co-locate"][i][1] / data["MI standalone"][i][1]])


# 绘制 CI 图表
plt.figure(figsize=(10, 6))
for key in ["CI standalone", "CI co-locate", "CI co-locate/standalone"]:
    x = [item[0] for item in data[key]]
    y = [item[1] for item in data[key]]
    plt.plot(x, y, label=key)

plt.title("CI Performance Comparison (Normalized)")
plt.xlabel("# SM")
plt.ylabel("Normalized Throughput")
plt.legend()
plt.grid(True)
plt.savefig(output_dir + "/ci_throughput.png")  
plt.close()

# 绘制 MI 图表
plt.figure(figsize=(10, 6))
for key in ["MI standalone", "MI co-locate", "MI co-locate/standalone"]:
    x = [item[0] for item in data[key]]
    y = [item[1] for item in data[key]]
    plt.plot(x, y, label=key)

plt.title("MI Performance Comparison (Normalized)")
plt.xlabel("# SM")
plt.ylabel("Normalized Throughput")
plt.legend()
plt.grid(True)
plt.savefig(output_dir + "/mi_throughput.png")
plt.close()

# 绘制 sum 图表
plt.figure(figsize=(10, 6))
for key in ["co-locate sum"]:
    x = [item[0] for item in data[key]]
    y = [item[1] for item in data[key]]
    plt.plot(x, y, label=key)

plt.title("Co-located Throughput Sum (Normalized)")
plt.xlabel("# SMs")
plt.ylabel("Normalized Throughput")
plt.legend()
plt.grid(True)
plt.savefig(output_dir + "/co_locate_sum_throughput.png")
plt.close()

import csv

# 将数据输出为 CSV 文件
csv_file_path = output_dir + "/data_output.csv"
with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 写入表头
    # 把所有空格转换为_
    keys = list(data.keys())
    keys = [key.replace(" ", "_") for key in keys]
    headers = ["n_SM"] + keys
    csv_writer.writerow(headers)
    
    # 获取所有 x 坐标（假设所有数据的 x 坐标相同）
    x_values = [item[0] for item in data["CI standalone"]]
    
    # 写入每一行数据
    for i in range(len(x_values)):
        row = [x_values[i]]
        for key in data.keys():
            if i < len(data[key]):
                row.append(data[key][i][1])  # 添加 y 值
            else:
                row.append(None)  # 如果某列数据不足，填充 None
        csv_writer.writerow(row)

print(f"Data has been saved to {csv_file_path}")