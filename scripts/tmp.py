import pandas as pd
import matplotlib.pyplot as plt
# from io import StringIO # 如果直接从文件读取，这个就不需要了

# CSV文件路径
file_path = '/home/rtx/gpu/sm-sched/scripts/results.csv' # 请确保这个路径是正确的

# 读取CSV文件，并将第一列作为索引
# 如果你的CSV文件的第一行是列名，并且第一列确实是你要用作行标识的，那么 index_col=0 是正确的
try:
    df = pd.read_csv(file_path, index_col=0)
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请检查路径。")
    exit()
except Exception as e:
    print(f"读取CSV文件时发生错误：{e}")
    exit()

# 只保留前3列数据 ('stream', 'gplux', 'mps')
columns_to_plot = ['stream', 'gplux', 'mps']

# 检查所选列是否存在于DataFrame中
missing_cols = [col for col in columns_to_plot if col not in df.columns]
if missing_cols:
    print(f"错误：以下指定的列在CSV文件中不存在：{', '.join(missing_cols)}")
    print(f"可用的列有：{', '.join(df.columns)}")
    exit()

df_selected = df[columns_to_plot].copy()

# 处理可能因选取列后产生的NaN值
df_selected = df_selected.fillna(0)

# 每3行绘制一个图
rows_per_plot = 3
num_plots = (len(df_selected) + rows_per_plot - 1) // rows_per_plot

for i in range(num_plots):
    start_row = i * rows_per_plot
    end_row = start_row + rows_per_plot
    chunk_df = df_selected.iloc[start_row:end_row]

    if chunk_df.empty:
        continue

    # 绘制柱状图
    # pandas的plot(kind='bar')会自动将DataFrame的索引（现在是CSV的第一列）
    # 作为X轴上的刻度标签。每一行数据成为一个X轴上的组。
    ax = chunk_df.plot(kind='bar', figsize=(12, 7), width=0.8)
    
    # 设置图表标题和标签
    # chunk_df.index.tolist() 将包含例如 ['graph', 'GEMM', 'sum-graph-gemm']
    plot_title_elements = chunk_df.index.tolist()
    # 您可以取消注释并自定义标题
    ax.set_title(f'Comparison for: {", ".join(plot_title_elements)}')
    
    ax.set_ylabel('Nomalized Throughput') # Y轴标签
    
    # X轴标签 (可选, pandas会自动使用索引的名称作为X轴标签，如果索引有名称的话)
    # 如果索引没有名称 (通常CSV第一列的列头是空的), 则不会有总的X轴标签，只有刻度标签
    # 如果您想强制一个X轴的名称，可以取消注释下面这行
    # ax.set_xlabel('Benchmark Groups') # 或者更具体的名称

    # 旋转X轴刻度标签以便更好地显示
    # chunk_df.index.name 是索引列的名称（如果CSV第一行第一列有内容）
    # chunk_df.index.values 是具体的行名
    plt.xticks(ticks=range(len(chunk_df.index)), labels=chunk_df.index, rotation=0, ha='center')
    # rotation=0: 水平显示, rotation=45: 45度倾斜
    # ha='right' 配合 rotation=45 可以让标签不重叠
    
    # 添加图例
    ax.legend(title='Metrics')

    # 调整布局以防止标签重叠
    plt.tight_layout()
    
    # 保存图表到文件
    # 使用行名来创建更具描述性的文件名
    safe_elements_for_filename = "_".join(plot_title_elements).replace(' ', '_').replace('/', '_')
    filename = f'bar_chart_{safe_elements_for_filename}_group_{i+1}.png'
    plt.savefig(filename)
    print(f'Saved plot to {filename}')
    
    # 关闭当前图表，以便为下一个图表释放内存
    plt.close()

print("All plots generated and saved.")