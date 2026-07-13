import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_p_all_in_one(Iis, Ijs, x_values, filename="plot.png"):

    fig, ax = plt.subplots()

    with open("roofline.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["x"] + [f"Ij{Ij}" for Ij in Ijs])
        y_values = []
        for Ii in Iis:
            for Ij in Ijs:
                ki = max(1, 1 / Ii)  # 确保 ki 不小于 1
                kj = max(1, 1 / Ij)  # 确保 kj 不小于 1
                Pi = np.minimum(ki * x_values, ki * x_values / (x_values / Ii + (1 - x_values) / Ij))
                Pj = np.minimum(kj * (1 - x_values), kj * (1 - x_values) / (x_values / Ii + (1 - x_values) / Ij))
                y = Pi + Pj
                y_values.append(y)
                # ax.plot(x_values, Pi + Pj, label=f"            ", linewidth=3)
        for i, x in enumerate(x_values):
            row = [x] + [y[i] for y in y_values]
            writer.writerow(row)
    # ax.set_xlabel("x")
    # ax.set_ylabel("P")
    # ax.set_title(f"P, Ii = {Ii}")
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_linewidth(2.5)  # 左边框加粗
    # ax.spines['bottom'].set_linewidth(2.5)  # 下边框加粗
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.annotate('', xy=(1, 0), xytext=(0, 0),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
    # ax.annotate('', xy=(0, 1), xytext=(0, 0),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
    # ax.legend(prop={'size': 20})  # 将字体大小设置为 20（2 倍默认大小）
    # ax.tick_params(axis='both', which='major', labelsize=20 )  # 刻度字体大小
    # ax.grid(True)

    # plt.savefig(filename)  # 保存图像到文件
    # plt.close(fig) #关闭图像,释放内存

    # print(f"图像已保存到文件: {filename}")


if __name__ == "__main__":
    # 设置参数
    Iis = [0.3]
    Ijs = [1.5, 2, 4]  # 不同的 Ij 值
    x_values = np.linspace(0, 1, 100)  # 从 0 到 1 的 x 值，100 个点
    filename = "roofline.png"  # 设置文件名

    # 绘制图像并保存到文件
    plot_p_all_in_one(Iis, Ijs, x_values, filename)