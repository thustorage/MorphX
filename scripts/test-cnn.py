import torch
import torch.nn as nn
from typing import List, Union, cast
import time # 用于计时

# --- VGG19 配置 ---
# (数字: 输出通道数, 'M': MaxPool)
vgg19_cfg: List[Union[str, int]] = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
    512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'
]

# --- 辅助函数：构建卷积/池化层 ---
def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            # Conv2d -> ReLU (最简结构)
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# --- VGG 模型定义 (精简版) ---
class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 固定输出大小
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # Dropout 通常在训练时使用，推理时可以省略以求简洁
            # nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平
        x = self.classifier(x)
        return x

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 1. 设置设备 (优先使用 GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU 不可用，使用 CPU")

    s0 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        # 2. 创建 VGG19 模型实例 (1000 类)
        print("创建 VGG19 模型结构...")
        model = VGG(make_layers(vgg19_cfg), num_classes=1000)

        # 3. 将所有参数设置为 0 (不加载预训练权重)
        print("将所有模型参数初始化为 0...")
        with torch.no_grad(): # 确保在非训练模式下操作
            for param in model.parameters():
                param.zero_() # 直接原地将参数置零

        # 4. 将模型移至 GPU (如果可用) 并设置为评估模式
        model.to(device)
        model.eval() # 关键：设置为评估模式
        print(f"模型已移至 {device} 并设置为评估模式")

        # 5. 创建一个随机输入张量 (模拟一批图像)
        # batch_size=4, channels=3, height=224, width=224
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        print(f"创建随机输入张量，形状: {dummy_input.shape}, 设备: {dummy_input.device}")

        # 6. 执行推理 (在 torch.no_grad() 下以节省资源)
        print("开始推理...")
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        end_time = time.time()
        print(f"推理完成，耗时: {end_time - start_time:.4f} 秒")

        # 7. 打印输出信息
        print(f"输出张量形状: {output.shape}") # 预期: [batch_size, num_classes] -> [4, 1000]
        # 由于所有权重和偏置都为0，且经过ReLU，预期输出应该全为0
        print(f"输出张量在 CPU 上的 L1 范数 (检查是否为0): {torch.linalg.norm(output.cpu(), ord=1).item()}")
        # print(f"部分输出值: {output[0, :10].cpu().numpy()}") # 查看前10个类别的值