import torch
import torchvision
import time
from torch.amp import autocast

# 检查硬件是否支持 bf16
if not torch.cuda.is_bf16_supported():
    print("Warning: Your GPU does not support BF16 natively.")

model = torchvision.models.resnet50().cuda()
model = model.to(memory_format=torch.channels_last)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 模拟输入
input_data = torch.randn(256, 3, 256, 256).cuda()
input_data = input_data.to(memory_format=torch.channels_last)
target = torch.randint(0, 1000, (128,)).cuda()

print("Starting training loop with BF16 precision:", flush=True)

iteration = 0
while True:
    start_time = time.time()
    optimizer.zero_grad()

    # 使用 autocast 开启 bf16 上下文
    # dtype 指定为 torch.bfloat16
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input_data)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"Iteration {iteration}: Loss: {loss.item()}, Time: {end_time - start_time}s", flush=True)
    iteration += 1