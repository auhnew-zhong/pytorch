import torch
import torch.nn as nn

from test_nn import SimpleNN

# 随机输入
x = torch.randn(1, 2)
print(x)

# 前向传播
model = SimpleNN()
output = model(x)
print(output)

# 定义损失函数（例如均方误差 MSE）
criterion = nn.MSELoss()

# 假设目标值为 1
target = torch.randn(1, 1)

# 计算损失
loss = criterion(output, target)
print(loss)
