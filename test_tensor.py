import torch

# 创建一个未初始化的5x3矩阵
x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化的5x3矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个5x3的零矩阵，类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接从数据创建tensor
x = torch.tensor([55.0, 3.0])
print(x.dtype)
print(x)