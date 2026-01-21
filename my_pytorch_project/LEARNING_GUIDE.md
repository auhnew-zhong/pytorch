# 学习指南（新手版）

这份文档帮助你快速理解并上手当前子项目的代码结构与运行流程。建议按顺序阅读与操作。

## 1. 项目全局结构

- data：数据存放位置（原始、处理后、划分）
- models：模型与自定义层
- utils：数据、指标、可视化工具
- configs：训练配置
- scripts：训练、评估、推理脚本
- notebooks：探索性分析
- checkpoints：模型保存
- logs：训练日志
- tests：单元测试
- main.py：主入口

## 2. 运行入口与最小闭环

先跑通一个最小闭环：训练 -> 输出 loss。

```bash
python /home/auhnewzhong/pytorch/my_pytorch_project/main.py
```

你会看到类似输出：
```
final_loss=...
```

这一步确认：模型、数据、优化器、训练循环都能跑通。

## 3. 先理解 4 个核心文件

### 3.1 模型定义

- [models/simple_nn.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/models/simple_nn.py)

重点理解：
- `__init__` 里定义层（`nn.Linear`）
- `forward` 里定义前向传播流程

### 3.2 数据集构建

- [utils/data_utils.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/data_utils.py)

重点理解：
- `make_tensor_dataset` 如何把数据转成 `TensorDataset`

### 3.3 训练脚本

- [scripts/train.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/scripts/train.py)

建议按顺序看：
- 读取配置 `load_config`
- 构造随机数据
- 构造模型 `SimpleNN`
- 损失函数 + 优化器
- 训练循环（forward、loss、backward、step）

### 3.4 配置文件

- [configs/default.yaml](file:///home/auhnewzhong/pytorch/my_pytorch_project/configs/default.yaml)

学习配置驱动训练流程的方式，后续做实验可以直接新建 `experiment*.yaml`。

## 4. 按“从输入到输出”的路径学习

建议你顺着“数据流”理解项目：

1) 输入数据在 `train.py` 里生成  
2) 数据被封装为 `TensorDataset`  
3) `DataLoader` 按 batch 迭代  
4) `model(batch_x)` 做前向传播  
5) `criterion(output, batch_y)` 计算损失  
6) `loss.backward()` 反向传播  
7) `optimizer.step()` 更新参数  

理解这条链路，基本就理解了训练过程。

## 5. 评估与推理

### 5.1 评估

```bash
python /home/auhnewzhong/pytorch/my_pytorch_project/scripts/evaluate.py --config configs/default.yaml
```

你会得到 `mse=...`，它来自 [utils/metrics.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/metrics.py)。

### 5.2 推理

```bash
python /home/auhnewzhong/pytorch/my_pytorch_project/scripts/predict.py --config configs/default.yaml
```

这一步只做前向传播，并且在 `torch.no_grad()` 下运行，不构建计算图。

## 6. 单元测试

```bash
python -m unittest /home/auhnewzhong/pytorch/my_pytorch_project/tests/test_smoke.py
```

这会验证模型输入/输出的 shape 是否正确，是最小的“稳定性检查”。

## 7. 新手建议的学习顺序

1) 先跑 `main.py`  
2) 理解 `SimpleNN` 的结构与 forward  
3) 理解训练循环与 loss 计算  
4) 修改 `configs/default.yaml` 观察训练变化  
5) 把 `SimpleNN` 改成 3 层或更大 hidden_dim  
6) 给 `evaluate.py` 增加新指标  
7) 用真实数据替换 `train.py` 里的随机数据  

## 8. 常见问题定位

- `ModuleNotFoundError`  
  - 检查是否从项目根目录运行
- loss 不下降  
  - 学习率过大/过小、数据噪声过高
- shape 不匹配  
  - 检查 `input_dim` 与数据维度是否一致

## 9. 下一步可做的小练习

- 把 `SimpleNN` 改成带 `Dropout` 的版本  
- 用 `torch.manual_seed` 固定随机性  
- 保存与加载模型权重（`torch.save`/`torch.load`）  
- 加一个 `DataLoader` 的 `num_workers` 参数观察性能  

## 10. 设备与张量（CPU/CUDA）

核心原则：模型参数和输入张量必须在同一个设备上。

常用写法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

推理/评估阶段通常配合：

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

## 11. 保存与恢复训练（checkpoint）

训练过程中保存模型，之后可以恢复继续训练。

本项目提供了最小的 checkpoint 工具：
- [utils/checkpointing.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/checkpointing.py)

对应示例脚本：
- [scripts/train_with_checkpoint.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_with_checkpoint.py)

运行：

```bash
python /home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_with_checkpoint.py --config configs/default.yaml --save checkpoints/last.pt
python /home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_with_checkpoint.py --config configs/default.yaml --resume checkpoints/last.pt --save checkpoints/last.pt
```

学习目标：
- 理解 `state_dict()` 是什么
- 理解保存的不只是模型，还可以保存优化器状态、epoch/step

## 12. 用真实数据替换随机数据（CSV 示例）

项目提供了一个最小 CSV 数据集示例：
- [data/raw/example_regression.csv](file:///home/auhnewzhong/pytorch/my_pytorch_project/data/raw/example_regression.csv)

对应数据集类：
- [utils/csv_dataset.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/csv_dataset.py)

训练脚本：
- [scripts/train_csv.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_csv.py)

配置文件：
- [configs/csv_example.yaml](file:///home/auhnewzhong/pytorch/my_pytorch_project/configs/csv_example.yaml)

运行：

```bash
python /home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_csv.py --config configs/csv_example.yaml
```

学习目标：
- 理解 `Dataset` 的 `__len__` 与 `__getitem__`
- 理解训练集/验证集切分
- 理解 `DataLoader` 的 shuffle 与 batch_size

## 13. 训练日志与如何读输出

建议你先把输出结构读懂，再做更复杂的日志系统：
- `epoch=... train_mse=... val_mse=...` 表示每轮训练/验证的平均损失
- `loss` 越小通常代表拟合更好（但要警惕过拟合）

当你准备做“可视化”时，可以把 loss 记录到列表并用：
- [utils/visualization.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/visualization.py)

## 14. 调试与排错清单

最常用的排错步骤：
- 打印 `x.shape`、`y.shape`、`pred.shape`
- 打印 `model` 结构，确认层的输入输出维度
- 先用很小的数据量跑通（比如 8 条样本、训练 1 个 epoch）
- 先在 CPU 跑通再迁移到 GPU

## 15. 推荐练习路线（带对应文件）

1) 修改网络结构：改 `hidden_dim` 或加一层  
   - [models/simple_nn.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/models/simple_nn.py)
2) 跑通 checkpoint 恢复  
   - [scripts/train_with_checkpoint.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/scripts/train_with_checkpoint.py)
3) 把 CSV 换成你自己的数据  
   - [utils/csv_dataset.py](file:///home/auhnewzhong/pytorch/my_pytorch_project/utils/csv_dataset.py)

如果你希望，我可以继续把这套示例升级为“真实项目训练模板”：含 train/val/test 划分保存、日志文件、最优模型保存、命令行参数更完整。
