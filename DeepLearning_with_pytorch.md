# 面向 C++/TSL 背景的深度学习学习路线（含 PyTorch 项目清单）

## 1. 你现在的优势与差距

**优势（来自 C++/TSL/编译器背景）**

- 熟悉编译流程、IR、优化：对“计算图 + 自动求导”这类概念很容易类比。
- 有扎实的位运算、数值精度意识：理解数值稳定性、溢出等问题更轻松。
- 熟悉 Linux、构建系统（CMake）：安装、编译、部署不会成为障碍。

**可能的短板**

- Python / NumPy 不够熟练；
- 机器学习基础（损失函数、梯度下降、过拟合）还需要系统补一下；
- 对主流网络结构（CNN/RNN/Transformer）目前只是“听过名字”。

下面的路线就是针对这些特点设计的。

---

## 2. 学习总路线概览

按阶段划分，每个阶段都有“目标 + 建议动作 + 对应项目”。

1. **阶段 A：打基础（Python + NumPy + ML 基本概念）**
2. **阶段 B：PyTorch 入门（张量、autograd、训练循环）**
3. **阶段 C：经典模型与实战（CV/NLP 项目）**
4. **阶段 D：工程化与部署（日志、配置、导出、服务）**

> 建议节奏：  
> - A：1–2 周  
> - B：1–2 周  
> - C：2–4 周  
> - D：2+ 周（和真实需求结合）

---

## 3. 阶段 A：打基础

### 3.1 目标

- 能用 Python 写中等复杂度脚本（函数、类、模块、异常处理）。
- 熟练使用 NumPy 进行矩阵运算。
- 理解机器学习的基本概念：训练集/验证集、损失函数、梯度下降、过拟合/正则化。

### 3.2 建议内容

- **Python 语法**
  - 面向对象、上下文管理器、迭代器/生成器。
  - 与 C++ 的差异：动态类型、duck typing、异常风格。

- **NumPy**
  - `ndarray` 的 shape/broadcasting；
  - 基本线性代数（矩阵乘、转置、求和等）；
  - 用 NumPy 手写一次线性回归/逻辑回归（不借助框架）。

- **ML 基础**
  - 线性回归 / 逻辑回归；
  - 损失函数（MSE、交叉熵）；
  - 梯度下降、学习率、批量/小批量。

---

## 4. 阶段 B：PyTorch 入门（从“能跑一个模型”开始）

### 4.1 目标

- 能看懂/修改 PyTorch 示例代码；
- 了解核心 API：`Tensor`、`nn.Module`、`optim`、`DataLoader`、`autograd`；
- 能自己写出一个完整的训练脚本（包括训练/验证循环、保存模型）。

### 4.2 推荐学习顺序

1. **PyTorch 官方 Tutorials（强烈建议过一遍）**
   - 官网 “60-minute blitz” 系列：
     - Tensors
     - Autograd
     - Neural Networks
     - Training a classifier (用 CIFAR10)
   - 这些内容够你把框架骨架打通。

2. **第一个完整项目：MNIST 手写数字分类**
   - 网络：简单 MLP 或 2–3 层卷积。
   - 重点：
     - `Dataset/DataLoader` 使用；
     - 正向计算 → 计算 loss → `loss.backward()` → `optimizer.step()`；
     - 在验证集上计算准确率。

3. **第二个项目：CIFAR-10 图片分类**
   - 网络：简化版 ResNet 或自定义 CNN。
   - 重点：
     - 学习数据增强（`torchvision.transforms`）；
     - 学习学习率调度器（`torch.optim.lr_scheduler`）。

---

## 5. 阶段 C：结合兴趣方向的实战（CV / NLP）

### 5.1 计算机视觉方向（CV）

- **目标**：熟悉 CNN 及常用任务（分类、检测、分割）。

推荐顺序：

1. **迁移学习：用预训练 ResNet 做分类**
   - 思路：
     - 加载 `torchvision.models.resnet18(weights=...)`；
     - 冻结前面层，只训练最后的全连接层；
     - 换成你自己的小数据集（比如几类自定义图片）。
   - 学到的东西：
     - 预训练权重的使用；
     - 冻结/解冻参数；
     - 过拟合与正则化（L2 / dropout / data augmentation）。

2. **目标检测/分割（只做一个小 Demo 即可）**
   - 用 `torchvision` 的参考实现：
     - 检测：`torchvision/references/detection`
     - 分割：`torchvision/references/segmentation`
   - 改少量代码跑通一个公开数据集，了解：
     - 多任务损失（分类 + 回归）；
     - 更复杂的数据 pipeline。

### 5.2 NLP 方向

- **目标**：理解序列建模和 Transformer 的基本用法。

推荐方向：

1. **文本分类（情感分析）**
   - 使用预训练 BERT（如 `transformers` 库） 或简单 LSTM。
   - 学习：
     - 文本 tokenization；
     - padding/mask；
     - 序列长度管理。

2. **序列到序列（Seq2Seq）或小型翻译任务**
   - 重点在于理解：
     - Encoder-Decoder 结构；
     - 注意力机制的基本思路。

---

## 6. 阶段 D：工程化与部署（从“能跑”到“好用”）

### 6.1 训练脚本工程化

- 配置管理：
  - 使用 `argparse` 或更高级的配置工具（如 `hydra`）来管理超参数、路径等。
- 日志与可视化：
  - 使用 TensorBoard / wandb 记录 loss/acc 曲线。
- 断点续训：
  - 定期保存 checkpoint；
  - 能从 checkpoint 恢复训练。

### 6.2 推理与部署

- 模型导出：
  - PyTorch → TorchScript（`torch.jit.trace/script`）；
  - 或导出 ONNX（`torch.onnx.export`）。
- 简单服务化：
  - 用 Flask/FastAPI 包一层 HTTP API；
  - 或使用 TorchServe / TensorRT 等更专业工具（后续再深入）。

---

## 7. PyTorch 入门项目清单（含难度与顺序）

下面是结合你背景给出的 **项目 + GitHub 仓库** 建议（按难度从低到高）。

> 说明：链接可能随着时间有变动，建议打开时留意仓库的 README 与 star 数。

### 7.1 入门级（Level 1–2）

1. **pytorch/examples（官方示例集）**  
   - 仓库：<https://github.com/pytorch/examples>  
   - 推荐子项目：
     - `mnist`：最基础的分类任务；
     - `vae`：变分自编码器（可后面再看）。
   - 难度：★☆☆☆☆–★★☆☆☆
   - 学习重点：
     - PyTorch 基本使用；
     - 训练循环结构。

2. **简单 CIFAR-10 项目（kuangliu/pytorch-cifar）**  
   - 仓库：<https://github.com/kuangliu/pytorch-cifar>  
   - 难度：★★☆☆☆
   - 学习重点：
     - 各种经典 CNN 结构（ResNet、DenseNet 等）；
     - 数据增强、LR 调度器。

3. **中文教程风格项目：pytorch-book**  
   - 仓库：<https://github.com/chenyuntc/pytorch-book>  
   - 难度：★☆☆☆☆–★★★☆☆（按章节递增）
   - 学习重点：
     - 边看中文解释边读代码；
     - 适合作为系统 tutorial。

### 7.2 中级项目（Level 3–4）

4. **图像分类参考实现（ImageNet 风格）**  
   - 仓库：<https://github.com/pytorch/vision/tree/main/references/classification>  
   - 难度：★★★☆☆  
   - 学习重点：
     - 工程级训练脚本结构；
     - 多 GPU / 分布式训练的基本模式。

5. **情感分析（bentrevett/pytorch-sentiment-analysis）**  
   - 仓库：<https://github.com/bentrevett/pytorch-sentiment-analysis>  
   - 难度：★★★☆☆  
   - 学习重点：
     - NLP pipeline；
     - RNN / LSTM / attention。

6. **Seq2Seq 翻译（bentrevett/pytorch-seq2seq）**  
   - 仓库：<https://github.com/bentrevett/pytorch-seq2seq>  
   - 难度：★★★☆☆–★★★★☆  
   - 学习重点：
     - Encoder-Decoder；
     - Teacher forcing、beam search 等概念。

### 7.3 进阶项目（Level 4–5）

7. **目标检测（torchvision detection references）**  
   - 仓库：<https://github.com/pytorch/vision/tree/main/references/detection>  
   - 难度：★★★★☆  
   - 学习重点：
     - 更复杂的数据结构（多 bbox、多类别）；
     - 多任务 loss 设计。

8. **语义分割（torchvision segmentation references）**  
   - 仓库：<https://github.com/pytorch/vision/tree/main/references/segmentation>  
   - 难度：★★★★☆  
   - 学习重点：
     - 像素级预测；
     - 大分辨率输入的显存优化。

9. **服务化部署：TorchServe 示例**  
   - 仓库：<https://github.com/pytorch/serve>  
   - 难度：★★★★☆  
   - 学习重点：
     - 模型打包（`.mar`）；
     - REST 推理接口；
     - 简单性能调优。

---

## 8. 如何结合你当前的 TSL/编译器工作

给你几个“把现有技能用上”的角度：

- **把 autograd/计算图类比成 IR：**
  - 正向图 = 你现在的 LLVM IR；
  - 反向图 = 基于 IR 的自动微分 pass；
  - 你可以尝试写一些 toy 级别的自动求导（比如对简单表达式求导），把概念打通。

- **把训练脚本看成“编译 + 优化 + 执行”流水线：**
  - 数据预处理 = 前端/预处理器；
  - 模型定义 = IR 构建器；
  - 优化器更新 = 后端优化与 codegen 的“调优循环”。

- **性能调优经验可以直接迁移：**
  - 批大小、并行度、内存布局优化；
  - 分析训练脚本的“热点操作”（类似你看 profile）。

---

## 9. 推荐实际行动（今天就可以做）

1. 在一台有 CUDA 的机器上安装 PyTorch（或 CPU 版本）；
2. 跑通 `pytorch/examples/mnist`，保证完整训练一轮；
3. 把训练脚本里：
   - 前向过程；
   - 反向过程；
   - 参数更新  
   分别用你的“编译器眼光”画出简单的“计算图 + 数据流图”，这会让你对以后看到更复杂网络也不慌。

如果你愿意，我可以在下一步**帮你写一个“专门给你用的 PyTorch 模板训练脚本”**，里面把常用结构（参数解析、日志、checkpoint）都搭好，你只需要改“模型部分”和“数据部分”就能快速做实验。