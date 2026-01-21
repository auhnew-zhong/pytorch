# pytorch（个人学习与实验目录）

这个仓库用于记录我在 PyTorch 相关方向的学习、实验代码与小型子项目。

## 目录说明

- `my_pytorch_project/`：一个最小可运行的 PyTorch 训练项目骨架（models/utils/configs/scripts/tests 等）
- `pytorch-mlir-compiler/`：一个 Torch FX -> 自定义 IR -> MLIR/LLVM IR 的最小骨架工程
- `checkpoints/`：训练产物（默认被 git 忽略）
- `test_*.py`：零散的学习/验证脚本
- `DeepLearning_with_pytorch.md`：学习笔记

## 快速开始

### 运行 my_pytorch_project

```bash
python my_pytorch_project/main.py
python my_pytorch_project/scripts/train.py --config configs/default.yaml
python my_pytorch_project/scripts/evaluate.py --config configs/default.yaml
python my_pytorch_project/scripts/predict.py --config configs/default.yaml
python -m unittest my_pytorch_project/tests/test_smoke.py
```

### 运行 pytorch-mlir-compiler（阅读/构建提示）

该目录是最小骨架工程，Python 侧会尝试 import C++ 扩展 `_ai_compiler_mlir`。
要让其端到端跑通，通常需要本机已有 MLIR/LLVM 开发环境并能构建 pybind 扩展。

## Git 约定

- 已添加 [.gitignore](file:///home/auhnewzhong/pytorch/.gitignore)，默认忽略缓存、日志、checkpoint、构建产物与数据目录
- 建议只提交：源代码、配置、文档、最小可复现脚本
