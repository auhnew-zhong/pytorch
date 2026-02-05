# pytorch-mlir-compiler 项目阅读文档

## 0. 一句话总结

这是一个“从 PyTorch FX 图到自定义 IR，再到 MLIR 文本”的最小骨架工程：Python 侧负责捕获与 IR 构建/优化，C++ 侧提供 MLIR Dialect/Pass/Pipeline，并通过 pybind 暴露 `_ai_compiler_mlir.compile_from_serialized_ir` 接口给 Python 调用。

当前实现仍是“脚手架/教学 Demo”，但 C++ 侧已会消费传入的序列化 IR：会解析 inputs/nodes/outputs，构建 `func.func @main` 与多个 callee 函数，并将连线关系以 `func.call` 的形式体现在 MLIR 里。

---

## 1. 目录结构与代码地图

项目根目录：[/pytorch-mlir-compiler](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler)

- CMake 工程入口
  - [CMakeLists.txt](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/CMakeLists.txt)
  - [cpp/CMakeLists.txt](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/CMakeLists.txt)
- Python 包（`python/ai_compiler`）
  - 顶层 API：[python/ai_compiler/__init__.py](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/__init__.py)
  - 前端（FX 捕获、IR 构建）：[python/ai_compiler/frontend](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/frontend)
  - IR（节点/图定义、pass）：[python/ai_compiler/ir](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/ir)
  - 后端（序列化 + 调用 C++ 扩展）：[python/ai_compiler/backend/mlir_export.py](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/backend/mlir_export.py)
- C++/MLIR 扩展（`cpp`）
  - Dialect：[cpp/include/ai_compiler/Dialect.h](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/include/ai_compiler/Dialect.h)、[cpp/lib/Dialect.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Dialect.cpp)
  - Pass：[cpp/include/ai_compiler/Passes.h](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/include/ai_compiler/Passes.h)、[cpp/lib/Passes.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Passes.cpp)
  - Pipeline：[cpp/include/ai_compiler/Pipeline.h](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/include/ai_compiler/Pipeline.h)、[cpp/lib/Pipeline.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Pipeline.cpp)
  - Python 绑定：[cpp/lib/PyBind.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/PyBind.cpp)

---

## 2. 端到端数据流：从 PyTorch 到 MLIR 文本

顶层入口函数是 `ai_compiler.compile_torch_model`：
- [compile_torch_model](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/__init__.py#L7-L12)

流水线（当前版本）：

1) **FX 捕获**：`torch.fx.symbolic_trace(model)` + `ShapeProp`  
   - 实现：[capture_fx_graph](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/frontend/fx_capture.py#L1-L12)
   - 输出：`torch.fx.GraphModule`，并填充 `node.meta['tensor_meta']`（用于后续 dtype/shape 推导）

2) **FX -> 自定义 IR**：遍历 `graph_module.graph.nodes`，构造 `IRGraph/IRNode/IRValue`  
   - 实现：[build_ir_from_fx](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/frontend/ir_builder.py#L5-L57)
   - 特点：
     - placeholder 的 dtype/shape 来自 `example_inputs`
     - 非 placeholder 节点输出 dtype 被标为 `"unknown"`，shape 为 `None`（没有做 shape 推导）
     - op 名称用 `node.target`（不是 `node.op`）

3) **IR Pass Pipeline**：做最小 IR 层优化  
   - 实现：[run_ir_pass_pipeline](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/ir/passes.py#L14-L16)
   - 当前仅做 `eliminate_noop_nodes`：丢弃 `aten.dropout`、`aten.identity`

4) **IR 序列化**：转成 dict（inputs/outputs/nodes/attrs）  
   - 实现：[serialize_ir_graph](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/backend/mlir_export.py#L17-L36)

5) **调用 C++ 扩展编译**：`_ai_compiler_mlir.compile_from_serialized_ir(serialized, target)`  
   - Python 侧入口：[ir_to_mlir_and_compile](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/backend/mlir_export.py#L9-L14)
   - C++ 侧入口：[compileFromSerializedIr](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/PyBind.cpp#L23-L213)
   - 当前返回值：MLIR 文本字符串（`std::string mlirText`）

---

## 3. Python 侧：你应该怎么读

### 3.1 顶层 API（你给用户暴露什么）

`python/ai_compiler/__init__.py` 把前端/IR/后端串成一条 `compile_torch_model`：
- FX 图捕获
- 构建自定义 IR
- IR pass pipeline
- 导出/编译（调用扩展）

这是“从用户视角”理解项目的最佳入口点。

### 3.2 前端：FX 捕获（当前最简）

`capture_fx_graph` 做 `fx.symbolic_trace(model)` 后会用 `ShapeProp` 结合 `example_inputs` 进行形状推导，将结果写入 `node.meta`。

这意味着：
- 能捕获到的算子集合取决于 `symbolic_trace` 的能力与模型结构
- 对动态控制流/数据相关分支，FX 捕获可能会失败或不完整

### 3.3 前端：FX -> IR（当前 IR 更像“抽象计算图”）

`build_ir_from_fx` 的核心机制是用 `env: Dict[fx.Node, IRValue]` 维护“FX 节点到 IRValue”的映射：
- placeholder：从 `example_inputs` 取 dtype/shape，加入 `ir_graph.inputs`
- 其他节点：收集输入 `IRValue`，将 `node.kwargs` 作为 attrs，创建一个输出 `IRValue` 并加入 `ir_graph.nodes`
- output：把 output 引用的 IRValue 加入 `ir_graph.outputs`

当前 IRNode 的 `op` 字段是 `node.target` 的字符串化结果（例如 `aten.add`、`operator.getitem` 等），方便后续做 pattern match/lowering。

### 3.4 IR：数据结构与 pass

- IR 数据结构：[nodes.py](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/ir/nodes.py)
  - `IRValue`：value 的 name/dtype/shape
  - `IRNode`：op、inputs、outputs、attrs
  - `IRGraph`：inputs、outputs、nodes
- IR pass pipeline：[passes.py](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/python/ai_compiler/ir/passes.py)
  - 当前只有“删除 noop 节点”的 pass

如果你要扩展 IR 优化能力，通常会在这里加：
- 常量折叠（constant folding）
- shape 推导/类型推导
- 算子融合（fusion）
- 规范化（canonicalization）

---

## 4. C++ 侧：MLIR Dialect/Pass/Pipeline 与 pybind

### 4.1 Dialect

- Dialect 声明：[Dialect.h](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/include/ai_compiler/Dialect.h)
- 构造实现：[Dialect.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Dialect.cpp)

当前 Dialect 仅完成注册框架，没有定义任何 op/type/attr。

### 4.2 Pass 与 Pipeline

- Pass API：[Passes.h](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/include/ai_compiler/Passes.h)
- Pass 实现：[Passes.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Passes.cpp)
  - `AISimplifyPass::runOnOperation()` 目前为空实现
  - `registerAIPasses()` 用 `PassRegistration` 注册了 `"ai-simplify"`
- Pipeline：[Pipeline.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/Pipeline.cpp)
  - `buildAIPipeline` 把 `createAISimplifyPass()` 加入 PassManager

### 4.3 pybind 扩展入口与当前行为

`cpp/lib/PyBind.cpp` 定义了 Python 扩展模块 `_ai_compiler_mlir`：
- [PYBIND11_MODULE](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/PyBind.cpp#L53-L55)

`compile_from_serialized_ir(ir, target)` 当前行为：
- 解析 `inputs/nodes/outputs` 并生成 MLIR 类型
- 创建 `func.func @main`，将 inputs 作为 block 参数
- 为每个 op 生成私有 callee func，并在 main 中用 `func.call` 表达连线关系
- 跑 pass pipeline（当前仍为空实现）
- 返回 MLIR 文本字符串

所以：当前“编译结果”主要用于验证“序列化 IR -> MLIR 文本”的连通性与可视化结构，而不是完整的 Torch->MLIR lowering。

---

## 5. 构建与运行（当前工程的实际情况）

### 5.1 依赖

从 CMake 能看出的硬依赖：
- MLIR（`find_package(MLIR REQUIRED CONFIG)`）：[CMakeLists.txt](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/CMakeLists.txt#L6)
- Python3 开发环境（Interpreter + Development）：[CMakeLists.txt](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/CMakeLists.txt#L7)

从代码能看出的额外依赖：
- pybind11 头文件（`#include <pybind11/pybind11.h>`）：[PyBind.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/PyBind.cpp#L4-L6)
- PyTorch（Python 前端依赖 `torch.fx`）
- LLVM（MLIR 的 translate 接口需要 LLVM 支撑）

### 5.2 重要提示：当前 CMake 更像“最小链接示例”

`cpp/CMakeLists.txt` 已改为构建 Python 扩展模块（`MODULE` + `pybind11::module`），并显式引入 MLIR/LLVM include dirs。仍然缺少：
- Python 包安装/打包（`python/` 目录没有 `setup.py/pyproject.toml`）
- 一键构建脚本（需要手动配置 MLIR/LLVM 与 pybind11 的路径）

因此更合理的定位是：这里展示了“如何把 MLIR + pybind 粘起来”，但距离一键构建/可 pip 安装还有工程化工作要补。

---

## 6. 如何在阅读中建立“正确心智模型”

建议你按这个顺序理解：

1) 从 Python 顶层入口读起：`compile_torch_model`  
2) 看 FX 捕获与 FX node 的结构（重点理解 `node.op/node.target/node.args/node.kwargs`）  
3) 看 IRGraph/IRNode/IRValue 的设计，弄清楚：为什么需要“独立于 FX 的 IR”  
4) 看 pass pipeline：理解“为什么编译器要有中间层优化”  
5) 看 `serialize_ir_graph`：理解跨语言边界的数据格式约束  
6) 看 C++ 的 `compileFromSerializedIr`：理解 MLIRContext、Dialect、PassManager、ModuleOp、translate 到 LLVMIR 这条链

---

## 7. 下一步工程化/实现路线（如果你要把它变成真正编译器）

按实现优先级给一个可落地路线：

1) **把 C++ 侧生成的 MLIR 从“函数调用骨架”扩展到具体算子**
   - 在 [PyBind.cpp](file:///home/auhnewzhong/pytorch/pytorch-mlir-compiler/cpp/lib/PyBind.cpp) 中为常见 op 建立真实 MLIR op
   - 增加最小的 dialect/type 规则，避免大量使用 fallback type

2) **完善类型/形状系统**
   - Python IR 层做 dtype/shape 推导，或把推导放到 MLIR 的 type system

3) **扩展 pass 与 pipeline**
   - IR 层：做 canonicalization/fusion
   - MLIR 层：加合法化、bufferize、lowering 到 LLVM 的 pipeline

4) **打包与构建体验**
   - 增加 `pyproject.toml`，支持 `pip install -e .`
   - CMake 侧用 pybind11 的标准集成方式，确保 `_ai_compiler_mlir` 可被 import

---

## 8. 最小使用示例（理解行为用）

Python 侧（需要 `_ai_compiler_mlir` 已正确构建并可 import）：

```python
import torch
from ai_compiler import compile_torch_model


class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = M()
mlir_text = compile_torch_model(model, (torch.randn(2, 3), torch.randn(2, 3)))
print(type(mlir_text))
print(mlir_text[:200])
```

当前你应该把返回值理解为“序列化 IR 是否成功转成 MLIR 文本”的验证信号，而不是最终可执行产物。
