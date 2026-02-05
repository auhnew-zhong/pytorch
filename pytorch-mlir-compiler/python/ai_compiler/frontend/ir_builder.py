from typing import Dict, List, Any

from ai_compiler.ir.nodes import IRGraph, IRNode, IRValue


def _infer_output_metadata_from_fx_node(node) -> tuple[str, List[int] | None]:
    """Extract dtype/shape from fx node metadata if shape propagation populated it."""
    meta = getattr(node, "meta", None)
    if not isinstance(meta, dict):
        return "unknown", None
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is None:
        return "unknown", None
    # Some ops return tuples; normalize to a single tensor_meta when possible.
    if isinstance(tensor_meta, (list, tuple)) and tensor_meta and not hasattr(tensor_meta, "dtype"):
        tensor_meta = tensor_meta[0]
    dtype = getattr(tensor_meta, "dtype", None)
    shape = getattr(tensor_meta, "shape", None)
    dtype_str = str(dtype) if dtype is not None else "unknown"
    shape_list = list[Any](shape) if shape is not None else None
    return dtype_str, shape_list


def build_ir_from_fx(graph_module, example_inputs) -> IRGraph:
    """Lower an FX GraphModule into the project's IRGraph representation."""
    import torch.fx as fx

    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)

    ir_graph = IRGraph()
    # env maps fx.Node -> IRValue to track already materialized values.
    env: Dict[Any, IRValue] = {}
    placeholder_index = 0

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            # Placeholders correspond to model inputs. We read real input tensors
            # to capture dtype/shape for the IR.
            name = node.name  # 原料名字，如"面粉"
            tensor = example_inputs[placeholder_index]  # 原料样品
            placeholder_index += 1
            dtype = str(tensor.dtype)  # 原料类型，如"面粉"
            shape = list(tensor.shape)  # 原料形状，如"1袋"
            value = IRValue(name=name, dtype=dtype, shape=shape)
            ir_graph.add_input(value)  # 把原料瓶放进盒子
            env[node] = value  # 记录：中文"面粉" → 乐高块A

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue  # 跳过原料和成品

        # 1. 收集需要的原料（从翻译词典找对应的乐高块）
        inputs: List[IRValue] = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                inputs.append(env[arg])  # 查词典："面粉" → 乐高块A

        # 2. 记录特殊要求（如"搅拌均匀"、"小火慢炖"）
        attrs = dict[Any, Any](node.kwargs)

        # 3. 制作新的乐高块（烹饪结果）
        outputs: List[IRValue] = []
        out_name = node.name  # 步骤名，如"和面"
        # Prefer metadata from ShapeProp; fallback to unknown if unavailable.
        dtype, shape = _infer_output_metadata_from_fx_node(node)  # 推断结果
        value = IRValue(name=out_name, dtype=dtype, shape=shape)
        outputs.append(value)

        # 4. 制作乐高步骤卡
        # Normalize target to string, so C++ side can use it as a stable op name.
        target = node.target if isinstance(node.target, str) else str(node.target)
        ir_node = IRNode(op=target, inputs=inputs, outputs=outputs, attrs=attrs)
        
        # 5. 把步骤卡放进盒子，更新词典
        ir_graph.add_node(ir_node)
        env[node] = value  # 记录："和好的面" → 乐高块B

    for node in graph_module.graph.nodes:
        if node.op != "output":
            continue
        # FX output can be a single node or a tuple/list of nodes.
        result = node.args[0]
        if isinstance(result, fx.Node):
            ir_graph.add_output(env[result])
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, fx.Node):
                    ir_graph.add_output(env[item])

    return ir_graph
