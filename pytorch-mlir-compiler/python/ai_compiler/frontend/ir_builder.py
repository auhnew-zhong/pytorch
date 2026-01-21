from typing import Dict, List, Any

from ai_compiler.ir.nodes import IRGraph, IRNode, IRValue

def build_ir_from_fx(graph_module, example_inputs) -> IRGraph:
    import torch.fx as fx

    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)

    ir_graph = IRGraph()
    env: Dict[Any, IRValue] = {}
    placeholder_index = 0

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            tensor = example_inputs[placeholder_index]
            placeholder_index += 1
            dtype = str(tensor.dtype)
            shape = list(tensor.shape)
            value = IRValue(name=name, dtype=dtype, shape=shape)
            ir_graph.add_input(value)
            env[node] = value

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        inputs: List[IRValue] = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                inputs.append(env[arg])

        attrs = dict(node.kwargs)
        outputs: List[IRValue] = []
        out_name = node.name
        value = IRValue(name=out_name, dtype="unknown", shape=None)
        outputs.append(value)

        target = node.target if isinstance(node.target, str) else str(node.target)
        ir_node = IRNode(op=target, inputs=inputs, outputs=outputs, attrs=attrs)
        ir_graph.add_node(ir_node)
        env[node] = value

    for node in graph_module.graph.nodes:
        if node.op != "output":
            continue
        result = node.args[0]
        if isinstance(result, fx.Node):
            ir_graph.add_output(env[result])
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, fx.Node):
                    ir_graph.add_output(env[item])

    return ir_graph
