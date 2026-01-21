from ai_compiler.ir.nodes import IRGraph


def eliminate_noop_nodes(ir_graph: IRGraph) -> IRGraph:
    new_nodes = []
    for node in ir_graph.nodes:
        if node.op in ("aten.dropout", "aten.identity"):
            continue
        new_nodes.append(node)
    ir_graph.nodes = new_nodes
    return ir_graph


def run_ir_pass_pipeline(ir_graph: IRGraph) -> IRGraph:
    ir_graph = eliminate_noop_nodes(ir_graph)
    return ir_graph

