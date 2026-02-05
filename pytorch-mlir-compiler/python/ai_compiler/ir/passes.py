from ai_compiler.ir.nodes import IRGraph


def eliminate_noop_nodes(ir_graph: IRGraph) -> IRGraph:
    """Drop operators that do not affect semantics for this demo pipeline."""
    new_nodes = []
    for node in ir_graph.nodes:
        # No-op nodes are pruned to keep the IR compact.
        if node.op in ("aten.dropout", "aten.identity"):
            continue
        new_nodes.append(node)
    ir_graph.nodes = new_nodes
    return ir_graph


def run_ir_pass_pipeline(ir_graph: IRGraph) -> IRGraph:
    """Run all IR-level optimization passes in order."""
    ir_graph = eliminate_noop_nodes(ir_graph)
    return ir_graph
