from ai_compiler.ir.nodes import IRGraph

try:
    import _ai_compiler_mlir as _mlir_backend
except ImportError:
    _mlir_backend = None


def ir_to_mlir_and_compile(ir_graph: IRGraph, target: str = "cpu"):
    """Serialize the IR graph and invoke the C++ MLIR backend."""
    if _mlir_backend is None:
        raise RuntimeError("MLIR backend extension is not available")
    # Convert Python IR objects into a pure-Python dict for pybind transfer.
    serialized = serialize_ir_graph(ir_graph)
    handle = _mlir_backend.compile_from_serialized_ir(serialized, target)
    return handle


def serialize_ir_graph(ir_graph: IRGraph) -> dict:
    """Serialize IRGraph into a dict schema consumable by the C++ side."""
    data = {
        "inputs": [],
        "outputs": [],
        "nodes": [],
    }
    for v in ir_graph.inputs:
        data["inputs"].append({"name": v.name, "dtype": v.dtype, "shape": v.shape})
    for v in ir_graph.outputs:
        data["outputs"].append({"name": v.name, "dtype": v.dtype, "shape": v.shape})
    for n in ir_graph.nodes:
        data["nodes"].append(
            {
                "op": n.op,
                "inputs": [v.name for v in n.inputs],
                "outputs": [v.name for v in n.outputs],
                "attrs": n.attrs,
            }
        )
    return data
