from .frontend.fx_capture import capture_fx_graph
from .frontend.ir_builder import build_ir_from_fx
from .ir.passes import run_ir_pass_pipeline
from .backend.mlir_export import ir_to_mlir_and_compile


def compile_torch_model(model, example_inputs, target: str = "cpu"):
    graph_module = capture_fx_graph(model, example_inputs)
    ir_graph = build_ir_from_fx(graph_module, example_inputs)
    ir_graph = run_ir_pass_pipeline(ir_graph)
    handle = ir_to_mlir_and_compile(ir_graph, target=target)
    return handle

