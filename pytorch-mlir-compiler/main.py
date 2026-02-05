import argparse
import sys
from pathlib import Path


def _build_example_model(input_dim: int, hidden_dim: int, output_dim: int):
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return M().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "python"))

    import torch

    torch.manual_seed(args.seed)

    model = _build_example_model(args.input_dim, args.hidden_dim, args.output_dim)
    example_inputs = (torch.randn(args.batch, args.input_dim),)

    with torch.no_grad():
        torch_out = model(*example_inputs)
    print("torch_forward:", {"dtype": str(torch_out.dtype), "shape": list(torch_out.shape)})

    from ai_compiler import compile_torch_model

    try:
        handle = compile_torch_model(model, example_inputs, target=args.target)
        preview = str(handle)
        print("compile_result_type:", type(handle))
        print("compile_result_preview:", preview[:800])
        return 0
    except Exception as e:
        print("compile_failed:", repr(e))

    from ai_compiler.frontend.fx_capture import capture_fx_graph
    from ai_compiler.frontend.ir_builder import build_ir_from_fx
    from ai_compiler.ir.passes import run_ir_pass_pipeline
    from ai_compiler.backend.mlir_export import serialize_ir_graph

    gm = capture_fx_graph(model, example_inputs)
    ir = build_ir_from_fx(gm, example_inputs)
    ir = run_ir_pass_pipeline(ir)
    serialized = serialize_ir_graph(ir)

    print("fallback_fx_nodes:", len(list(gm.graph.nodes)))
    print("fallback_ir_nodes:", len(ir.nodes))
    print("fallback_serialized_ir_keys:", sorted(serialized.keys()))
    print("fallback_ops:", [n["op"] for n in serialized["nodes"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
