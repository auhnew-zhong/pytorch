def capture_fx_graph(model, example_inputs):
    """Trace a PyTorch module into an FX GraphModule and populate shape metadata."""
    import torch
    import torch.fx as fx
    from torch.fx.passes.shape_prop import ShapeProp

    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    # Build a symbolic graph from the eager module.
    traced = fx.symbolic_trace(model)
    # Run in eval mode to keep tracing stable and avoid training-time randomness.
    traced.eval()
    # Propagate dtype/shape so later IR building can read node.meta["tensor_meta"].
    with torch.no_grad():
        ShapeProp(traced).propagate(*example_inputs)
    return traced
