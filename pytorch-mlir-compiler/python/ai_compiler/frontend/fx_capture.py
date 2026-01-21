def capture_fx_graph(model, example_inputs):
    import torch.fx as fx

    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    traced = fx.symbolic_trace(model)
    return traced
