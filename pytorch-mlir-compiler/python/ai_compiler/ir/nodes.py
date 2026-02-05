from typing import List, Dict, Any, Optional


class IRValue:
    """A typed value in the compiler IR."""
    def __init__(self, name: str, dtype: str, shape: Optional[List[int]]):
        self.name = name
        self.dtype = dtype
        self.shape = shape


class IRNode:
    """An operation node with inputs, outputs, and attributes."""
    def __init__(self, op: str, inputs: List[IRValue], outputs: List[IRValue], attrs: Dict[str, Any]):
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


class IRGraph:
    """A directed acyclic graph of IRNodes with explicit inputs/outputs."""
    def __init__(self):
        self.inputs: List[IRValue] = []
        self.outputs: List[IRValue] = []
        self.nodes: List[IRNode] = []

    def add_node(self, node: IRNode):
        # Append node in execution order.
        self.nodes.append(node)

    def add_input(self, value: IRValue):
        # Register a graph input value.
        self.inputs.append(value)

    def add_output(self, value: IRValue):
        # Register a graph output value.
        self.outputs.append(value)
