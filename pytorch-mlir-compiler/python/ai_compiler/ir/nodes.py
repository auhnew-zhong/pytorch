from typing import List, Dict, Any, Optional


class IRValue:
    def __init__(self, name: str, dtype: str, shape: Optional[List[int]]):
        self.name = name
        self.dtype = dtype
        self.shape = shape


class IRNode:
    def __init__(self, op: str, inputs: List[IRValue], outputs: List[IRValue], attrs: Dict[str, Any]):
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


class IRGraph:
    def __init__(self):
        self.inputs: List[IRValue] = []
        self.outputs: List[IRValue] = []
        self.nodes: List[IRNode] = []

    def add_node(self, node: IRNode):
        self.nodes.append(node)

    def add_input(self, value: IRValue):
        self.inputs.append(value)

    def add_output(self, value: IRValue):
        self.outputs.append(value)

