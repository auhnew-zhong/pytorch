from typing import Any


import unittest
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.simple_nn import SimpleNN


class TestSimpleNN(unittest.TestCase):
    def test_forward_shape(self):
        model = SimpleNN(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(4, 2)
        y = model(x)
        self.assertEqual(tuple[Any, ...](y.shape), (4, 1))


if __name__ == "__main__":
    unittest.main()
