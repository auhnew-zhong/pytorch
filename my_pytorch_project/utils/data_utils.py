import torch
from torch.utils.data import TensorDataset


def make_tensor_dataset(features, targets, dtype=torch.float32):
    x = features if torch.is_tensor(features) else torch.tensor(features, dtype=dtype)
    y = targets if torch.is_tensor(targets) else torch.tensor(targets, dtype=dtype)
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    if y.dtype != dtype:
        y = y.to(dtype=dtype)
    return TensorDataset(x, y)
