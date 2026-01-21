import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def binary_accuracy(logits, target):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(target.dtype)
    return torch.mean((preds == target).to(torch.float32))
