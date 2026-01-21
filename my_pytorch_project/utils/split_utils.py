from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: list[int]
    val: list[int]
    test: list[int]


def make_split(n: int, *, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 0) -> Split:
    if n <= 0:
        raise ValueError("n must be positive")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("invalid ratios")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_n = int(round(n * test_ratio))
    val_n = int(round(n * val_ratio))
    train_n = n - val_n - test_n

    train = indices[:train_n]
    val = indices[train_n : train_n + val_n]
    test = indices[train_n + val_n :]
    return Split(train=train, val=val, test=test)
