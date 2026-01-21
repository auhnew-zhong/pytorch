from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, path: str | Path, *, x_cols: list[str], y_col: str, dtype=torch.float32):
        self.path = Path(path)
        self.x_cols = x_cols
        self.y_col = y_col
        self.dtype = dtype

        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"empty csv: {self.path}")

        x = []
        y = []
        for row in rows:
            x.append([float(row[c]) for c in x_cols])
            y.append([float(row[y_col])])

        self.x = torch.tensor(x, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
