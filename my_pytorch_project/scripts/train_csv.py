from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.simple_nn import SimpleNN
from utils.checkpointing import save_checkpoint
from utils.csv_dataset import CSVDataset
from utils.split_utils import make_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/csv_example.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))

    root = Path(__file__).resolve().parents[1]
    csv_path = root / config["csv_path"]
    dataset = CSVDataset(csv_path, x_cols=list(config["x_cols"]), y_col=str(config["y_col"]))

    split = make_split(len(dataset), val_ratio=float(config["val_ratio"]), test_ratio=0.0, seed=int(config["seed"]))
    train_set = Subset(dataset, split.train)
    val_set = Subset(dataset, split.val)

    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(config["batch_size"]), shuffle=False)

    device = torch.device(config.get("device", "cpu"))
    model = SimpleNN(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    for epoch in range(int(config["epochs"])):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch_x.shape[0]
            train_n += int(batch_x.shape[0])

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += float(loss.item()) * batch_x.shape[0]
                val_n += int(batch_x.shape[0])

        print(
            f"epoch={epoch + 1}/{int(config['epochs'])} "
            f"train_mse={train_loss / max(train_n, 1):.6f} "
            f"val_mse={val_loss / max(val_n, 1):.6f}"
        )

    save_checkpoint(str(root / "checkpoints" / "csv_last.pt"), model, optimizer, epoch=int(config["epochs"]), config=config)


if __name__ == "__main__":
    main()
