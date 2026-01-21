from torch._tensor import Tensor


from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.simple_nn import SimpleNN
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.data_utils import make_tensor_dataset


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
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save", default="checkpoints/last.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))

    device = torch.device(config.get("device", "cpu"))

    x = torch.randn(64, int(config["input_dim"]))
    y = torch.randn(64, int(config["output_dim"]))
    dataset = make_tensor_dataset(x, y)
    loader = DataLoader[tuple[Tensor, ...]](dataset, batch_size=int(config["batch_size"]), shuffle=True)

    model = SimpleNN(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))

    epochs = int(config["epochs"])
    for epoch in range(start_epoch, epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1}/{epochs} loss={loss.item():.6f}")

        if args.save:
            save_checkpoint(
                args.save,
                model,
                optimizer,
                epoch=epoch + 1,
                config=config,
            )


if __name__ == "__main__":
    main()
