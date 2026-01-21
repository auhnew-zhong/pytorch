from torch._tensor import Tensor


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
from utils.data_utils import make_tensor_dataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"]) 

    x = torch.randn(32, config["input_dim"])
    y = torch.randn(32, config["output_dim"])
    dataset = make_tensor_dataset(x, y)
    loader = DataLoader[tuple[Tensor, ...]](dataset, batch_size=config["batch_size"], shuffle=True)

    model = SimpleNN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    losses = []
    for _ in range(config["epochs"]):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"train_loss={loss.item():.6f}")

    print(f"final_loss={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
