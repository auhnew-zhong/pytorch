import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.simple_nn import SimpleNN


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
    model = SimpleNN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
    )
    model.eval()

    x = torch.randn(1, config["input_dim"])
    with torch.no_grad():
        pred = model(x)
    print(pred)


if __name__ == "__main__":
    main()
