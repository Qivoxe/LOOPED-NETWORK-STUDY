
import yaml
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.looped_network import LoopedNetwork
from src.optimizers import SGDMomentum, AdamWCustom
from src.training.trainer import Trainer
from src.training.reproducibility import set_seed


def main():

    config_path = Path("configs/baseline.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])

    # Synthetic data: pipeline validation only.
    X, y = make_moons(
        n_samples=config["dataset"]["samples"],
        noise=0.2,
        random_state=config["seed"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["dataset"]["validation_fraction"],
        random_state=config["seed"],
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    def create_loader(X, y, shuffle):

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

        return DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=shuffle,
        )

    train_loader = create_loader(X_train, y_train, True)
    validation_loader = create_loader(X_val, y_val, False)

    model = LoopedNetwork(**config["model"])

    optimizer_config = config["optimizer"]

    if optimizer_config["name"] == "sgd":

        optimizer = SGDMomentum(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            momentum=0.9,
            weight_decay=optimizer_config["weight_decay"],
        )

    elif optimizer_config["name"] == "adamw":

        optimizer = AdamWCustom(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )

    else:
        raise ValueError("Unsupported optimizer")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config["device"],
    )

    history = trainer.fit(
        train_loader,
        validation_loader,
        epochs=config["training"]["epochs"],
    )

    output_path = Path("results/metrics/baseline.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(output_path, index=False)

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
