
import time
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


def load_config():
    with open("configs/experiment_matrix.yaml", "r") as file:
        return yaml.safe_load(file)


def prepare_dataset(config):
    """Create one fixed dataset and split for all experiments."""

    dataset_config = config["dataset"]

    X, y = make_moons(
        n_samples=dataset_config["samples"],
        noise=dataset_config["noise"],
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=dataset_config["validation_fraction"],
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val


def create_loaders(X_train, X_val, y_train, y_val, batch_size, seed):

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def create_optimizer(name, model, config):

    settings = config["optimizer_settings"][name]

    if name == "sgd":
        return SGDMomentum(
            model.parameters(),
            lr=settings["learning_rate"],
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )

    if name == "adamw":
        return AdamWCustom(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer: {name}")


def run_experiment(
    config,
    seed,
    depth,
    optimizer_name,
    dataset,
):

    X_train, X_val, y_train, y_val = dataset

    # Reinitialize RNG before every model initialization.
    set_seed(seed)

    model_config = config["model"].copy()
    model_config["num_steps"] = depth

    model = LoopedNetwork(**model_config)

    optimizer = create_optimizer(
        optimizer_name,
        model,
        config,
    )

    train_loader, val_loader = create_loaders(
        X_train,
        X_val,
        y_train,
        y_val,
        config["training"]["batch_size"],
        seed,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.get("device", "cpu"),
    )

    start_time = time.perf_counter()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
    )

    total_time = time.perf_counter() - start_time

    run_id = f"depth_{depth}_{optimizer_name}_seed_{seed}"

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)

    history_df.insert(0, "run_id", run_id)
    history_df.insert(1, "seed", seed)
    history_df.insert(2, "depth", depth)
    history_df.insert(3, "optimizer", optimizer_name)

    history_df["total_run_time"] = total_time

    history_df.to_csv(
        output_dir / f"{run_id}.csv",
        index=False,
    )

    final = history[-1]

    return {
        "run_id": run_id,
        "seed": seed,
        "depth": depth,
        "optimizer": optimizer_name,
        "status": "completed",
        "final_train_loss": final["train_loss"],
        "final_validation_loss": final["validation_loss"],
        "final_validation_accuracy": final["validation_accuracy"],
        "final_gradient_norm": final["gradient_norm"],
        "total_run_time": total_time,
    }


def main():

    config = load_config()

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(config)

    seeds = config["experiment"]["seeds"]
    depths = config["recurrent_depths"]
    optimizers = config["optimizers"]

    total_runs = len(seeds) * len(depths) * len(optimizers)

    print("=" * 60)
    print("LOOPED NETWORK EXPERIMENT EXECUTION")
    print("=" * 60)

    print(f"Total experiments: {total_runs}")
    print(f"Device: {config.get('device', 'cpu')}")

    results = []

    for depth in depths:

        for optimizer_name in optimizers:

            for seed in seeds:

                run_id = f"depth_{depth}_{optimizer_name}_seed_{seed}"

                print("\n" + "-" * 60)
                print(f"Running: {run_id}")

                result = run_experiment(
                    config=config,
                    seed=seed,
                    depth=depth,
                    optimizer_name=optimizer_name,
                    dataset=dataset,
                )

                results.append(result)

                print(
                    f"Validation loss: "
                    f"{result['final_validation_loss']:.4f}"
                )

                print(
                    f"Validation accuracy: "
                    f"{result['final_validation_accuracy']:.4f}"
                )

    summary = pd.DataFrame(results)

    summary.to_csv(
        output_dir / "experiment_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Completed runs: {len(results)}/{total_runs}")
    print(f"Summary saved to: {output_dir / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
