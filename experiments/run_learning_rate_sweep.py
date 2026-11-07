
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.looped_network import LoopedNetwork
from src.optimizers import SGDMomentum, AdamWCustom
from src.training.trainer import Trainer
from src.training.reproducibility import set_seed

from experiments.run_experiments import (
    prepare_dataset,
    create_loaders,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "learning_rate_experiment.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def create_optimizer(name, model, config, learning_rate):
    settings = config["optimizer_settings"][name]

    if name == "sgd":
        return SGDMomentum(
            model.parameters(),
            lr=learning_rate,
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )

    if name == "adamw":
        return AdamWCustom(
            model.parameters(),
            lr=learning_rate,
            weight_decay=settings["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer: {name}")


def format_lr(value):
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def run_experiment(config, dataset, seed, depth, optimizer_name, lr):
    set_seed(seed)

    model_config = config["model"].copy()
    model_config["num_steps"] = depth

    model = LoopedNetwork(**model_config)

    optimizer = create_optimizer(
        optimizer_name,
        model,
        config,
        lr,
    )

    train_loader, val_loader = create_loaders(
        *dataset,
        config["training"]["batch_size"],
        seed,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config.get("device", "cpu"),
    )

    start = time.perf_counter()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
    )

    elapsed = time.perf_counter() - start

    run_id = (
        f"depth_{depth}_{optimizer_name}_"
        f"lr_{format_lr(lr)}_seed_{seed}"
    )

    history_df = pd.DataFrame(history)

    history_df.insert(0, "run_id", run_id)
    history_df.insert(1, "seed", seed)
    history_df.insert(2, "depth", depth)
    history_df.insert(3, "optimizer", optimizer_name)
    history_df.insert(4, "learning_rate", lr)
    history_df["total_run_time"] = elapsed

    output_dir = (
        ROOT / config["output"]["directory"]
    )

    runs_dir = output_dir / "individual_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(
        runs_dir / f"{run_id}.csv",
        index=False,
    )

    metric_columns = [
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "gradient_norm",
    ]

    finite = np.isfinite(
        history_df[metric_columns].to_numpy(dtype=float)
    ).all()

    final = history_df.iloc[-1]

    return {
        "run_id": run_id,
        "seed": seed,
        "depth": depth,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "status": "completed" if finite else "non_finite",
        "final_train_loss": final["train_loss"],
        "final_validation_loss": final["validation_loss"],
        "final_validation_accuracy": final["validation_accuracy"],
        "final_gradient_norm": final["gradient_norm"],
        "total_run_time": elapsed,
    }


def main():
    config = load_config()

    seeds = config["experiment"]["seeds"]
    depths = config["recurrent_depths"]
    optimizers = config["optimizers"]

    for optimizer in optimizers:
        rates = config["learning_rates"][optimizer]

        if not rates or any(rate <= 0 for rate in rates):
            raise ValueError(
                f"Invalid learning rates for {optimizer}: {rates}"
            )

    dataset = prepare_dataset(config)

    output_dir = ROOT / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(
        len(config["learning_rates"][optimizer])
        * len(depths)
        * len(seeds)
        for optimizer in optimizers
    )

    print("=" * 65)
    print("LEARNING-RATE SENSITIVITY EXPERIMENT")
    print("=" * 65)
    print(f"Total runs: {total_runs}")
    print(f"Depths: {depths}")
    print(f"Seeds: {seeds}")

    results = []

    for optimizer_name in optimizers:
        for lr in config["learning_rates"][optimizer_name]:
            for depth in depths:
                for seed in seeds:

                    run_id = (
                        f"depth_{depth}_{optimizer_name}_"
                        f"lr_{format_lr(lr)}_seed_{seed}"
                    )

                    print("\n" + "-" * 65)
                    print(f"Running: {run_id}")

                    try:
                        result = run_experiment(
                            config=config,
                            dataset=dataset,
                            seed=seed,
                            depth=depth,
                            optimizer_name=optimizer_name,
                            lr=lr,
                        )

                    except Exception as exc:
                        result = {
                            "run_id": run_id,
                            "seed": seed,
                            "depth": depth,
                            "optimizer": optimizer_name,
                            "learning_rate": lr,
                            "status": "failed",
                            "error": repr(exc),
                        }

                        print(f"FAILED: {exc}")

                    results.append(result)

                    if result["status"] != "failed":
                        print(
                            f"Status: {result['status']} | "
                            f"Validation loss: "
                            f"{result['final_validation_loss']:.4f}"
                        )

    summary = pd.DataFrame(results)

    summary.to_csv(
        output_dir / "experiment_summary.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT SWEEP FINISHED")
    print("=" * 65)
    print(f"Results: {len(results)}/{total_runs}")
    print(summary["status"].value_counts().to_string())
    print(f"Summary: {output_dir / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
