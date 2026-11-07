
import yaml
from pathlib import Path


def load_experiment_config():

    path = Path("configs/experiment_matrix.yaml")

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    seeds = config["experiment"]["seeds"]
    depths = config["recurrent_depths"]
    optimizers = config["optimizers"]

    assert len(seeds) == 3
    assert len(depths) == 4

    assert set(optimizers) == {"sgd", "adamw"}

    return config


if __name__ == "__main__":

    config = load_experiment_config()

    total_runs = (
        len(config["experiment"]["seeds"])
        * len(config["recurrent_depths"])
        * len(config["optimizers"])
    )

    print("Experiment:", config["experiment"]["name"])
    print("Seeds:", config["experiment"]["seeds"])
    print("Depths:", config["recurrent_depths"])
    print("Optimizers:", config["optimizers"])
    print("Total runs:", total_runs)
