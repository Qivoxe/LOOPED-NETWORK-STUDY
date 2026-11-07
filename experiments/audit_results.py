
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"

EXPECTED_DEPTHS = [1, 2, 4, 8]
EXPECTED_OPTIMIZERS = ["sgd", "adamw"]
EXPECTED_SEEDS = [42, 123, 2026]
EXPECTED_EPOCHS = 15

METRICS_TO_CHECK = [
    "train_loss",
    "validation_loss",
    "train_accuracy",
    "validation_accuracy",
    "gradient_norm",
    "epoch_time",
]


def main():
    summary_path = METRICS / "experiment_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    files = sorted(METRICS.glob("depth_*_seed_*.csv"))

    errors = []

    expected_runs = {
        (depth, optimizer, seed)
        for depth in EXPECTED_DEPTHS
        for optimizer in EXPECTED_OPTIMIZERS
        for seed in EXPECTED_SEEDS
    }

    observed_runs = set()

    print("Starting experiment integrity audit...\n")

    if len(files) != 24:
        errors.append(f"Expected 24 CSV files, found {len(files)}")

    if len(summary) != 24:
        errors.append(
            f"Expected 24 summary rows, found {len(summary)}"
        )

    for file in files:
        try:
            df = pd.read_csv(file)

            if df.empty:
                errors.append(f"{file.name}: empty CSV")
                continue

            required_columns = [
                "run_id", "seed", "depth", "optimizer",
                "epoch", *METRICS_TO_CHECK
            ]

            missing = set(required_columns) - set(df.columns)

            if missing:
                errors.append(
                    f"{file.name}: missing columns {sorted(missing)}"
                )
                continue

            metadata = df.iloc[0]

            depth = int(metadata["depth"])
            optimizer = str(metadata["optimizer"])
            seed = int(metadata["seed"])

            key = (depth, optimizer, seed)
            observed_runs.add(key)

            if key not in expected_runs:
                errors.append(f"{file.name}: unexpected configuration")

            if len(df) != EXPECTED_EPOCHS:
                errors.append(
                    f"{file.name}: expected {EXPECTED_EPOCHS} epochs, "
                    f"found {len(df)}"
                )

            if not df["epoch"].tolist() == list(
                range(1, EXPECTED_EPOCHS + 1)
            ):
                errors.append(f"{file.name}: incorrect epoch sequence")

            for column in METRICS_TO_CHECK:
                values = pd.to_numeric(df[column], errors="coerce")

                if not np.isfinite(values).all():
                    errors.append(
                        f"{file.name}: non-finite values in {column}"
                    )

            for column in ["seed", "depth", "optimizer", "run_id"]:
                if df[column].nunique() != 1:
                    errors.append(
                        f"{file.name}: inconsistent {column}"
                    )

            if df["train_loss"].lt(0).any():
                errors.append(f"{file.name}: negative training loss")

            if df["validation_loss"].lt(0).any():
                errors.append(f"{file.name}: negative validation loss")

        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    missing_runs = expected_runs - observed_runs
    duplicate_runs = len(observed_runs) != len(files)

    if missing_runs:
        errors.append(f"Missing configurations: {sorted(missing_runs)}")

    if duplicate_runs:
        errors.append("Duplicate experiment configurations detected")

    if not errors:
        print("PASS: All 24 experiment files are structurally valid.")
        print("PASS: All expected configurations are present.")
        print("PASS: All runs contain 15 epochs.")
        print("PASS: Metric values are finite.")
        print("\nExperiment integrity audit completed.")
    else:
        print("AUDIT FAILED:\n")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
