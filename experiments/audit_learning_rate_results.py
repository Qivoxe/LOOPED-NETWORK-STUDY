
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "experiment_summary.csv"
)

EXPECTED_SEEDS = {42, 123, 2026}
EXPECTED_OPTIMIZERS = {"sgd", "adamw"}
EXPECTED_DEPTHS = {1, 2, 4, 8}

EXPECTED_LRS = {
    "sgd": {0.001, 0.003, 0.01, 0.03},
    "adamw": {0.0001, 0.0003, 0.001, 0.003},
}

EXPECTED_RUNS = 96

METRIC_COLUMNS = [
    "final_train_loss",
    "final_validation_loss",
    "final_validation_accuracy",
    "final_gradient_norm",
    "total_run_time",
]


def fail(message):
    raise AssertionError(message)


def main():
    print("=" * 60)
    print("LEARNING-RATE EXPERIMENT INTEGRITY AUDIT")
    print("=" * 60)

    if not CSV_PATH.exists():
        fail(f"Summary file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "run_id",
        "seed",
        "depth",
        "optimizer",
        "learning_rate",
        "status",
        *METRIC_COLUMNS,
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(f"Missing columns: {sorted(missing)}")

    # 1. Total number of runs.
    print("\n[1] Checking total runs...")

    if len(df) != EXPECTED_RUNS:
        fail(
            f"Expected {EXPECTED_RUNS} runs, "
            f"found {len(df)}"
        )

    print(f"PASS: Found {len(df)} run records.")

    # 2. Unique run IDs.
    print("\n[2] Checking unique run IDs...")

    if df["run_id"].isna().any():
        fail("Some run IDs are missing.")

    duplicates = df[df["run_id"].duplicated(keep=False)]

    if not duplicates.empty:
        fail(
            "Duplicate run IDs:\n"
            + duplicates["run_id"].to_string(index=False)
        )

    print("PASS: All run IDs are unique.")

    # 3. Expected parameter values.
    print("\n[3] Checking experiment configurations...")

    if set(df["optimizer"].unique()) != EXPECTED_OPTIMIZERS:
        fail("Unexpected optimizer set.")

    if set(df["depth"].unique()) != EXPECTED_DEPTHS:
        fail("Unexpected recurrent depths.")

    if set(df["seed"].unique()) != EXPECTED_SEEDS:
        fail("Unexpected seed set.")

    for optimizer in EXPECTED_OPTIMIZERS:
        actual_lrs = set(
            df.loc[
                df["optimizer"] == optimizer,
                "learning_rate"
            ].unique()
        )

        if actual_lrs != EXPECTED_LRS[optimizer]:
            fail(
                f"Unexpected learning rates for {optimizer}: "
                f"{actual_lrs}"
            )

    print("PASS: Optimizers, depths, seeds and learning rates match.")

    # 4. Every configuration must have exactly three seeds.
    print("\n[4] Checking seed coverage...")

    group_columns = [
        "optimizer",
        "learning_rate",
        "depth",
    ]

    for key, group in df.groupby(group_columns):
        seeds = set(group["seed"])

        if seeds != EXPECTED_SEEDS:
            fail(
                f"Incomplete seed coverage for {key}: {seeds}"
            )

        if len(group) != 3:
            fail(
                f"Expected 3 runs for {key}, "
                f"found {len(group)}"
            )

    print("PASS: All 32 configurations contain 3 seeds.")

    # 5. Status counts.
    print("\n[5] Checking run statuses...")

    status_counts = df["status"].value_counts(dropna=False)

    print(status_counts.to_string())

    if df["status"].isna().any():
        fail("Missing run status detected.")

    completed = df[df["status"] == "completed"]
    failed = df[df["status"] != "completed"]

    if len(completed) != 95:
        fail(
            f"Expected 95 completed runs, "
            f"found {len(completed)}"
        )

    if len(failed) != 1:
        fail(
            f"Expected 1 non-completed run, "
            f"found {len(failed)}"
        )

    print("PASS: 95 completed runs and 1 non-completed run.")

    # 6. Completed runs must have finite metrics.
    print("\n[6] Checking completed-run metrics...")

    for column in METRIC_COLUMNS:
        values = pd.to_numeric(
            completed[column],
            errors="coerce"
        )

        if values.isna().any():
            fail(
                f"Missing or non-numeric values in "
                f"{column} for completed runs."
            )

        if not np.isfinite(values.to_numpy()).all():
            fail(
                f"Non-finite values found in {column} "
                f"for completed runs."
            )

    print("PASS: All completed-run metrics are finite.")

    # 7. Check failed-run identity.
    print("\n[7] Inspecting non-completed run...")

    failed_row = failed.iloc[0]

    print(
        failed_row[
            [
                "run_id",
                "optimizer",
                "learning_rate",
                "depth",
                "seed",
                "status",
                "final_validation_loss",
            ]
        ].to_string()
    )

    expected_failed_run = (
        failed_row["optimizer"] == "sgd"
        and failed_row["learning_rate"] == 0.03
        and failed_row["depth"] == 8
        and failed_row["seed"] == 2026
    )

    if not expected_failed_run:
        fail("The non-completed run does not match the expected run.")

    print("PASS: The expected unstable configuration is identified.")

    # 8. Final report.
    print("\n" + "=" * 60)
    print("AUDIT PASSED")
    print("=" * 60)

    print(f"Total runs:       {len(df)}")
    print(f"Completed runs:   {len(completed)}")
    print(f"Non-completed:    {len(failed)}")
    print(f"Configurations:   {len(df.groupby(group_columns))}")
    print("Data integrity checks: PASS")


if __name__ == "__main__":
    main()
