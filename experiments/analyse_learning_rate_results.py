
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "experiment_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Keep only successfully completed runs for performance statistics.
completed = df[df["status"] == "completed"].copy()

metric_columns = [
    "final_train_loss",
    "final_validation_loss",
    "final_validation_accuracy",
    "final_gradient_norm",
    "total_run_time",
]

for column in metric_columns:
    completed[column] = pd.to_numeric(
        completed[column],
        errors="coerce"
    )

    if not np.isfinite(completed[column]).all():
        raise ValueError(
            f"Invalid metric values in completed runs: {column}"
        )

group_columns = [
    "optimizer",
    "learning_rate",
    "depth",
]

# Performance statistics from successful runs.
performance = (
    completed.groupby(group_columns)
    .agg(
        successful_runs=("run_id", "count"),

        mean_train_loss=("final_train_loss", "mean"),
        std_train_loss=("final_train_loss", "std"),

        mean_validation_loss=(
            "final_validation_loss", "mean"
        ),
        std_validation_loss=(
            "final_validation_loss", "std"
        ),

        mean_validation_accuracy=(
            "final_validation_accuracy", "mean"
        ),
        std_validation_accuracy=(
            "final_validation_accuracy", "std"
        ),

        mean_gradient_norm=(
            "final_gradient_norm", "mean"
        ),
        std_gradient_norm=(
            "final_gradient_norm", "std"
        ),

        mean_runtime=("total_run_time", "mean"),
        std_runtime=("total_run_time", "std"),
    )
    .reset_index()
)

# Failure statistics from all runs.
failure_stats = (
    df.assign(
        failed=(df["status"] != "completed").astype(int)
    )
    .groupby(group_columns)
    .agg(
        total_runs=("run_id", "count"),
        failed_runs=("failed", "sum"),
    )
    .reset_index()
)

failure_stats["failure_rate"] = (
    failure_stats["failed_runs"]
    / failure_stats["total_runs"]
)

# Combine performance and reliability statistics.
summary = performance.merge(
    failure_stats,
    on=group_columns,
    how="outer",
)

summary = summary.sort_values(
    ["optimizer", "learning_rate", "depth"]
)

output_path = OUTPUT_DIR / "statistical_summary.csv"

summary.to_csv(output_path, index=False)

print("=" * 75)
print("LEARNING-RATE STATISTICAL ANALYSIS")
print("=" * 75)

print("\nConfiguration-level statistics:")

display_columns = [
    "optimizer",
    "learning_rate",
    "depth",
    "successful_runs",
    "total_runs",
    "mean_validation_loss",
    "std_validation_loss",
    "mean_validation_accuracy",
    "mean_gradient_norm",
    "failed_runs",
    "failure_rate",
]

print(
    summary[display_columns]
    .round(5)
    .to_string(index=False)
)

print("\nOverall run statistics:")
print(f"Total runs: {len(df)}")
print(f"Successful runs: {len(completed)}")
print(f"Failed runs: {len(df) - len(completed)}")

print("\nFailure rate:")
print(
    f"{(len(df) - len(completed)) / len(df) * 100:.2f}%"
)

print("\nSaved statistical summary:")
print(output_path)
