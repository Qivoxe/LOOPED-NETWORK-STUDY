
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "experiment_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load individual experiment records.
df = pd.read_csv(SUMMARY_PATH)

required_columns = {
    "optimizer",
    "learning_rate",
    "depth",
    "status",
    "final_validation_loss",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Separate successful and failed runs.
completed = df[df["status"] == "completed"].copy()

# Calculate statistics directly from individual runs.
summary = (
    df.groupby(
        ["optimizer", "learning_rate", "depth"],
        as_index=False
    )
    .agg(
        total_runs=("status", "size"),
        completed_runs=(
            "status",
            lambda values: (values == "completed").sum()
        ),
        failed_runs=(
            "status",
            lambda values: (values != "completed").sum()
        ),
    )
)

# Calculate mean validation loss using completed runs only.
means = (
    completed.groupby(
        ["optimizer", "learning_rate", "depth"]
    )["final_validation_loss"]
    .mean()
    .reset_index(name="mean_validation_loss")
)

summary = summary.merge(
    means,
    on=["optimizer", "learning_rate", "depth"],
    how="left",
)

# Generate one heatmap per optimizer.
for optimizer in ["sgd", "adamw"]:

    subset = summary[
        summary["optimizer"] == optimizer
    ].copy()

    loss_matrix = subset.pivot(
        index="learning_rate",
        columns="depth",
        values="mean_validation_loss",
    )

    completed_matrix = subset.pivot(
        index="learning_rate",
        columns="depth",
        values="completed_runs",
    )

    total_matrix = subset.pivot(
        index="learning_rate",
        columns="depth",
        values="total_runs",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(
        loss_matrix.values,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(range(len(loss_matrix.columns)))
    ax.set_xticklabels(loss_matrix.columns)

    ax.set_yticks(range(len(loss_matrix.index)))
    ax.set_yticklabels(loss_matrix.index)

    ax.set_xlabel("Recurrent depth")
    ax.set_ylabel("Learning rate")
    ax.set_title(
        f"{optimizer.upper()} — Validation Loss"
    )

    # Annotate each cell with loss and completed-run count.
    for row in range(loss_matrix.shape[0]):
        for col in range(loss_matrix.shape[1]):

            loss = loss_matrix.iloc[row, col]
            successful = int(completed_matrix.iloc[row, col])
            total = int(total_matrix.iloc[row, col])

            if pd.notna(loss):
                label = (
                    f"{loss:.4f}\n"
                    f"n={successful}/{total}"
                )
            else:
                label = (
                    f"No valid mean\n"
                    f"n={successful}/{total}"
                )

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean validation loss")

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{optimizer}_learning_rate_heatmap.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nHeatmaps generated successfully.")

# Print incomplete configurations for verification.
incomplete = summary[summary["completed_runs"] < summary["total_runs"]]

print("\nIncomplete configurations:")
if incomplete.empty:
    print("None")
else:
    print(
        incomplete[
            [
                "optimizer",
                "learning_rate",
                "depth",
                "completed_runs",
                "total_runs",
                "failed_runs",
            ]
        ].to_string(index=False)
    )
