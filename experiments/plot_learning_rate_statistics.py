
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "statistical_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "learning_rate_sensitivity"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

optimizers = ["sgd", "adamw"]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

metrics = [
    (
        "mean_validation_loss",
        "std_validation_loss",
        "Mean validation loss",
    ),
    (
        "mean_gradient_norm",
        "std_gradient_norm",
        "Mean gradient norm",
    ),
]

for col, optimizer in enumerate(optimizers):

    subset = df[df["optimizer"] == optimizer]

    for row, (mean_col, std_col, title) in enumerate(metrics):

        ax = axes[row, col]

        for learning_rate in sorted(
            subset["learning_rate"].unique()
        ):

            group = subset[
                subset["learning_rate"] == learning_rate
            ].sort_values("depth")

            ax.errorbar(
                group["depth"],
                group[mean_col],
                yerr=group[std_col],
                marker="o",
                capsize=4,
                label=f"LR={learning_rate:g}",
            )

            # Explicitly identify configurations with failed runs.
            incomplete = group[
                group["successful_runs"] < group["total_runs"]
]

            for _, point in incomplete.iterrows():
                ax.annotate(
                    (
                        f"n={int(point['successful_runs'])}/"
                        f"{int(point['total_runs'])}"
                    ),
                    (
                        point["depth"],
                        point[mean_col],
                    ),
                    xytext=(7, 10),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_title(f"{optimizer.upper()} — {title}")
        ax.set_xlabel("Recurrent depth")
        ax.set_ylabel(title)
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle(
    "Learning-rate sensitivity across recurrent depths",
    fontsize=15,
)

fig.tight_layout()

output_path = (
    OUTPUT_DIR
    / "learning_rate_depth_statistics.png"
)

fig.savefig(output_path, dpi=200)
plt.close(fig)

print(f"Saved: {output_path}")
print("Statistical plots generated successfully.")
