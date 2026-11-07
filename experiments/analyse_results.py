
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    summary_path = METRICS_DIR / "experiment_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")

    summary = pd.read_csv(summary_path)

    run_files = sorted(METRICS_DIR.glob("depth_*_seed_*.csv"))

    if len(run_files) != 24:
        raise ValueError(
            f"Expected 24 experiment files, found {len(run_files)}"
        )

    epoch_data = pd.concat(
        [pd.read_csv(file) for file in run_files],
        ignore_index=True
    )

    return summary, epoch_data


def plot_metric(summary, metric, ylabel, filename):
    grouped = (
        summary.groupby(["depth", "optimizer"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer in ["sgd", "adamw"]:
        data = grouped[grouped["optimizer"] == optimizer]

        ax.errorbar(
            data["depth"],
            data["mean"],
            yerr=data["std"].fillna(0),
            marker="o",
            capsize=5,
            linewidth=2,
            label=optimizer.upper()
        )

    ax.set_xlabel("Recurrent depth")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Recurrent Depth")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def plot_learning_curves(epoch_data):
    data = epoch_data[epoch_data["depth"] == 8]

    fig, ax = plt.subplots(figsize=(9, 5))

    for optimizer in ["sgd", "adamw"]:
        subset = data[data["optimizer"] == optimizer]

        grouped = (
            subset.groupby("epoch")
            .agg(
                train_mean=("train_loss", "mean"),
                train_std=("train_loss", "std"),
                val_mean=("validation_loss", "mean"),
                val_std=("validation_loss", "std")
            )
            .reset_index()
        )

        epochs = grouped["epoch"]

        for metric, std_metric, label, linestyle in [
            ("train_mean", "train_std", "Training", "-"),
            ("val_mean", "val_std", "Validation", "--")
        ]:
            mean = grouped[metric]
            std = grouped[std_metric].fillna(0)

            ax.plot(
                epochs,
                mean,
                linestyle=linestyle,
                linewidth=2,
                label=f"{optimizer.upper()} {label}"
            )

            ax.fill_between(
                epochs,
                mean - std,
                mean + std,
                alpha=0.12
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves — Recurrent Depth 8")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "learning_curves_depth8.png",
        dpi=300
    )
    plt.close(fig)


def main():
    summary, epoch_data = load_results()

    plot_metric(
        summary,
        "final_validation_loss",
        "Final validation loss",
        "validation_loss_vs_depth.png"
    )

    plot_metric(
        summary,
        "final_gradient_norm",
        "Final gradient norm",
        "gradient_norm_vs_depth.png"
    )

    plot_metric(
        summary,
        "total_run_time",
        "Total run time (seconds)",
        "runtime_vs_depth.png"
    )

    plot_learning_curves(epoch_data)

    print("\nAnalysis completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR}")

    for file in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {file.name}")


if __name__ == "__main__":
    main()
