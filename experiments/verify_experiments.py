
import pandas as pd
from pathlib import Path

directory = Path("results/metrics")

summary_path = directory / "experiment_summary.csv"

summary = pd.read_csv(summary_path)

print("EXPERIMENT SUMMARY")
print("=" * 50)

print("Total runs:", len(summary))
print("\nRun status:")
print(summary["status"].value_counts())

print("\nResults by optimizer:")

print(
    summary.groupby("optimizer")[
        ["final_validation_loss", "final_validation_accuracy"]
    ].mean()
)

print("\nResults by depth:")

print(
    summary.groupby("depth")[
        ["final_validation_loss", "final_validation_accuracy"]
    ].mean()
)

assert len(summary) == 24
assert (summary["status"] == "completed").all()

print("\nAll experiment records verified.")
