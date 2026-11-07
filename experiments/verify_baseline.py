
import pandas as pd
import numpy as np
from pathlib import Path

path = Path("results/metrics/baseline.csv")

if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_csv(path)

required_columns = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "gradient_norm",
    "epoch_time",
]

assert all(col in df.columns for col in required_columns)
assert len(df) > 0

numeric = df[required_columns].to_numpy(dtype=float)

assert np.isfinite(numeric).all(), "Non-finite metric detected"

assert (df["epoch_time"] >= 0).all()
assert (df["gradient_norm"] >= 0).all()

print("\nBASELINE VERIFICATION")
print("=" * 40)

print(df.tail())

print("\nInitial training loss:", df["train_loss"].iloc[0])
print("Final training loss:", df["train_loss"].iloc[-1])

print("Initial validation loss:", df["validation_loss"].iloc[0])
print("Final validation loss:", df["validation_loss"].iloc[-1])

print("\nAverage epoch time:", df["epoch_time"].mean())

print("\nCSV verification successful.")
