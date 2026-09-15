"""
CRITICAL FIX: label leakage.

The original label was thresholding `returns_1d`, which is ALSO one of
the 18 technical features - meaning the label was derivable directly
from a feature the model can see, rather than representing something to
be predicted. This shifts the label to represent NEXT-DAY movement
(the standard, correct setup: predict tomorrow's UP/FLAT/DOWN using
only information known as of today's close).

Applies the fix to all three already-built processed CSVs in place: features_technical_only.csv, features_technical_3yr.csv,
features_full.csv.

Usage:
    python fix_label_leakage.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")

FILES = [
    "features_technical_only.csv",
    "features_technical_3yr.csv",
    "features_full.csv",
]


def fix_file(filename):
    path = os.path.join(DATA_PROCESSED, filename)
    if not os.path.exists(path):
        print(f"[{filename}] NOT FOUND, skipping.")
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    before = len(df)

    # The forward-looking label for day t = the old (same-day) label
    # computed at day t+1, shifted backward by one row WITHIN each
    # ticker's chronological sequence.
    df["label"] = df.groupby("ticker")["label"].shift(-1)

    # Drop the last day per ticker (now has no valid forward label)
    df = df.dropna(subset=["label"])
    after = len(df)

    df.to_csv(path, index=False)

    print(f"[{filename}] relabeled: {before} -> {after} rows "
          f"({before - after} last-day-per-ticker rows dropped, expected ~15).")

    balance = df["label"].value_counts(normalize=True) * 100
    print(f"  New class balance: {dict(balance.round(1))}")
    print(f"  New split sizes: {dict(df['split'].value_counts())}\n")


def main():
    print("Fixing label leakage across all processed feature files...\n")
    for f in FILES:
        fix_file(f)
    print("Done. Labels now represent NEXT-DAY movement (correct, "
          "leakage-free setup). Re-verify class balance/split sizes above "
          "before proceeding to model training.")


if __name__ == "__main__":
    main()
