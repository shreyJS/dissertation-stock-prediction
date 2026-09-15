"""
One-off repair for the yfinance CSV header-artifact bug: cleans any
already-saved {TICKER}_ohlcv.csv files in place, coercing price/volume
columns to numeric and dropping rows that don't survive (the corrupted
header-artifact rows).

Usage:
    python repair_ohlcv_csvs.py
"""

import os
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")


def main():
    for symbol in TICKERS:
        path = os.path.join(DATA_RAW, f"{symbol}_ohlcv.csv")
        if not os.path.exists(path):
            print(f"[{symbol}] file not found, skipping.")
            continue

        df = pd.read_csv(path)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        numeric_cols = [c for c in ["Open", "High",
                                    "Low", "Close", "Volume"] if c in df.columns]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        before = len(df)
        df = df.dropna(subset=numeric_cols + [date_col])
        after = len(df)

        df.to_csv(path, index=False)
        status = f"cleaned {before - after} bad row(s)" if before != after else "already clean"
        print(f"[{symbol}] {status}, {after} rows saved.")


if __name__ == "__main__":
    main()
