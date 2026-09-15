import os
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
           "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
           "AVGO", "NOW", "MU"]

DATA_RAW = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", "data", "raw")

for symbol in TICKERS:
    path = os.path.join(DATA_RAW, f"{symbol}_ohlcv_3yr.csv")
    if not os.path.exists(path):
        print(f"{symbol}: FILE MISSING")
        continue
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print(
        f"{symbol}: {len(df)} rows, {df[date_col].min()} to {df[date_col].max()}")
