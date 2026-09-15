"""
Model B needs 3 years of price-only training data. Model A/B are distinguished by training data VOLUME: 12mo vs
3yr, not lookback window). This is separate from the 12-month OHLCV
fetched in Phase 1, which only supports Model A and the sentiment-aligned
multimodal Models C/D.

Models A/B are price-only, so this is a longer yfinance pull.

Usage:
    python backfill_ohlcv_3yr.py
"""

import os
import time

import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)


def fetch_3yr_ohlcv(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="3y", progress=False, auto_adjust=True)
    if df.empty:
        return df
    # Same MultiIndex-flattening fix as the 12-month backfill.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df


def main():
    print(f"Fetching 3-year OHLCV for {len(TICKERS)} tickers (Model B)...\n")

    failures = []
    for i, symbol in enumerate(TICKERS, 1):
        out_path = os.path.join(DATA_RAW, f"{symbol}_ohlcv_3yr.csv")
        if os.path.exists(out_path):
            print(f"[{i}/{len(TICKERS)}] {symbol}: already exists, skipping.")
            continue

        try:
            df = fetch_3yr_ohlcv(symbol)
            if df.empty:
                print(f"[{i}/{len(TICKERS)}] {symbol}: FAILED - no data returned.")
                failures.append(symbol)
                continue

            # Defensive numeric coercion, same as build_features.py's fix,
            # applied here too so the saved file is already clean.
            numeric_cols = [c for c in ["Open", "High",
                                        "Low", "Close", "Volume"] if c in df.columns]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=numeric_cols)

            df.to_csv(out_path, index=False)
            print(f"[{i}/{len(TICKERS)}] {symbol}: {len(df)} rows, "
                  f"{df['Date'].min()} to {df['Date'].max()}")
        except Exception as e:
            print(f"[{i}/{len(TICKERS)}] {symbol}: ERROR - {e}")
            failures.append(symbol)

        time.sleep(0.5)  # being polite to yfinance

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} ticker(s) failed: {failures}")
        print("Rerun this script - completed tickers are skipped automatically.")
    else:
        print("All 15 tickers' 3-year OHLCV fetched successfully.")


if __name__ == "__main__":
    main()
