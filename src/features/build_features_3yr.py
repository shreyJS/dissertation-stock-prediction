"""
Builds the technical-only feature set for Model B (3-year training
window), reusing the exact same 18-feature technical indicator logic as
build_features.py - kept in sync manually since Model B needs no
sentiment/macro merge, just a longer OHLCV history.

Usage:
    python build_features_3yr.py
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

UP_DOWN_THRESHOLD = 0.005
SPLIT_RATIOS = {"train": 0.70, "calibration": 0.10,
                "validation": 0.10, "test": 0.10}


# --- Identical technical indicator logic to build_features.py ---------

def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)

    df["sma_5_ratio"] = close.rolling(5).mean() / close
    df["sma_10_ratio"] = close.rolling(10).mean() / close
    df["sma_20_ratio"] = close.rolling(20).mean() / close

    df["ema_10_ratio"] = close.ewm(span=10, adjust=False).mean() / close
    df["ema_20_ratio"] = close.ewm(span=20, adjust=False).mean() / close

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper_dist"] = (bb_mid + 2 * bb_std - close) / close
    df["bb_lower_dist"] = (close - (bb_mid - 2 * bb_std)) / close
    df["bb_width"] = (4 * bb_std) / bb_mid

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / close

    df["volume_change_1d"] = volume.pct_change(1)

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv_roll_mean = obv.rolling(20).mean()
    obv_roll_std = obv.rolling(20).std()
    df["obv_zscore"] = (obv - obv_roll_mean) / obv_roll_std.replace(0, np.nan)

    return df


TECHNICAL_FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_10d",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio",
    "ema_10_ratio", "ema_20_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper_dist", "bb_lower_dist", "bb_width",
    "atr_14",
    "volume_change_1d",
    "obv_zscore",
]
assert len(TECHNICAL_FEATURE_COLS) == 18


def label_movement(returns_1d: pd.Series, threshold: float = UP_DOWN_THRESHOLD) -> pd.Series:
    labels = pd.Series("FLAT", index=returns_1d.index)
    labels[returns_1d > threshold] = "UP"
    labels[returns_1d < -threshold] = "DOWN"
    labels[returns_1d.isna()] = np.nan
    return labels


def chronological_split(all_data: pd.DataFrame) -> pd.DataFrame:
    unique_dates = sorted(all_data["date"].unique())
    n = len(unique_dates)
    train_end = int(n * SPLIT_RATIOS["train"])
    calib_end = train_end + int(n * SPLIT_RATIOS["calibration"])
    val_end = calib_end + int(n * SPLIT_RATIOS["validation"])

    date_to_split = {}
    for i, d in enumerate(unique_dates):
        if i < train_end:
            date_to_split[d] = "train"
        elif i < calib_end:
            date_to_split[d] = "calibration"
        elif i < val_end:
            date_to_split[d] = "validation"
        else:
            date_to_split[d] = "test"

    all_data["split"] = all_data["date"].map(date_to_split)
    return all_data


def main():
    print(
        f"Building Model B (3-year technical-only) features for {len(TICKERS)} tickers...\n")

    all_dfs = []
    for symbol in TICKERS:
        path = os.path.join(DATA_RAW, f"{symbol}_ohlcv_3yr.csv")
        if not os.path.exists(path):
            print(f"[{symbol}] MISSING 3yr OHLCV file, skipping.")
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
        if len(df) < before:
            print(f"[{symbol}] cleaned {before - len(df)} bad row(s).")

        df[date_col] = df[date_col].dt.normalize()
        df = df.rename(columns={date_col: "date"}).sort_values("date")

        df = compute_technical_features(df)
        # Label uses NEXT-DAY return (shift -1), not same-day returns_1d -
        # returns_1d is also a technical feature, so same-day labeling
        # would leak the label through a feature. df is single-ticker here.
        forward_return = df["returns_1d"].shift(-1)
        df["label"] = label_movement(forward_return)
        df["ticker"] = symbol

        all_dfs.append(df)
        print(f"[{symbol}] {len(df)} rows built.")

    if not all_dfs:
        print("ERROR: no data built.")
        sys.exit(1)

    all_data = pd.concat(all_dfs, ignore_index=True)
    print(
        f"\nCombined: {len(all_data)} rows across {all_data['ticker'].nunique()} tickers.")

    all_data = chronological_split(all_data)

    before = len(all_data)
    all_data_clean = all_data.dropna(subset=TECHNICAL_FEATURE_COLS + ["label"])
    after = len(all_data_clean)
    print(
        f"Dropped {before - after} rows with NaN (rolling-indicator warm-up). {after} rows remain.\n")

    print("--- Class balance check (Model B, 3yr, ±0.5% threshold) ---")
    balance = all_data_clean["label"].value_counts(normalize=True) * 100
    print(balance.round(1))
    min_class_pct = balance.min()
    if min_class_pct < 15:
        print(
            f"\nCAUTION: smallest class is only {min_class_pct:.1f}% of data.")
    else:
        print(
            f"\nClass balance looks reasonable (smallest class {min_class_pct:.1f}%).")

    print("\n--- Split sizes (Model B, 3yr) ---")
    print(all_data_clean["split"].value_counts())

    out_cols = ["date", "ticker", "label", "split"] + TECHNICAL_FEATURE_COLS
    out_path = os.path.join(DATA_PROCESSED, "features_technical_3yr.csv")
    all_data_clean[out_cols].to_csv(out_path, index=False)
    print(f"\nSaved Model B (3-year) technical feature set to {out_path}")


if __name__ == "__main__":
    main()
