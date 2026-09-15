"""
Phase 3: Feature engineering.

Builds:
  - 18-feature technical-only vector (Models A/B) - implemented manually
    (not via pandas-ta/ta libraries, which have version-compatibility
    issues with recent numpy) so indicators are transparent and testable.
  - ~40-feature multimodal vector (Models C/D): the 18 technical features
    + sentiment (static + temporal + peer, ~19 features) + macro
    (VIX, 10Y yield, earnings window, ~3 features).
  - UP/FLAT/DOWN labels at the default +/-0.5% daily-return threshold.
  - Chronological 70/10/10/10 split (train/calibration/val/test), applied
    at the SAME date cutoffs across all 15 tickers (not per-ticker
    quantiles) so no ticker's test period leaks into another's training
    period via a shared macro/sentiment regime.

Prerequisites: Phase 1 (data/raw/*) and Phase 2 (sentiment_output/*)must
both be complete.

Usage:
    python build_features.py
"""

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
SENTIMENT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "sentiment_output")
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

# Sub-sector groupings for peer sentiment (from ticker table) -
# used to compute "how is my peer group feeling today" as a feature,
# distinct from the ticker's own sentiment.
SUBSECTOR = {
    "AAPL": "consumer_electronics",
    "MSFT": "software", "CRM": "software", "ORCL": "software",
    "ADBE": "software", "NOW": "software",
    "NVDA": "semiconductors", "AMD": "semiconductors", "INTC": "semiconductors",
    "QCOM": "semiconductors", "AVGO": "semiconductors", "MU": "semiconductors",
    "GOOGL": "internet_services",
    "META": "social_technology",
    "AMZN": "cloud_ecommerce",
}

UP_DOWN_THRESHOLD = 0.005  # +/- 0.5% default

SPLIT_RATIOS = {"train": 0.70, "calibration": 0.10,
                "validation": 0.10, "test": 0.10}


# ---------------------------------------------------------------------
# Technical indicators (18 features, locked set for Models A/B)
# ---------------------------------------------------------------------

def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the original dataframe with 18 added technical feature columns.
    Implemented manually with standard formulas - no external TA library
    dependency, so behaviour is fully transparent and testable.
    """
    df = ohlcv.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # 1-3: returns
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)

    # 4-6: simple moving averages (as ratio to close, so they're scale-free)
    df["sma_5_ratio"] = close.rolling(5).mean() / close
    df["sma_10_ratio"] = close.rolling(10).mean() / close
    df["sma_20_ratio"] = close.rolling(20).mean() / close

    # 7-8: exponential moving averages (ratio to close)
    df["ema_10_ratio"] = close.ewm(span=10, adjust=False).mean() / close
    df["ema_20_ratio"] = close.ewm(span=20, adjust=False).mean() / close

    # 9: RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 10-12: MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # 13-15: Bollinger Bands (20-period, 2 std)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper_dist"] = (bb_mid + 2 * bb_std - close) / close
    df["bb_lower_dist"] = (close - (bb_mid - 2 * bb_std)) / close
    df["bb_width"] = (4 * bb_std) / bb_mid

    # 16: ATR-14 (Average True Range)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / close  # normalised by price

    # 17: Volume change
    df["volume_change_1d"] = volume.pct_change(1)

    # 18: OBV (On-Balance Volume), normalised via rolling z-score to keep
    # it stationary-ish rather than an unbounded cumulative series
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
assert len(
    TECHNICAL_FEATURE_COLS) == 18, "Technical feature set must stay at 18."


# ---------------------------------------------------------------------
# Sentiment features (static + temporal + peer)
# ---------------------------------------------------------------------

def compute_sentiment_features(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Input: one ticker's daily sentiment (date_only, finbert_mean,
    finbert_std, vader_mean, n_articles).
    Output: same rows, reindexed to a continuous daily calendar (so joins
    against trading days work) with static + temporal sentiment features.
    """
    df = daily_sentiment.copy()
    df["date"] = pd.to_datetime(df["date_only"])
    df = df.set_index("date").sort_index()

    # Reindex to continuous calendar days so rolling windows are correct
    # across weekends/gaps, then forward-fill sparse days (a day with no
    # news isn't "neutral", it's "no new information" - ffill carries the
    # last known sentiment state forward, which is the more reasonable
    # assumption for a daily prediction task).
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df["n_articles"] = df["n_articles"].fillna(0)
    df[["finbert_mean", "finbert_std", "vader_mean"]] = (
        df[["finbert_mean", "finbert_std", "vader_mean"]].ffill()
    )

    out = pd.DataFrame(index=df.index)

    # Static (4): today's raw sentiment snapshot
    out["sent_finbert_mean"] = df["finbert_mean"]
    out["sent_finbert_std"] = df["finbert_std"].fillna(0)
    out["sent_vader_mean"] = df["vader_mean"]
    out["sent_n_articles"] = df["n_articles"]

    # Static extended (2): positive/negative article share (article-level
    # data would be more precise, but day-level mean as proxy keeps this
    # lightweight - documented simplification)
    out["sent_finbert_positive_flag"] = (df["finbert_mean"] > 0.1).astype(int)
    out["sent_finbert_negative_flag"] = (df["finbert_mean"] < -0.1).astype(int)

    # Temporal (8): rolling windows over 3 and 7 days
    for window in [3, 7]:
        out[f"sent_finbert_mean_{window}d"] = df["finbert_mean"].rolling(
            window).mean()
        out[f"sent_vader_mean_{window}d"] = df["vader_mean"].rolling(
            window).mean()
        out[f"sent_finbert_std_{window}d"] = df["finbert_mean"].rolling(
            window).std()
        out[f"sent_n_articles_{window}d"] = df["n_articles"].rolling(
            window).sum()

    # Momentum (2): change in sentiment vs. prior day / prior week
    out["sent_momentum_1d"] = df["finbert_mean"].diff(1)
    out["sent_momentum_7d"] = df["finbert_mean"].diff(7)

    # Divergence (1): how much FinBERT and VADER disagree today
    out["sent_finbert_vader_divergence"] = (
        df["finbert_mean"] - df["vader_mean"]).abs()

    out["date"] = out.index
    return out.reset_index(drop=True)


SENTIMENT_FEATURE_COLS = [
    "sent_finbert_mean", "sent_finbert_std", "sent_vader_mean", "sent_n_articles",
    "sent_finbert_positive_flag", "sent_finbert_negative_flag",
    "sent_finbert_mean_3d", "sent_vader_mean_3d", "sent_finbert_std_3d", "sent_n_articles_3d",
    "sent_finbert_mean_7d", "sent_vader_mean_7d", "sent_finbert_std_7d", "sent_n_articles_7d",
    "sent_momentum_1d", "sent_momentum_7d",
    "sent_finbert_vader_divergence",
]  # 17 features


# ---------------------------------------------------------------------
# Macro features (VIX, 10Y yield, earnings window)
# ---------------------------------------------------------------------

def fetch_earnings_windows(symbol: str) -> set:
    """Returns a set of dates within +/-1 day of a known/estimated earnings date."""
    try:
        tk = yf.Ticker(symbol)
        edf = tk.get_earnings_dates(limit=12)
        if edf is None or edf.empty:
            return set()
        dates = pd.to_datetime(edf.index).tz_localize(None)
        window_dates = set()
        for d in dates:
            for offset in [-1, 0, 1]:
                window_dates.add((d + pd.Timedelta(days=offset)).normalize())
        return window_dates
    except Exception as e:
        print(f"    WARNING: could not fetch earnings dates for {symbol}: {e}")
        return set()


MACRO_FEATURE_COLS = ["macro_vix", "macro_tnx_10y", "macro_earnings_window"]


# ---------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------

def label_movement(returns_1d: pd.Series, threshold: float = UP_DOWN_THRESHOLD) -> pd.Series:
    labels = pd.Series("FLAT", index=returns_1d.index)
    labels[returns_1d > threshold] = "UP"
    labels[returns_1d < -threshold] = "DOWN"
    labels[returns_1d.isna()] = np.nan
    return labels


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def build_ticker_features(symbol: str, macro: pd.DataFrame) -> pd.DataFrame:
    ohlcv_path = os.path.join(DATA_RAW, f"{symbol}_ohlcv.csv")
    sentiment_path = os.path.join(
        SENTIMENT_DIR, f"{symbol}_sentiment_daily.csv")

    if not os.path.exists(ohlcv_path):
        print(f"  [{symbol}] MISSING OHLCV file, skipping.")
        return pd.DataFrame()
    if not os.path.exists(sentiment_path):
        print(f"  [{symbol}] MISSING sentiment file, skipping.")
        return pd.DataFrame()

    ohlcv = pd.read_csv(ohlcv_path)
    date_col = "Date" if "Date" in ohlcv.columns else ohlcv.columns[0]

    # Fix for yfinance issue: saving a MultiIndex-column
    # DataFrame (which yf.download returns even for a single ticker) via
    # to_csv() leaves a stray header row in the data (e.g. literally
    # "AAPL" in every price column) that gets misread as a real row on
    # reload, changing the column dtype to string. Coerce to numeric and
    # drop whatever does not survive, this cleanly removes the bad row(s)
    # regardless of exactly how many or where they land.
    numeric_cols = [c for c in ["Open", "High", "Low",
                                "Close", "Volume"] if c in ohlcv.columns]
    for col in numeric_cols:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv[date_col] = pd.to_datetime(ohlcv[date_col], errors="coerce")
    before_clean = len(ohlcv)
    ohlcv = ohlcv.dropna(subset=numeric_cols + [date_col])
    if len(ohlcv) < before_clean:
        print(f"    Cleaned {before_clean - len(ohlcv)} corrupted OHLCV row(s) "
              f"(known yfinance CSV header artifact).")

    ohlcv[date_col] = ohlcv[date_col].dt.normalize()
    ohlcv = ohlcv.rename(columns={date_col: "date"}).sort_values("date")

    ohlcv = compute_technical_features(ohlcv)

    sentiment_daily = pd.read_csv(sentiment_path)
    sent_features = compute_sentiment_features(sentiment_daily)

    # Merge technical + sentiment on date (left join on trading days -
    # sentiment was reindexed to calendar days and ffilled, so every
    # trading day has a value as long as it's within sentiment's range)
    merged = ohlcv.merge(sent_features, on="date", how="left")

    # Merge macro
    merged = merged.merge(macro, on="date", how="left")

    # Earnings window flag
    earnings_dates = fetch_earnings_windows(symbol)
    merged["macro_earnings_window"] = merged["date"].isin(
        earnings_dates).astype(int)

    # Peer sentiment: mean finbert sentiment of other tickers in the same
    # sub-sector, on the same day (computed after all tickers are loaded -
    # placeholder here, filled in by caller)
    merged["ticker"] = symbol
    merged["subsector"] = SUBSECTOR[symbol]

    # Label
    # Label uses NEXT-DAY return (shift -1), NOT same-day returns_1d.
    # returns_1d is also a technical FEATURE, so labeling off the same-day
    # value would let the model read the label directly off a feature -
    # trivial leakage. The label must represent something to be predicted:
    # tomorrow's movement, given only information known as of today.
    forward_return = merged["returns_1d"].shift(-1)
    merged["label"] = label_movement(forward_return)

    return merged


def add_peer_sentiment(all_data: pd.DataFrame) -> pd.DataFrame:
    """Adds 2 peer-sentiment features: same-subsector mean FinBERT sentiment
    (excluding the ticker itself) and its 1-day momentum.

    Falls back to the overall cross-ticker mean (excluding self) for any
    ticker whose sub-sector has no other members - AAPL (consumer
    electronics), GOOGL (internet services), META (social technology),
    and AMZN (cloud/e-commerce) are all singletons in this 15-ticker
    universe. Without this fallback, their peer-group denominator is
    zero, peer_sent_finbert_mean is NaN for every one of their rows, and
    since it's a required feature column, dropna() silently drops all
    four tickers entirely from the multimodal dataset - caught by
    test_all_15_tickers_present during testing.
    """
    # Sub-sector peer mean (excluding self)
    subsector_agg = (
        all_data.groupby(["subsector", "date"])["sent_finbert_mean"]
        .agg(["sum", "count"])
        .reset_index()
    )
    all_data = all_data.merge(
        subsector_agg, on=["subsector", "date"], how="left")
    subsector_denom = all_data["count"] - 1
    subsector_peer_mean = (
        all_data["sum"] - all_data["sent_finbert_mean"]) / subsector_denom.replace(0, np.nan)
    all_data = all_data.drop(columns=["sum", "count"])

    # Market-wide peer mean (excluding self) - fallback for singleton sub-sectors
    market_agg = (
        all_data.groupby("date")["sent_finbert_mean"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "market_sum", "count": "market_count"})
    )
    all_data = all_data.merge(market_agg, on="date", how="left")
    market_denom = all_data["market_count"] - 1
    market_peer_mean = (
        all_data["market_sum"] - all_data["sent_finbert_mean"]) / market_denom.replace(0, np.nan)
    all_data = all_data.drop(columns=["market_sum", "market_count"])

    # Use sub-sector peer mean where it exists (has real peers), otherwise
    # fall back to the market-wide mean.
    all_data["peer_sent_finbert_mean"] = subsector_peer_mean.fillna(
        market_peer_mean)

    all_data = all_data.sort_values(["ticker", "date"])
    all_data["peer_sent_momentum_1d"] = (
        all_data.groupby("ticker")["peer_sent_finbert_mean"].diff(1)
    )
    return all_data


PEER_FEATURE_COLS = ["peer_sent_finbert_mean", "peer_sent_momentum_1d"]


def chronological_split(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a split label using SHARED date cutoffs across all tickers
    (not per-ticker quantiles), so the split boundary is a single set of
    calendar dates - avoids one ticker's "test" period overlapping
    another's "train" period in time.
    """
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
    print(f"Building features for {len(TICKERS)} tickers...\n")

    macro_path = os.path.join(DATA_RAW, "macro.csv")
    if not os.path.exists(macro_path):
        print("ERROR: macro.csv not found. Run Phase 1 first.")
        sys.exit(1)
    macro = pd.read_csv(macro_path)
    macro = macro.rename(columns={macro.columns[0]: "date", "VIX": "macro_vix",
                                  "TNX_10Y_yield": "macro_tnx_10y"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    macro = macro[["date", "macro_vix", "macro_tnx_10y"]]

    all_ticker_dfs = []
    for symbol in TICKERS:
        print(f"[{symbol}]")
        df = build_ticker_features(symbol, macro)
        if not df.empty:
            all_ticker_dfs.append(df)
            print(f"  {len(df)} rows built.")
        print()

    if not all_ticker_dfs:
        print("ERROR: no ticker data built. Check Phase 1/2 outputs exist.")
        sys.exit(1)

    all_data = pd.concat(all_ticker_dfs, ignore_index=True)
    print(
        f"Combined: {len(all_data)} rows across {all_data['ticker'].nunique()} tickers.\n")

    print("Adding peer sentiment features...")
    all_data = add_peer_sentiment(all_data)

    print("Assigning chronological split...")
    all_data = chronological_split(all_data)

    # Drop rows where technical indicators are not yet defined (first ~20
    # trading days per ticker, due to rolling windows) or label is NaN
    # (most recent day, since returns_1d needs the next value... actually
    # returns_1d uses backward diff so this is the FIRST day, not last -
    # but also drop the very last day per ticker since some models may
    # need a forward-looking check; keep as-is here, drop only NaNs)
    feature_cols = TECHNICAL_FEATURE_COLS + SENTIMENT_FEATURE_COLS + \
        PEER_FEATURE_COLS + MACRO_FEATURE_COLS
    before = len(all_data)
    all_data_clean = all_data.dropna(subset=feature_cols + ["label"])
    after = len(all_data_clean)
    print(f"Dropped {before - after} rows with NaN features/label "
          f"(warm-up period for rolling indicators). {after} rows remain.\n")

    print(f"Total feature count: {len(feature_cols)} "
          f"(18 technical + {len(SENTIMENT_FEATURE_COLS)} sentiment + "
          f"{len(PEER_FEATURE_COLS)} peer + {len(MACRO_FEATURE_COLS)} macro)")

    # --- Class balance check ---
    print("\n--- Class balance check (±0.5% threshold) ---")
    balance = all_data_clean["label"].value_counts(normalize=True) * 100
    print(balance.round(1))
    min_class_pct = balance.min()
    if min_class_pct < 15:
        print(f"\nCAUTION: smallest class is only {min_class_pct:.1f}% of data. "
              f"Consider widening the "
              f"threshold beyond +/-0.5% if this looks too imbalanced for "
              f"stable training.")
    else:
        print(f"\nClass balance looks reasonable (smallest class {min_class_pct:.1f}%). "
              f"Threshold default of +/-0.5% is fine to keep.")

    # --- Split size check ---
    print("\n--- Split sizes ---")
    print(all_data_clean["split"].value_counts())

    # --- Save ---
    out_path = os.path.join(DATA_PROCESSED, "features_full.csv")
    # Save only meta columns + the 40 named features - all_data_clean also
    # carries raw OHLCV and 'subsector' columns used internally, which
    # doesn't affect training (features are always selected by explicit
    # name) but does not match the documented 40-feature schema either.
    full_cols = ["date", "ticker", "label", "split"] + feature_cols
    all_data_clean[full_cols].to_csv(out_path, index=False)
    print(f"\nSaved full feature set to {out_path}")

    # Also save a technical-only version for Models A/B - dropna ONLY on
    # technical features (Models A/B are price-only and don't consume
    # sentiment/macro, so they shouldn't lose rows to sentiment NaNs).
    ab_technical_dropna = all_data.dropna(
        subset=TECHNICAL_FEATURE_COLS + ["label"])
    ab_cols = ["date", "ticker", "label", "split"] + TECHNICAL_FEATURE_COLS
    ab_path = os.path.join(DATA_PROCESSED, "features_technical_only.csv")
    ab_technical_dropna[ab_cols].to_csv(ab_path, index=False)
    print(f"Saved technical-only (18-feature) set for Models A/B to {ab_path} "
          f"({len(ab_technical_dropna)} rows - independent of sentiment availability, "
          f"so this is NOT reduced by the 54% sentiment-driven drop above).")

    print(
        "\nNOTE: Model B trains on 3 YEARS of price data. "
        "The current OHLCV pull only covers ~1.5 years, so it does NOT yet "
        "support Model B's full training window - a separate, longer "
        "OHLCV-only backfill (no Finnhub/sentiment needed) is still required "
        "before Phase 4/Model B."
    )

    print("\n--- Technical-only (Models A/B) class balance ---")
    ab_balance = ab_technical_dropna["label"].value_counts(
        normalize=True) * 100
    print(ab_balance.round(1))
    print("\n--- Technical-only (Models A/B) split sizes ---")
    print(ab_technical_dropna["split"].value_counts())


if __name__ == "__main__":
    main()
