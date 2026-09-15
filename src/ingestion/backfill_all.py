"""
Phase 1: Full data acquisition for all 15 tickers.

Combines:
  1. Finnhub company news (adaptive chunking - see finnhub_backfill_adaptive.py)
  2. yfinance OHLCV price data
  3. Macro data: VIX (^VIX) and 10Y Treasury yield (^TNX)

Saves incrementally per ticker to data/raw/ so network changes doesn't lose completed tickers - rerun and it skips ones already done.

Usage:
    $env:FINNHUB_API_KEY="key_here"
    python backfill_all.py
"""

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

# --- Config -----------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",   # Tier 1
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",       # Tier 2
    "AVGO", "NOW", "MU",                                 # Tier 2
]

DAYS_BACK = 365
API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"

CAP_THRESHOLD = 230
MIN_WINDOW_DAYS = 1
SLEEP_BETWEEN_CALLS = 0.3
MAX_RETRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)


# --- Finnhub adaptive news backfill (same logic as the validated smoke test) --

def _fetch_window(symbol, from_date, to_date):
    params = {"symbol": symbol, "from": from_date.isoformat(),
              "to": to_date.isoformat(), "token": API_KEY}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
        except requests.exceptions.RequestException as e:
            wait = 3 * attempt
            print(f"    network error ({e}) on {symbol} {from_date}..{to_date}, "
                  f"attempt {attempt}/{MAX_RETRIES}, waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []
        if resp.status_code == 429:
            time.sleep(2 * attempt)
            continue
        wait = 3 * attempt
        print(f"    WARNING: HTTP {resp.status_code} for {symbol} "
              f"{from_date}..{to_date}, attempt {attempt}/{MAX_RETRIES}, "
              f"waiting {wait}s: {resp.text[:150]}")
        time.sleep(wait)
    # None (not []) marks a confirmed failure - distinct from a confirmed-
    # empty response - so the caller knows to keep trying rather than
    # silently treating the window as "no articles here."
    return None


def _adaptive_fetch(symbol, from_date, to_date, stats):
    result = _fetch_window(symbol, from_date, to_date)
    stats["n_requests"] += 1
    time.sleep(SLEEP_BETWEEN_CALLS)

    window_days = (to_date - from_date).days

    if result is None:
        # Failed outright. If the window is still splittable, don't give
        # up on the whole span - split it so a bad request only costs us
        # a small piece, not the entire window.
        if window_days > MIN_WINDOW_DAYS:
            mid = from_date + timedelta(days=max(1, window_days // 2))
            stats["n_failure_splits"] += 1
            left = _adaptive_fetch(symbol, from_date, mid, stats)
            right = _adaptive_fetch(symbol, mid, to_date, stats)
            return left + right
        stats["n_permanent_failures"] += 1
        stats.setdefault("failed_windows", []).append(
            (from_date.isoformat(), to_date.isoformat()))
        print(f"    PERMANENT FAILURE (1-day window): {symbol} "
              f"{from_date}..{to_date} - genuine gap, logged.")
        return []

    articles = result
    if len(articles) >= CAP_THRESHOLD and window_days > MIN_WINDOW_DAYS:
        mid = from_date + timedelta(days=max(1, window_days // 2))
        stats["n_splits"] += 1
        left = _adaptive_fetch(symbol, from_date, mid, stats)
        right = _adaptive_fetch(symbol, mid, to_date, stats)
        return left + right

    if len(articles) >= CAP_THRESHOLD and window_days <= MIN_WINDOW_DAYS:
        stats["n_unresolved_caps"] += 1

    return articles


def backfill_news(symbol: str, days_back: int) -> pd.DataFrame:
    today = date.today()
    start = today - timedelta(days=days_back)
    stats = {"n_requests": 0, "n_splits": 0, "n_unresolved_caps": 0,
             "n_failure_splits": 0, "n_permanent_failures": 0, "failed_windows": []}

    all_articles = _adaptive_fetch(symbol, start, today, stats)
    if not all_articles:
        print(f"    No news articles returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
    else:
        df = df.drop_duplicates(subset=["headline", "datetime"])
    df["date"] = pd.to_datetime(df["datetime"], unit="s")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"    News: {len(df)} unique articles, {stats['n_requests']} requests, "
          f"{stats['n_splits']} splits, {stats['n_unresolved_caps']} unresolved caps, "
          f"{stats['n_permanent_failures']} permanent failures, "
          f"range {df['date'].min().date()} to {df['date'].max().date()}")
    if stats["failed_windows"]:
        print(
            f"    Permanently failed windows for {symbol}: {stats['failed_windows']}")
    return df


# --- OHLCV via yfinance -------------------------------------------------

def backfill_ohlcv(symbol: str, days_back: int) -> pd.DataFrame:
    period_days = days_back + 10  # small buffer
    df = yf.download(
        symbol, period=f"{period_days}d", progress=False, auto_adjust=True)
    if df.empty:
        print(f"    No OHLCV data returned for {symbol}.")
        return df
    # yf.download returns MultiIndex columns (Price, Ticker) even for a
    # single symbol - flatten before saving, or to_csv() leaves a stray
    # header row in the data that corrupts numeric dtypes on reload.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    print(
        f"    OHLCV: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


# --- Macro data (fetched once, not per-ticker) --------------------------

def fetch_macro(days_back: int) -> pd.DataFrame:
    period_days = days_back + 10
    vix = yf.download(
        "^VIX", period=f"{period_days}d", progress=False, auto_adjust=True)
    tnx = yf.download(
        "^TNX", period=f"{period_days}d", progress=False, auto_adjust=True)

    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    tnx = tnx[["Close"]].rename(columns={"Close": "TNX_10Y_yield"})

    macro = vix.join(tnx, how="outer").reset_index()
    print(
        f"Macro: {len(macro)} rows, {macro['Date'].min().date()} to {macro['Date'].max().date()}")
    return macro


# --- Main ----------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: Set FINNHUB_API_KEY first.")
        sys.exit(1)

    print(f"Phase 1 backfill: {len(TICKERS)} tickers, {DAYS_BACK} days\n")

    # Macro first - shared across all tickers, only needs fetching once
    macro_path = os.path.join(DATA_RAW, "macro.csv")
    if os.path.exists(macro_path):
        print(f"Macro data already exists at {macro_path}, skipping.\n")
    else:
        macro_df = fetch_macro(DAYS_BACK)
        macro_df.to_csv(macro_path, index=False)
        print(f"Saved macro data to {macro_path}\n")

    failures = []

    for i, symbol in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}] {symbol}")

        news_path = os.path.join(DATA_RAW, f"{symbol}_news.csv")
        ohlcv_path = os.path.join(DATA_RAW, f"{symbol}_ohlcv.csv")

        if os.path.exists(news_path) and os.path.exists(ohlcv_path):
            print(f"    Already done, skipping.\n")
            continue

        try:
            if not os.path.exists(news_path):
                news_df = backfill_news(symbol, DAYS_BACK)
                if news_df.empty:
                    failures.append(f"{symbol}: no news data")
                else:
                    news_df.to_csv(news_path, index=False)

            if not os.path.exists(ohlcv_path):
                ohlcv_df = backfill_ohlcv(symbol, DAYS_BACK)
                if ohlcv_df.empty:
                    failures.append(f"{symbol}: no OHLCV data")
                else:
                    ohlcv_df.to_csv(ohlcv_path, index=False)

        except Exception as e:
            print(f"    ERROR on {symbol}: {e}")
            failures.append(f"{symbol}: exception - {e}")

        print()

    print("=" * 60)
    if failures:
        print(f"Completed with {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        print("\nRerun this script - completed tickers are skipped, only "
              "the failed/missing ones will be retried.")
    else:
        print("All 15 tickers backfilled successfully (news + OHLCV) + macro data.")
    print(f"\nData saved to: {os.path.abspath(DATA_RAW)}")


if __name__ == "__main__":
    main()
