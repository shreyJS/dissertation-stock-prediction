"""
Adaptive chunked Finnhub company-news backfill.

Improves on the fixed-5-day-chunk version: instead of guessing a chunk
size, this requests a window and checks whether the response is near the
free-tier article cap. If it is, the window is too wide FOR THAT SPECIFIC
PERIOD (news volume varies day to day, ticker to ticker) - so it splits
the window in half and retries each half, recursively, until every chunk
comes back safely under the cap or hits a minimum window size (1 day).

This self-tunes per ticker and per time period, rather than relying on a
single fixed CHUNK_DAYS that was only validated against AAPL.

Usage as a script (smoke test on one ticker, short range):
    $env:FINNHUB_API_KEY="key_here"
    python finnhub_backfill_adaptive.py

Usage as a library:
    from finnhub_backfill_adaptive import backfill_ticker
    df = backfill_ticker("AAPL", days_back=365)
"""

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"

# If a chunk's response count is >= this, treat it as "possibly truncated"
# and split the window in half rather than trusting it. Finnhub's cap
# appears to sit right around 250, so 230 leaves a safety margin.
CAP_THRESHOLD = 230

# Never split below this - a 1-day window is as granular as Finnhub's
# `from`/`to` params support anyway (they're date-level, not time-level).
MIN_WINDOW_DAYS = 1

SLEEP_BETWEEN_CALLS = 0.3
MAX_RETRIES = 3


def _fetch_window(symbol: str, from_date: date, to_date: date) -> list:
    params = {
        "symbol": symbol,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": API_KEY,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(BASE_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []
        if resp.status_code == 429:
            wait = 2 * attempt
            print(f"  Rate limited on {symbol} {from_date}..{to_date}, "
                  f"waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        print(f"  WARNING: {symbol} {from_date}..{to_date} -> "
              f"HTTP {resp.status_code}: {resp.text[:200]}")
        return []
    print(
        f"  FAILED after {MAX_RETRIES} retries: {symbol} {from_date}..{to_date}")
    return []


def _adaptive_fetch(symbol: str, from_date: date, to_date: date,
                    stats: dict, verbose: bool) -> list:
    """Recursively fetch a window, splitting if near the article cap."""
    articles = _fetch_window(symbol, from_date, to_date)
    stats["n_requests"] += 1
    time.sleep(SLEEP_BETWEEN_CALLS)

    window_days = (to_date - from_date).days

    if len(articles) >= CAP_THRESHOLD and window_days > MIN_WINDOW_DAYS:
        # Likely truncated - split and retry each half
        mid = from_date + timedelta(days=max(1, window_days // 2))
        if verbose:
            print(f"  SPLIT: {symbol} {from_date}..{to_date} returned "
                  f"{len(articles)} (>= cap threshold {CAP_THRESHOLD}), "
                  f"splitting at {mid}")
        stats["n_splits"] += 1
        left = _adaptive_fetch(symbol, from_date, mid, stats, verbose)
        right = _adaptive_fetch(symbol, mid, to_date, stats, verbose)
        return left + right

    if len(articles) >= CAP_THRESHOLD and window_days <= MIN_WINDOW_DAYS:
        # Hit the floor - can't split further.
        stats["n_unresolved_caps"] += 1
        if verbose:
            print(f"  UNRESOLVED CAP: {symbol} {from_date}..{to_date} "
                  f"(1-day window) still returned {len(articles)} articles - "
                  f"true daily volume may exceed what the free tier can "
                  f"return for this single day.")

    if verbose:
        print(f"  [{symbol}] {from_date} to {to_date}: {len(articles)} articles")

    return articles


def backfill_ticker(symbol: str, days_back: int = 365, verbose: bool = True) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in environment.")

    today = date.today()
    start = today - timedelta(days=days_back)

    stats = {"n_requests": 0, "n_splits": 0, "n_unresolved_caps": 0}
    all_articles = _adaptive_fetch(symbol, start, today, stats, verbose)

    if not all_articles:
        if verbose:
            print(f"  No articles returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    before = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
    else:
        df = df.drop_duplicates(subset=["headline", "datetime"])
    after = len(df)

    df["date"] = pd.to_datetime(df["datetime"], unit="s")
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        print(f"\n  [{symbol}] TOTAL: {after} unique articles "
              f"({before - after} duplicates removed)")
        print(f"  [{symbol}] {stats['n_requests']} requests, "
              f"{stats['n_splits']} splits, "
              f"{stats['n_unresolved_caps']} unresolved (1-day) caps")
        print(
            f"  [{symbol}] date range covered: {df['date'].min()} to {df['date'].max()}")

    return df


if __name__ == "__main__":
    TEST_SYMBOL = "AAPL"
    TEST_DAYS_BACK = 60

    print(
        f"Smoke-testing ADAPTIVE backfill: {TEST_SYMBOL}, last {TEST_DAYS_BACK} days\n")
    df = backfill_ticker(TEST_SYMBOL, days_back=TEST_DAYS_BACK)

    if df.empty:
        print("\nFAILED: no data returned. Do not proceed to full backfill yet.")
        sys.exit(1)

    days_covered = (df["date"].max() - df["date"].min()).days
    print(f"\n--- Smoke test result ---")
    print(f"Articles: {len(df)}")
    print(f"Span covered: {days_covered} days (requested {TEST_DAYS_BACK})")
    print(
        "\nIf 'unresolved caps' above is 0, every window is confirmed clean "
        "- safe to scale to all 15 tickers / 12 months. If it's non-zero, "
        "those specific days may still be missing some articles (genuinely "
        "very high single-day volume) - note it as a documented limitation "
        "rather than a silent gap."
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw",
                            f"{TEST_SYMBOL}_news_smoketest_adaptive.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
