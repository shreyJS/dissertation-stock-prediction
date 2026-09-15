"""
Chunked Finnhub company-news backfill.

Finnhub's free tier caps each company-news response at ~240-250 articles,
so a single 12-month request silently only returns the most recent ~5-7
days within that window for high-volume tickers. This module requests in
small date chunks instead, so no chunk ever hits the article cap.

Usage as a library:
    from finnhub_backfill import backfill_ticker
    df = backfill_ticker("AAPL", days_back=365)

Usage as a script (smoke test on one ticker, short range):
    $env:FINNHUB_API_KEY="key_here"
    python finnhub_backfill.py
"""

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"

# Chunk size in days. 5 gives real room under the ~240-250 article cap
# even for very high-volume tickers (AAPL/TSLA/NVDA)
CHUNK_DAYS = 5

# Finnhub free tier: 60 calls/minute. Small sleep keeps this comfortably
# under that even with network jitter, without meaningfully slowing the
# backfill (15 tickers * ~73 chunks/ticker * 0.3s =~ 5.5 min of sleep time).
SLEEP_BETWEEN_CALLS = 0.3

MAX_RETRIES = 3


def _fetch_chunk(symbol: str, from_date: date, to_date: date) -> list:
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
            # Rate limited - back off and retry
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


def backfill_ticker(symbol: str, days_back: int = 365, verbose: bool = True) -> pd.DataFrame:
    """
    Backfill company news for one ticker over the last `days_back` days,
    chunked to avoid the free-tier per-request article cap.

    Returns a deduplicated DataFrame sorted by datetime.
    """
    if not API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in environment.")

    today = date.today()
    start = today - timedelta(days=days_back)

    all_articles = []
    chunk_start = start
    n_chunks = 0
    n_at_cap = 0

    while chunk_start < today:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), today)
        articles = _fetch_chunk(symbol, chunk_start, chunk_end)
        n_chunks += 1

        if len(articles) >= 240:
            # Even a 5-day chunk hit the cap - flag it. This would mean
            # coverage is still being silently truncated for this chunk.
            n_at_cap += 1
            if verbose:
                print(f"  CAUTION: chunk {chunk_start}..{chunk_end} returned "
                      f"{len(articles)} articles - close to/at the cap. "
                      f"Consider shrinking CHUNK_DAYS for this ticker.")

        all_articles.extend(articles)

        if verbose:
            print(
                f"  [{symbol}] {chunk_start} to {chunk_end}: {len(articles)} articles")

        chunk_start = chunk_end
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_articles:
        if verbose:
            print(
                f"  No articles returned for {symbol} across {n_chunks} chunks.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    before = len(df)
    # Dedup: Finnhub sometimes returns the same article id across
    # overlapping chunk boundaries.
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")
    else:
        df = df.drop_duplicates(subset=["headline", "datetime"])
    after = len(df)

    df["date"] = pd.to_datetime(df["datetime"], unit="s")
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        print(f"  [{symbol}] TOTAL: {after} unique articles "
              f"({before - after} duplicates removed) across {n_chunks} chunks, "
              f"{n_at_cap} chunk(s) near/at cap.")
        print(
            f"  [{symbol}] date range covered: {df['date'].min()} to {df['date'].max()}")

    return df


if __name__ == "__main__":
    # Smoke test: ONE ticker, ~60 days, before committing to the full
    # 15-ticker / 12-month run.
    TEST_SYMBOL = "AAPL"
    TEST_DAYS_BACK = 60

    print(
        f"Smoke-testing chunked backfill: {TEST_SYMBOL}, last {TEST_DAYS_BACK} days\n")
    df = backfill_ticker(TEST_SYMBOL, days_back=TEST_DAYS_BACK)

    if df.empty:
        print("\nFAILED: no data returned. Do not proceed to full backfill yet.")
        sys.exit(1)

    expected_min_days_covered = TEST_DAYS_BACK - 10  # allow a little slack
    days_covered = (df["date"].max() - df["date"].min()).days

    print(f"\n--- Smoke test result ---")
    print(f"Articles: {len(df)}")
    print(f"Span covered: {days_covered} days (requested {TEST_DAYS_BACK})")

    if days_covered >= expected_min_days_covered:
        print("PASS: chunked backfill covers close to the full requested "
              "range. Safe to scale to all 15 tickers / 12 months.")
    else:
        print("CAUTION: covered span is meaningfully short of what was "
              "requested. Inspect the per-chunk log above before scaling up "
              "- may need a smaller CHUNK_DAYS for this ticker.")

    # Save smoke-test output for inspection
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw",
                            f"{TEST_SYMBOL}_news_smoketest.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
