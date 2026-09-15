"""
Repairs specific date-window gaps left by transient failures in the
original backfill_all.py run. Those failures were wrongly treated as
"confirmed empty" instead of being retried harder - this script re-fetches
exactly those windows with longer backoff and more retries, then merges
the results into the existing per-ticker CSVs (dedup-safe).

Usage:
    $env:FINNHUB_API_KEY="key_here"
    python repair_gaps.py
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")

MAX_RETRIES = 6
BASE_BACKOFF = 4  # seconds, doubles each attempt
SLEEP_BETWEEN_CALLS = 0.5

FAILED_WINDOWS = [
    ("AMZN", "2026-01-05", "2026-01-28"),
    ("AMZN", "2026-03-09", "2026-03-10"),
    ("AMZN", "2026-04-23", "2026-04-24"),
    ("AMD", "2026-04-17", "2026-04-29"),
    ("AMD", "2026-07-27", "2026-07-30"),
    ("INTC", "2026-01-28", "2026-03-15"),
    ("INTC", "2026-04-12", "2026-04-14"),
    ("INTC", "2026-05-17", "2026-05-19"),
    ("INTC", "2026-07-07", "2026-07-18"),
    ("CRM", "2026-02-28", "2026-03-03"),
    ("ORCL", "2025-10-29", "2025-11-09"),
    ("ORCL", "2026-03-26", "2026-04-06"),
    ("ADBE", "2025-09-14", "2026-03-15"),
    ("QCOM", "2026-06-19", "2026-06-25"),
    ("AVGO", "2026-03-15", "2026-06-14"),
    ("NOW", "2026-04-17", "2026-04-23"),
    ("MU", "2026-03-15", "2026-06-14"),
    ("MU", "2026-07-24", "2026-07-27"),
]

# Windows wider than this get pre-split before fetching, since a window
# that failed outright (e.g. ADBE's 6-month gap) is too big to retry as
# one request even with more patience - break it into <=10-day pieces
# up front so a failure only costs us a small piece, not the whole span.
PRESPLIT_DAYS = 10


def _daterange_chunks(start: date, end: date, chunk_days: int):
    """Yield (chunk_start, chunk_end) date pairs covering [start, end)."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        yield cur, nxt
        cur = nxt


def _fetch_with_backoff(symbol, from_date, to_date):
    params = {"symbol": symbol, "from": from_date.isoformat(),
              "to": to_date.isoformat(), "token": API_KEY}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
        except requests.exceptions.RequestException as e:
            wait = BASE_BACKOFF * (2 ** (attempt - 1))
            print(f"    [{symbol}] network error ({e}) on {from_date}..{to_date}, "
                  f"attempt {attempt}/{MAX_RETRIES}, waiting {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []

        wait = BASE_BACKOFF * (2 ** (attempt - 1))
        print(f"    [{symbol}] HTTP {resp.status_code} on {from_date}..{to_date}, "
              f"attempt {attempt}/{MAX_RETRIES}, waiting {wait}s")
        time.sleep(wait)

    print(f"    [{symbol}] STILL FAILED after {MAX_RETRIES} attempts: "
          f"{from_date}..{to_date} - genuine gap, needs manual attention")
    return None  # None = confirmed failure, distinct from [] = confirmed empty


def repair_window(symbol, from_str, to_str):
    from_date = date.fromisoformat(from_str)
    to_date = date.fromisoformat(to_str)
    span = (to_date - from_date).days

    all_articles = []
    still_failed = []

    chunks = list(_daterange_chunks(from_date, to_date, PRESPLIT_DAYS)) if span > PRESPLIT_DAYS \
        else [(from_date, to_date)]

    for chunk_start, chunk_end in chunks:
        result = _fetch_with_backoff(symbol, chunk_start, chunk_end)
        time.sleep(SLEEP_BETWEEN_CALLS)
        if result is None:
            still_failed.append(
                (chunk_start.isoformat(), chunk_end.isoformat()))
        else:
            all_articles.extend(result)

    return all_articles, still_failed


def main():
    if not API_KEY:
        print("ERROR: Set FINNHUB_API_KEY first.")
        return

    print(f"Repairing {len(FAILED_WINDOWS)} known gap windows...\n")

    by_symbol = {}
    for symbol, from_str, to_str in FAILED_WINDOWS:
        by_symbol.setdefault(symbol, []).append((from_str, to_str))

    permanent_failures = []

    for symbol, windows in by_symbol.items():
        print(f"[{symbol}] repairing {len(windows)} window(s)")
        news_path = os.path.join(DATA_RAW, f"{symbol}_news.csv")

        existing = pd.read_csv(news_path) if os.path.exists(
            news_path) else pd.DataFrame()
        if not existing.empty and "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"])
        new_articles = []

        for from_str, to_str in windows:
            articles, still_failed = repair_window(symbol, from_str, to_str)
            new_articles.extend(articles)
            for f, t in still_failed:
                permanent_failures.append((symbol, f, t))

        if new_articles:
            new_df = pd.DataFrame(new_articles)
            new_df["date"] = pd.to_datetime(new_df["datetime"], unit="s")
            combined = pd.concat([existing, new_df], ignore_index=True)
            if "id" in combined.columns:
                combined = combined.drop_duplicates(subset="id")
            else:
                combined = combined.drop_duplicates(
                    subset=["headline", "datetime"])
            combined = combined.sort_values("date").reset_index(drop=True)
            combined.to_csv(news_path, index=False)
            print(f"    Added {len(new_df)} articles ({len(combined)} total after dedup). "
                  f"Full range now: {combined['date'].min().date()} to {combined['date'].max().date()}")
        else:
            print(f"    No new articles recovered for {symbol}.")
        print()

    print("=" * 60)
    if permanent_failures:
        print(
            f"{len(permanent_failures)} window(s) STILL failing after aggressive retry:")
        for symbol, f, t in permanent_failures:
            print(f"  - {symbol}: {f} to {t}")
        print(
            "\nThese are likely a real, transient Finnhub-side issue for that "
            "narrow window rather than a script bug (10-day pieces failing "
            "even with 6 retries and growing backoff is unusual). Treat as a "
            "small, documented data gap."
        )
    else:
        print("All known gaps repaired successfully.")


if __name__ == "__main__":
    main()
