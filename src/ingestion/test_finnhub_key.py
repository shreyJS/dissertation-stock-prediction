"""
Quick sanity check for a Finnhub free-tier API key.

Usage:
    export FINNHUB_API_KEY=key_here
    python test_finnhub_key.py

What it checks:
1. The key authenticates successfully.
2. A single company-news call for AAPL over a realistic backfill window
   (e.g. 30 days) returns articles spanning the *full* requested range,
   not just the most recent day or two.
"""

import os
import sys
from datetime import date, timedelta

import requests

API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"


def main():
    if not API_KEY:
        print("ERROR: Set FINNHUB_API_KEY as an environment variable first.")
        print("  export FINNHUB_API_KEY=key_here")
        sys.exit(1)

    to_date = date.today()
    from_date = to_date - timedelta(days=30)

    params = {
        "symbol": "AAPL",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": API_KEY,
    }

    print(f"Requesting AAPL news from {from_date} to {to_date} ...")
    resp = requests.get(BASE_URL, params=params, timeout=15)

    if resp.status_code != 200:
        print(f"FAILED: HTTP {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)

    articles = resp.json()
    if not isinstance(articles, list):
        print("FAILED: unexpected response shape (not a list).")
        print(articles)
        sys.exit(1)

    if len(articles) == 0:
        print("WARNING: 0 articles returned. Key works, but check symbol/date range.")
        sys.exit(0)

    dates = sorted({
        date.fromtimestamp(a["datetime"]).isoformat()
        for a in articles if "datetime" in a
    })

    print(f"SUCCESS: {len(articles)} articles returned.")
    print(f"Earliest article date: {dates[0]}")
    print(f"Latest article date:   {dates[-1]}")
    print(
        f"Distinct days covered: {len(dates)} / {(to_date - from_date).days + 1}")

    span_days = (date.fromisoformat(
        dates[-1]) - date.fromisoformat(dates[0])).days
    if span_days < 20:
        print(
            "\nCAUTION: coverage span is narrower than the requested 30 days "
            f"(only spans {span_days} days)."
        )
    else:
        print(
            "\nLooks good — full-range coverage in one request. "
        )


if __name__ == "__main__":
    main()
