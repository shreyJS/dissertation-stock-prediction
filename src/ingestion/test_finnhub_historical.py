"""
Confirms whether Finnhub's free-tier company-news endpoint actually
backfills OLD date ranges, or silently just returns recent news
regardless of the 'from' date requested.

Usage:
    $env:FINNHUB_API_KEY="key_here"
    python test_finnhub_historical.py
"""

import os
import sys
from datetime import date, timedelta

import requests

API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"


def check_window(label, from_date, to_date):
    params = {
        "symbol": "AAPL",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"{label}: FAILED HTTP {resp.status_code}")
        return
    articles = resp.json()
    if not articles:
        print(f"{label}: 0 articles returned.")
        return
    dates = sorted({date.fromtimestamp(
        a["datetime"]).isoformat() for a in articles if "datetime" in a})
    print(
        f"{label}: {len(articles)} articles, dates {dates[0]} to {dates[-1]}")


def main():
    if not API_KEY:
        print("ERROR: Set FINNHUB_API_KEY first.")
        sys.exit(1)

    today = date.today()

    # Window genuinely 6 months in the past
    check_window(
        "6-months-ago window",
        today - timedelta(days=200),
        today - timedelta(days=170),
    )

    # Window genuinely 3 months in the past
    check_window(
        "3-months-ago window",
        today - timedelta(days=100),
        today - timedelta(days=70),
    )

    # Recent window (control - this returns data)
    check_window(
        "recent window (control)",
        today - timedelta(days=7),
        today,
    )

    print(
        "\nIf the 6-months-ago and 3-months-ago windows return 0 articles "
        "(or only return dates from the LAST WEEK regardless of the window "
        "requested), the free tier does not backfill history for this "
        "endpoint."
    )


if __name__ == "__main__":
    main()
