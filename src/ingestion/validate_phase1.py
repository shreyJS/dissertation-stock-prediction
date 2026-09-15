"""
Data validation smoke test for Phase 1 output.

Checks testing priorities:
  1. All 15 tickers have both news and OHLCV files present
  2. No unexpected NaNs in price/sentiment columns
  3. Date coverage - flags any ticker whose news/OHLCV doesn't span close
     to the full requested window (catches exactly the kind of silent
     truncation bug this pipeline already hit once)
  4. Row count sanity (OHLCV should be ~252 trading days/year, news should
     be non-trivial for every ticker)

This does NOT check train/test leakage yet (that happens after the
chronological split in Phase 3) - this is a pre-flight check on raw data
only.

Usage:
    python validate_phase1.py
"""

import os
import sys

import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")

EXPECTED_DAYS_BACK = 365
# Trading days in a year is ~252, not 365 - don't flag OHLCV as short
# just because it's less than 365 rows.
MIN_OHLCV_ROWS = 200
MIN_NEWS_ARTICLES = 50


def main():
    issues = []
    warnings = []

    print(f"Validating Phase 1 data for {len(TICKERS)} tickers...\n")

    # --- Macro check ---
    macro_path = os.path.join(DATA_RAW, "macro.csv")
    if not os.path.exists(macro_path):
        issues.append("macro.csv is missing entirely.")
    else:
        macro = pd.read_csv(macro_path)
        n_nan_vix = macro["VIX"].isna().sum(
        ) if "VIX" in macro.columns else "col missing"
        n_nan_tnx = macro["TNX_10Y_yield"].isna().sum(
        ) if "TNX_10Y_yield" in macro.columns else "col missing"
        print(
            f"[macro] {len(macro)} rows, NaN VIX: {n_nan_vix}, NaN TNX: {n_nan_tnx}")
        if isinstance(n_nan_vix, int) and n_nan_vix > len(macro) * 0.05:
            warnings.append(
                f"macro.csv: {n_nan_vix} NaN VIX values (>5% of rows)")
        if isinstance(n_nan_tnx, int) and n_nan_tnx > len(macro) * 0.05:
            warnings.append(
                f"macro.csv: {n_nan_tnx} NaN TNX values (>5% of rows)")

    print()

    # --- Per-ticker checks ---
    for symbol in TICKERS:
        news_path = os.path.join(DATA_RAW, f"{symbol}_news.csv")
        ohlcv_path = os.path.join(DATA_RAW, f"{symbol}_ohlcv.csv")

        print(f"[{symbol}]")

        # Existence
        if not os.path.exists(news_path):
            issues.append(f"{symbol}: news CSV missing entirely.")
            print("    News: MISSING")
        if not os.path.exists(ohlcv_path):
            issues.append(f"{symbol}: OHLCV CSV missing entirely.")
            print("    OHLCV: MISSING")
        if not os.path.exists(news_path) or not os.path.exists(ohlcv_path):
            print()
            continue

        # News checks
        news = pd.read_csv(news_path)
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        n_bad_dates = news["date"].isna().sum()
        if n_bad_dates > 0:
            issues.append(
                f"{symbol}: {n_bad_dates} news rows with unparseable dates.")

        if len(news) < MIN_NEWS_ARTICLES:
            issues.append(
                f"{symbol}: only {len(news)} news articles (< {MIN_NEWS_ARTICLES} threshold).")

        news_span_days = (news["date"].max() -
                          news["date"].min()).days if not news.empty else 0
        news_earliest = news["date"].min().date() if not news.empty else None
        expected_earliest = pd.Timestamp.today().normalize() - \
            pd.Timedelta(days=EXPECTED_DAYS_BACK)
        gap_from_expected_start = (
            news["date"].min() - expected_earliest).days if not news.empty else None

        print(f"    News: {len(news)} articles, {n_bad_dates} bad dates, "
              f"span {news_span_days}d, earliest {news_earliest}")

        if gap_from_expected_start is not None and gap_from_expected_start > 14:
            warnings.append(
                f"{symbol}: news coverage starts {gap_from_expected_start} days later than "
                f"the requested window start ({expected_earliest.date()}) - possible residual gap."
            )

        key_cols = [c for c in ["headline", "summary",
                                "related"] if c in news.columns]
        for col in key_cols:
            n_nan = news[col].isna().sum()
            if n_nan > 0:
                warnings.append(
                    f"{symbol}: {n_nan} NaN values in news column '{col}'.")

        # OHLCV checks
        ohlcv = pd.read_csv(ohlcv_path)
        date_col = "Date" if "Date" in ohlcv.columns else ohlcv.columns[0]
        ohlcv[date_col] = pd.to_datetime(ohlcv[date_col], errors="coerce")

        if len(ohlcv) < MIN_OHLCV_ROWS:
            issues.append(
                f"{symbol}: only {len(ohlcv)} OHLCV rows (< {MIN_OHLCV_ROWS} threshold).")

        price_cols = [c for c in ["Open", "High", "Low",
                                  "Close", "Volume"] if c in ohlcv.columns]
        nan_report = {c: int(ohlcv[c].isna().sum()) for c in price_cols}
        total_nans = sum(nan_report.values())
        print(f"    OHLCV: {len(ohlcv)} rows, NaNs per column: {nan_report}")

        if total_nans > 0:
            warnings.append(f"{symbol}: OHLCV has NaNs: {nan_report}")

        # Duplicate dates in OHLCV (would break time-series indexing)
        n_dupe_dates = ohlcv[date_col].duplicated().sum()
        if n_dupe_dates > 0:
            issues.append(
                f"{symbol}: {n_dupe_dates} duplicate dates in OHLCV.")

        print()

    print("=" * 60)
    print("VALIDATION SUMMARY\n")

    if issues:
        print(f"{len(issues)} ISSUE(S) - should be addressed before Phase 2/3:")
        for i in issues:
            print(f"  [ISSUE] {i}")
    else:
        print("No blocking issues found.")

    print()
    if warnings:
        print(f"{len(warnings)} WARNING(S) - review, likely OK to proceed:")
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print("No warnings.")

    print()
    if issues:
        print(
            "Recommendation: fix ISSUE items before moving to Phase 2 (sentiment pipeline).")
        sys.exit(1)
    else:
        print("Recommendation: safe to proceed to Phase 2.")


if __name__ == "__main__":
    main()
