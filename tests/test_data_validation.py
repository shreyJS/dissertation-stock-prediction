"""
Data validation smoke tests.
Checks row counts, NaN rates, and that the chronological split is
genuinely leakage-free (test dates strictly after train dates).

Run:
    pytest tests/test_data_validation.py -v
"""

import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(REPO_ROOT, "data", "processed")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

PROCESSED_FILES = {
    "features_technical_only.csv": 18,
    "features_technical_3yr.csv": 18,
    "features_full.csv": 40,
}


def _require_file(path):
    if not os.path.exists(path):
        pytest.skip(
            f"{path} not found - run the relevant pipeline phase first")


class TestRawDataIntegrity:

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_ohlcv_has_no_nans_in_price_columns(self, ticker):
        path = os.path.join(DATA_RAW, f"{ticker}_ohlcv.csv")
        _require_file(path)
        df = pd.read_csv(path)
        price_cols = [c for c in ["Open", "High", "Low",
                                  "Close", "Volume"] if c in df.columns]
        for col in price_cols:
            assert df[col].isna().sum() == 0, f"{ticker}: NaNs found in {col}"

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_ohlcv_no_duplicate_dates(self, ticker):
        path = os.path.join(DATA_RAW, f"{ticker}_ohlcv.csv")
        _require_file(path)
        df = pd.read_csv(path)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        assert df[date_col].duplicated().sum(
        ) == 0, f"{ticker}: duplicate dates in OHLCV"

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_news_has_reasonable_row_count(self, ticker):
        path = os.path.join(DATA_RAW, f"{ticker}_news.csv")
        _require_file(path)
        df = pd.read_csv(path)
        assert len(
            df) >= 50, f"{ticker}: suspiciously few news articles ({len(df)})"


class TestProcessedFeatureFiles:

    @pytest.mark.parametrize("filename,expected_n_features", list(PROCESSED_FILES.items()))
    def test_feature_file_has_no_nans(self, filename, expected_n_features):
        path = os.path.join(DATA_PROCESSED, filename)
        _require_file(path)
        df = pd.read_csv(path)
        meta_cols = {"date", "ticker", "label", "split"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        assert len(feature_cols) == expected_n_features, (
            f"{filename}: expected {expected_n_features} feature columns, found {len(feature_cols)}"
        )
        nan_counts = df[feature_cols].isna().sum()
        assert nan_counts.sum(
        ) == 0, f"{filename}: NaNs found in feature columns:\n{nan_counts[nan_counts > 0]}"

    @pytest.mark.parametrize("filename", list(PROCESSED_FILES.keys()))
    def test_no_train_test_date_leakage(self, filename):
        """The core anti-leakage check: every date in the test split must
        be chronologically AFTER every date in the train split. If this
        ever fails, something is deeply wrong with the split logic."""
        path = os.path.join(DATA_PROCESSED, filename)
        _require_file(path)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        train_dates = df[df["split"] == "train"]["date"]
        test_dates = df[df["split"] == "test"]["date"]
        if train_dates.empty or test_dates.empty:
            pytest.skip(
                f"{filename}: train or test split empty, cannot check ordering")

        assert train_dates.max() < test_dates.min(), (
            f"{filename}: LEAKAGE - latest train date ({train_dates.max()}) is not "
            f"before earliest test date ({test_dates.min()})"
        )

    @pytest.mark.parametrize("filename", list(PROCESSED_FILES.keys()))
    def test_labels_are_valid_classes(self, filename):
        path = os.path.join(DATA_PROCESSED, filename)
        _require_file(path)
        df = pd.read_csv(path)
        assert set(df["label"].unique()) <= {"UP", "DOWN", "FLAT"}, (
            f"{filename}: unexpected label values found: {df['label'].unique()}"
        )

    @pytest.mark.parametrize("filename", list(PROCESSED_FILES.keys()))
    def test_all_15_tickers_present(self, filename):
        path = os.path.join(DATA_PROCESSED, filename)
        _require_file(path)
        df = pd.read_csv(path)
        present = set(df["ticker"].unique())
        missing = set(TICKERS) - present
        assert not missing, f"{filename}: missing tickers entirely: {missing}"
