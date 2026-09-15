"""
Targeted unit tests for feature engineering formulas and labeling logic -
the places a silent bug would be hardest to catch by eye (feature formulas, conformal calibration).

These import the ACTUAL functions from src/features/build_features.py
rather than reimplementing them, so a bug introduced there is caught
here.

Run:
    pytest tests/test_feature_formulas.py -v
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_FEATURES_PATH = os.path.join(
    REPO_ROOT, "src", "features", "build_features.py")


def _import_build_features():
    """Import build_features.py as a module without running its __main__."""
    if not os.path.exists(BUILD_FEATURES_PATH):
        pytest.skip(
            f"{BUILD_FEATURES_PATH} not found - run from repo root with src/features/build_features.py present")
    spec = importlib.util.spec_from_file_location(
        "build_features", BUILD_FEATURES_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_features"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bf():
    return _import_build_features()


@pytest.fixture
def synthetic_ohlcv():
    """20 days of simple synthetic price data - enough to clear all
    rolling-window warm-up periods (max window used is 20)."""
    n = 40
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "date": dates,
        "Open": close + rng.normal(0, 0.3, n),
        "High": close + np.abs(rng.normal(0, 0.5, n)),
        "Low": close - np.abs(rng.normal(0, 0.5, n)),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    })
    return df


class TestTechnicalIndicators:

    def test_sma_ratio_matches_independent_calculation(self, bf, synthetic_ohlcv):
        """Recompute SMA ratio via a separate pandas call (not the source
        function) and check they agree - catches window-size or
        division-direction bugs."""
        result = bf.compute_technical_features(synthetic_ohlcv)
        expected_sma5_ratio = synthetic_ohlcv["Close"].rolling(
            5).mean() / synthetic_ohlcv["Close"]
        pd.testing.assert_series_equal(
            result["sma_5_ratio"].reset_index(drop=True),
            expected_sma5_ratio.reset_index(drop=True),
            check_names=False,
        )

    def test_rsi_bounded_0_100(self, bf, synthetic_ohlcv):
        result = bf.compute_technical_features(synthetic_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (
            rsi <= 100).all(), "RSI must be bounded [0, 100]"

    def test_macd_hist_equals_macd_minus_signal(self, bf, synthetic_ohlcv):
        """macd_hist should be an exact identity, not an independently
        computed value - if it drifts, the formula was implemented wrong."""
        result = bf.compute_technical_features(synthetic_ohlcv)
        expected_hist = result["macd"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_hist"].reset_index(drop=True),
            expected_hist.reset_index(drop=True),
            check_names=False,
        )

    def test_bollinger_bands_widen_with_volatility(self, bf):
        """A low-volatility series should produce a narrower bb_width
        than a high-volatility series - sanity check on direction, not
        exact values."""
        n = 40
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        low_vol_close = 100 + np.linspace(0, 1, n)  # nearly flat
        high_vol_close = 100 + \
            np.cumsum(np.random.default_rng(1).normal(0, 3, n))

        def make_df(close):
            return pd.DataFrame({
                "date": dates, "Open": close, "High": close + 0.1,
                "Low": close - 0.1, "Close": close,
                "Volume": np.full(n, 1_000_000),
            })

        
        bf_mod = _import_build_features()
        low_vol_result = bf_mod.compute_technical_features(
            make_df(low_vol_close))
        high_vol_result = bf_mod.compute_technical_features(
            make_df(high_vol_close))

        assert low_vol_result["bb_width"].dropna().mean(
        ) < high_vol_result["bb_width"].dropna().mean()

    def test_atr_non_negative(self, bf, synthetic_ohlcv):
        result = bf.compute_technical_features(synthetic_ohlcv)
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all(), "ATR (a range measure) must never be negative"

    def test_technical_feature_count_is_18(self, bf):
        assert len(bf.TECHNICAL_FEATURE_COLS) == 18


class TestLabelingLogic:
    """Regression tests for the label leakage fix, the single
    most important thing to keep correct in this pipeline."""

    def test_label_uses_next_day_not_same_day_return(self, bf):
        """The label at row i must be derived from returns_1d at row i+1,
        NOT row i - this is the exact bug that was caught and fixed.
        If this regresses, the label becomes trivially derivable from a
        feature the model can see."""
        returns = pd.Series([0.01, -0.01, 0.006, -0.003, 0.0, np.nan])
        # Simulate the actual labeling call used in build_ticker_features:
        # label_movement(returns.shift(-1))
        forward_labels = bf.label_movement(returns.shift(-1))

        # Row 0's label should reflect returns[1] (-0.01 -> DOWN), not returns[0]
        assert forward_labels.iloc[0] == "DOWN", (
            "Label at row 0 should be based on NEXT row's return, not its own"
        )
        # Row 1's label should reflect returns[2] (0.006 -> UP)
        assert forward_labels.iloc[1] == "UP"

    def test_label_matches_threshold(self, bf):
        returns = pd.Series([0.006, -0.006, 0.001, -0.001])
        labels = bf.label_movement(returns, threshold=0.005)
        assert list(labels) == ["UP", "DOWN", "FLAT", "FLAT"]

    def test_last_row_per_ticker_has_no_valid_label(self, bf):
        """After shifting for the forward-looking label, the last day of
        any ticker's series must end up NaN (no 'tomorrow' exists) and
        must be dropped - if it isn't, that's a silent leakage-adjacent
        bug where a fabricated label sneaks into training."""
        returns = pd.Series([0.01, -0.01, 0.02])
        forward_labels = bf.label_movement(returns.shift(-1))
        assert pd.isna(
            forward_labels.iloc[-1]), "Last row must have an undefined forward label"
