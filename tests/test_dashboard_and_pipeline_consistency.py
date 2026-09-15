"""

#4 Dashboard functional check - all 15 stocks load without crashing,
   confidence scores are valid, SHAP data is present where expected.

#2 End-to-end dry run (consistency version) - rather than
   literally re-running the entire multi-hour pipeline from raw data,
   this checks that the final artifacts are mutually consistent (e.g.
   the dashboard's prediction row counts match what each model's
   metrics.json says it evaluated on). A mismatch here usually means
   two pipeline stages were run at different times against different
   versions of the data.

Run:
    pytest tests/test_dashboard_and_pipeline_consistency.py -v
"""

import json
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DATA_DIR = os.path.join(REPO_ROOT, "src", "dashboard", "data")
DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
EVAL_DIR = os.path.join(REPO_ROOT, "src", "eval")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

PREDICTION_FILES = {
    "predictions_a.csv": "model_a_metrics.json",
    "predictions_b.csv": "model_b_metrics.json",
    "predictions_c.csv": "model_c_metrics.json",
    "predictions_d.csv": "model_d_metrics.json",
}


def _require_file(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")


class TestDashboardDataFunctional:

    @pytest.mark.parametrize("pred_file", list(PREDICTION_FILES.keys()))
    def test_predictions_file_has_valid_probabilities(self, pred_file):
        path = os.path.join(DASHBOARD_DATA_DIR, pred_file)
        _require_file(path)
        df = pd.read_csv(path)

        prob_cols = ["prob_down", "prob_flat", "prob_up"]
        assert all(
            c in df.columns for c in prob_cols), f"{pred_file}: missing probability columns"

        prob_sums = df[prob_cols].sum(axis=1)
        assert (prob_sums.between(0.98, 1.02)).all(), (
            f"{pred_file}: class probabilities do not sum to ~1 for some rows"
        )
        for c in prob_cols:
            assert df[c].between(0, 1).all(
            ), f"{pred_file}: {c} has values outside [0, 1]"

    @pytest.mark.parametrize("pred_file", list(PREDICTION_FILES.keys()))
    def test_no_nans_in_predictions(self, pred_file):
        path = os.path.join(DASHBOARD_DATA_DIR, pred_file)
        _require_file(path)
        df = pd.read_csv(path)
        assert df.isna().sum().sum() == 0, f"{pred_file}: contains NaN values"

    @pytest.mark.parametrize("pred_file", list(PREDICTION_FILES.keys()))
    def test_all_15_tickers_loadable_without_crashing(self, pred_file):
        """Simulates what the dashboard does for every ticker in the
        sidebar selector - each one must produce a non-empty, mergeable
        result, or that ticker will silently show nothing/error.
        """
        path = os.path.join(DASHBOARD_DATA_DIR, pred_file)
        _require_file(path)
        preds = pd.read_csv(path)
        preds["date"] = pd.to_datetime(preds["date"])

        failures = []
        for ticker in TICKERS:
            ticker_preds = preds[preds["ticker"] == ticker]
            if ticker_preds.empty:
                failures.append(ticker)
                continue
            ohlcv_path = os.path.join(DATA_RAW, f"{ticker}_ohlcv.csv")
            if not os.path.exists(ohlcv_path):
                failures.append(f"{ticker} (no OHLCV file)")
                continue
            ohlcv = pd.read_csv(ohlcv_path)
            date_col = "Date" if "Date" in ohlcv.columns else ohlcv.columns[0]
            ohlcv[date_col] = pd.to_datetime(ohlcv[date_col], errors="coerce")
            ohlcv = ohlcv.rename(columns={date_col: "date"})
            # the exact merge the dashboard performs for the price-overlay chart
            merged = ticker_preds.merge(
                ohlcv[["date", "Close"]], on="date", how="left")
            if merged["Close"].isna().all():
                failures.append(f"{ticker} (merge produced no matching dates)")

        assert not failures, f"{pred_file}: tickers that would fail/show empty in the dashboard: {failures}"

    def test_shap_importance_files_exist_for_multimodal_models(self):
        for f in ["model_c_shap_importance.csv", "model_d_shap_importance.csv"]:
            path = os.path.join(EVAL_DIR, f)
            if not os.path.exists(path):
                pytest.skip(f"{f} not found - run Phase 7 first")
            df = pd.read_csv(path)
            assert "feature" in df.columns and "mean_abs_shap" in df.columns
            assert len(df) == 40, f"{f}: expected 40 features, found {len(df)}"


class TestEndToEndArtifactConsistency:
    """Cross-checks that separately-generated artifacts agree with each
    other - catches the class of bug where one script was re-run against
    updated data but a downstream artifact wasn't regenerated."""

    @pytest.mark.parametrize("pred_file,metrics_file", list(PREDICTION_FILES.items()))
    def test_dashboard_prediction_count_matches_metrics_test_count(self, pred_file, metrics_file):
        pred_path = os.path.join(DASHBOARD_DATA_DIR, pred_file)
        metrics_path = os.path.join(MODELS_DIR, metrics_file)
        _require_file(pred_path)
        _require_file(metrics_path)

        preds = pd.read_csv(pred_path)
        with open(metrics_path) as f:
            metrics = json.load(f)

        expected_n_test = metrics.get("n_test")
        if expected_n_test is None:
            pytest.skip(f"{metrics_file} has no n_test field")

        assert len(preds) == expected_n_test, (
            f"{pred_file} has {len(preds)} rows but {metrics_file} reports "
            f"n_test={expected_n_test} - these were likely generated from "
            f"different runs of the pipeline; regenerate dashboard data "
            f"after any retraining"
        )

    def test_all_four_models_present(self):
        missing = []
        for metrics_file in PREDICTION_FILES.values():
            path = os.path.join(MODELS_DIR, metrics_file)
            if not os.path.exists(path):
                missing.append(metrics_file)
        assert not missing, f"Missing model metrics files: {missing} - not all models have been trained yet"
