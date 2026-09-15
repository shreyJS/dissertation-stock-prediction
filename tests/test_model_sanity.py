"""
Model sanity checks - each model must beat a majority-class baseline, or something upstream is
broken (not just under-tuned).

IMPORTANT ADAPTATION: raw accuracy is a biased comparison
under class imbalance (a model that only ever predicts the majority
class "wins" on accuracy while being useless). So this test compares
against the majority-class baseline's MACRO F1 instead, computed fresh
from each dataset's actual test-split class distribution.

Run:
    pytest tests/test_model_sanity.py -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DATA_PROCESSED = os.path.join(REPO_ROOT, "data", "processed")

LABEL_MAP = {"DOWN": 0, "FLAT": 1, "UP": 2}

MODEL_CONFIG = [
    ("model_a_metrics.json", "features_technical_only.csv"),
    ("model_b_metrics.json", "features_technical_3yr.csv"),
    ("model_c_metrics.json", "features_full.csv"),
    ("model_d_metrics.json", "features_full.csv"),
]


def _require_file(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not found - train this model first")


def _majority_baseline_macro_f1(feature_file):
    """Computes the macro F1 a majority-class-only classifier would get
    on this dataset's actual test split."""
    path = os.path.join(DATA_PROCESSED, feature_file)
    _require_file(path)
    df = pd.read_csv(path)
    df["y"] = df["label"].map(LABEL_MAP)

    train_y = df[df["split"] == "train"]["y"]
    test_y = df[df["split"] == "test"]["y"]
    if train_y.empty or test_y.empty:
        pytest.skip(f"{feature_file}: train/test split empty")

    majority_class = train_y.mode()[0]
    baseline_preds = np.full(len(test_y), majority_class)
    return f1_score(test_y, baseline_preds, average="macro")


class TestModelBeatsBaseline:

    @pytest.mark.parametrize("metrics_file,feature_file", MODEL_CONFIG)
    def test_model_beats_majority_baseline_macro_f1(self, metrics_file, feature_file):
        metrics_path = os.path.join(MODELS_DIR, metrics_file)
        _require_file(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)

        baseline_macro_f1 = _majority_baseline_macro_f1(feature_file)
        model_macro_f1 = metrics["test_macro_f1"]

        assert model_macro_f1 > baseline_macro_f1, (
            f"{metrics_file}: model macro F1 ({model_macro_f1:.4f}) does not "
            f"beat the majority-class baseline's macro F1 ({baseline_macro_f1:.4f}) "
            f"this suggests something "
            f"upstream may be broken, not just under-tuned"
        )

    @pytest.mark.parametrize("metrics_file,feature_file", MODEL_CONFIG)
    def test_mcc_is_not_strongly_negative(self, metrics_file, feature_file):
        """A strongly negative MCC (systematically anti-correlated with
        truth) usually indicates a real bug - e.g. a flipped label
        encoding - rather than just weak signal. Small positive/negative
        values near zero are expected and fine."""
        metrics_path = os.path.join(MODELS_DIR, metrics_file)
        _require_file(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        mcc = metrics["test_mcc"]
        assert mcc > -0.15, (
            f"{metrics_file}: MCC ({mcc:.4f}) is strongly negative - this is "
            f"different from the small-negative-to-small-positive range seen "
            f"across all four models, and likely indicates a real bug "
            f"(e.g. a label/class-index mismatch) rather than weak signal"
        )


class TestConformalCalibration:
    """Sanity check on Model C's MAPIE conformal calibration - not a
    correctness proof, but catches the class of bug where the API
    integration silently breaks (e.g. after a library upgrade) and
    coverage collapses to something absurd like 0% or 100%."""

    def test_conformal_coverage_in_plausible_range(self):
        metrics_path = os.path.join(MODELS_DIR, "model_c_metrics.json")
        _require_file(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)

        coverage = metrics.get("conformal_empirical_coverage")
        target = metrics.get("conformal_confidence_level")
        if coverage is None:
            pytest.skip(
                "model_c_metrics.json has no conformal coverage recorded")

        assert 0.5 <= coverage <= 1.0, (
            f"Empirical coverage ({coverage:.4f}) is outside a plausible range "
            f"- likely indicates the conformal calibration step is broken, "
            f"not just affected by the known exchangeability-violation caveat"
        )

    def test_conformal_avg_set_size_is_sane(self):
        metrics_path = os.path.join(MODELS_DIR, "model_c_metrics.json")
        _require_file(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        avg_size = metrics.get("conformal_avg_set_size")
        if avg_size is None:
            pytest.skip(
                "model_c_metrics.json has no conformal set size recorded")
        assert 1.0 <= avg_size <= 3.0, (
            f"Average prediction set size ({avg_size:.2f}) must be between "
            f"1 and 3 (the number of classes) - a value outside this range "
            f"indicates a bug in the prediction-set construction"
        )
