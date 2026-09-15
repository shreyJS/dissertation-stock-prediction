"""
Phase 6: Model D - multimodal XGBoost.

Uses the same 40-feature multimodal set and chronological split as
Model C, but XGBoost consumes flat feature vectors. This
means Model D is not hit by the same train-set-size squeeze Model C
faced from sequence construction, though it is still bounded by the same
sentiment-alignment-constrained ~2,486 total labelled rows.

Usage:
    python train_xgboost.py
"""

import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from sklearn.utils.class_weight import compute_sample_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42

TECHNICAL_FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_10d",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio",
    "ema_10_ratio", "ema_20_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper_dist", "bb_lower_dist", "bb_width",
    "atr_14",
    "volume_change_1d",
    "obv_zscore",
]
SENTIMENT_FEATURE_COLS = [
    "sent_finbert_mean", "sent_finbert_std", "sent_vader_mean", "sent_n_articles",
    "sent_finbert_positive_flag", "sent_finbert_negative_flag",
    "sent_finbert_mean_3d", "sent_vader_mean_3d", "sent_finbert_std_3d", "sent_n_articles_3d",
    "sent_finbert_mean_7d", "sent_vader_mean_7d", "sent_finbert_std_7d", "sent_n_articles_7d",
    "sent_momentum_1d", "sent_momentum_7d",
    "sent_finbert_vader_divergence",
]
PEER_FEATURE_COLS = ["peer_sent_finbert_mean", "peer_sent_momentum_1d"]
MACRO_FEATURE_COLS = ["macro_vix", "macro_tnx_10y", "macro_earnings_window"]
ALL_FEATURE_COLS = TECHNICAL_FEATURE_COLS + \
    SENTIMENT_FEATURE_COLS + PEER_FEATURE_COLS + MACRO_FEATURE_COLS
assert len(ALL_FEATURE_COLS) == 40

LABEL_MAP = {"DOWN": 0, "FLAT": 1, "UP": 2}
LABEL_NAMES = ["DOWN", "FLAT", "UP"]


def main():
    print("Training Model D (multimodal XGBoost)...\n")

    path = os.path.join(DATA_PROCESSED, "features_full.csv")
    if not os.path.exists(path):
        print(
            f"ERROR: {path} not found. Run Phase 3 (build_features.py) first.")
        return

    df = pd.read_csv(path)
    df["y"] = df["label"].map(LABEL_MAP)

    train_df = df[df["split"] == "train"]
    calib_df = df[df["split"] == "calibration"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    print(f"Rows -> train: {len(train_df)}, calibration: {len(calib_df)}, "
          f"validation: {len(val_df)}, test: {len(test_df)}\n")

    X_train, y_train = train_df[ALL_FEATURE_COLS].values, train_df["y"].values
    X_val, y_val = val_df[ALL_FEATURE_COLS].values, val_df["y"].values
    X_test, y_test = test_df[ALL_FEATURE_COLS].values, test_df["y"].values

    majority_class = pd.Series(y_train).mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    print(f"Majority-class baseline (test set): {baseline_acc:.4f} "
          f"(always predicting '{LABEL_NAMES[majority_class]}')\n")

    # Sample weights for class imbalance - XGBoost's equivalent of the
    # class_weight approach used for Models A/B/C.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    # NOTE: also weighting the validation set for early stopping was
    # tried and reverted - on this small a validation set (407 rows) it
    # made the early-stopping signal too noisy (best_iteration landed at
    # 0, i.e. essentially untrained, and MCC dropped further). Weighting
    # only the training objective, with an unweighted early-stopping
    # signal, is the more stable choice here and is kept as final.

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,             # shallow trees - only ~1,364 training rows, avoid overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
        early_stopping_rounds=20,
    )

    print("Training with early stopping on validation set...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration} (of 300 max)\n")

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred)

    print(
        f"Test accuracy: {test_acc:.4f} (majority baseline: {baseline_acc:.4f})")
    if test_acc <= baseline_acc:
        print("NOTE: does not beat raw majority-class accuracy - check macro F1/MCC "
              "below before concluding this is uninformative (raw accuracy is a "
              "biased comparison under class imbalance, as established with Models A/B/C).")
    print(f"Macro F1: {test_f1_macro:.4f}")
    print(f"MCC: {test_mcc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Feature importance - quick look now; full SHAP analysis is Phase 7
    importances = model.feature_importances_
    top_features = sorted(
        zip(ALL_FEATURE_COLS, importances), key=lambda x: -x[1])[:10]
    print("Top 10 features by XGBoost gain-based importance:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    # --- Save ---
    model_path = os.path.join(MODELS_DIR, "model_d.json")
    model.save_model(model_path)

    metrics = {
        "model": "Model D (multimodal XGBoost)",
        "n_train": len(train_df),
        "n_calibration": len(calib_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_iteration": int(model.best_iteration),
        "majority_baseline_acc": float(baseline_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro),
        "test_mcc": float(test_mcc),
        "beats_baseline_accuracy": bool(test_acc > baseline_acc),
        "top_10_features": [{"feature": f, "importance": float(i)} for f, i in top_features],
    }
    metrics_path = os.path.join(MODELS_DIR, "model_d_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
