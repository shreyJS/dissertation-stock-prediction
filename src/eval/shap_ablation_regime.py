"""
Phase 7:

Requires: Model D already trained (model_d.json) and Model C already
trained (model_c.keras) from the previous phases.

Usage:
    python shap_ablation_regime.py
"""

import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "src", "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

SEED = 42
LOOKBACK_DAYS = 10
SHAP_BACKGROUND_SIZE = 50  

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

LABEL_MAP = {"DOWN": 0, "FLAT": 1, "UP": 2}
LABEL_NAMES = ["DOWN", "FLAT", "UP"]


class AdditiveAttention(layers.Layer):
    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="att_W", shape=(
            d, d), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(
            d,), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(
            d,), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        weights = tf.nn.softmax(ait, axis=1)
        weighted = inputs * tf.expand_dims(weights, axis=-1)
        context = tf.reduce_sum(weighted, axis=1)
        return context, weights


# ---------------------------------------------------------------------
# Section 1: Feature ablation
# ---------------------------------------------------------------------

def run_ablation(df):
    print("=" * 60)
    print("SECTION 1: Feature ablation (price-only 18 vs full 40)")
    print("Same rows, same split, same model type (XGBoost) - isolates")
    print("the effect of adding sentiment/peer/macro features.\n")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    y_train, y_val, y_test = train_df["y"].values, val_df["y"].values, test_df["y"].values

    results = {}
    for name, cols in [("price_only_18", TECHNICAL_FEATURE_COLS), ("full_40", ALL_FEATURE_COLS)]:
        X_train = train_df[cols].values
        X_val = val_df[cols].values
        X_test = test_df[cols].values

        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=SEED, early_stopping_rounds=20,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        results[name] = {"n_features": len(
            cols), "accuracy": acc, "macro_f1": f1_macro, "mcc": mcc}
        print(
            f"[{name}] n_features={len(cols)}, acc={acc:.4f}, macro_f1={f1_macro:.4f}, mcc={mcc:.4f}")

    delta_mcc = results["full_40"]["mcc"] - results["price_only_18"]["mcc"]
    delta_f1 = results["full_40"]["macro_f1"] - \
        results["price_only_18"]["macro_f1"]
    print(f"\nDelta from adding sentiment/peer/macro (22 features): "
          f"MCC {delta_mcc:+.4f}, macro F1 {delta_f1:+.4f}")
    if delta_mcc > 0.01 or delta_f1 > 0.01:
        print("Sentiment/macro features provide a modest but real improvement.")
    else:
        print("Sentiment/macro features do not provide a clear improvement on this "
              "held-out test set - a legitimate, reportable finding given the RQs.")

    return results


# ---------------------------------------------------------------------
# Section 2: Regime analysis
# ---------------------------------------------------------------------

def run_regime_analysis(df):
    print("\n" + "=" * 60)
    print("SECTION 2: Regime analysis (VIX high/low median split)\n")

    test_df = df[df["split"] == "test"].copy()
    # threshold from TRAIN only, no leakage
    vix_median = df[df["split"] == "train"]["macro_vix"].median()
    test_df["vix_regime"] = np.where(
        test_df["macro_vix"] >= vix_median, "high_vix", "low_vix")

    print(f"VIX median (from train split): {vix_median:.2f}")
    print(test_df["vix_regime"].value_counts())

    # Load Model D (XGBoost) and evaluate per-regime
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "model_d.json"))

    results = {}
    for regime in ["high_vix", "low_vix"]:
        sub = test_df[test_df["vix_regime"] == regime]
        X_sub = sub[ALL_FEATURE_COLS].values
        y_sub = sub["y"].values
        y_pred = model.predict(X_sub)

        acc = accuracy_score(y_sub, y_pred)
        f1_macro = f1_score(y_sub, y_pred, average="macro")
        mcc = matthews_corrcoef(y_sub, y_pred) if len(
            set(y_sub)) > 1 else float("nan")

        results[regime] = {
            "n": len(sub), "accuracy": acc, "macro_f1": f1_macro, "mcc": mcc}
        print(
            f"[{regime}] n={len(sub)}, acc={acc:.4f}, macro_f1={f1_macro:.4f}, mcc={mcc:.4f}")

    return results, vix_median


# ---------------------------------------------------------------------
# Section 3: SHAP for Model D (TreeExplainer - fast, exact)
# ---------------------------------------------------------------------

def run_shap_model_d(df):
    print("\n" + "=" * 60)
    print("SECTION 3: SHAP for Model D (XGBoost, TreeExplainer)\n")

    test_df = df[df["split"] == "test"]
    X_test = test_df[ALL_FEATURE_COLS].values

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "model_d.json"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shape: (n_samples, n_features, n_classes) in this shap version
    mean_abs_shap = np.abs(shap_values).mean(
        axis=(0, 2))  # average over samples AND classes

    importance_df = pd.DataFrame({
        "feature": ALL_FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("Top 15 features by mean |SHAP value| (Model D):")
    print(importance_df.head(15).to_string(index=False))

    importance_df.to_csv(os.path.join(
        EVAL_DIR, "model_d_shap_importance.csv"), index=False)
    np.save(os.path.join(EVAL_DIR, "model_d_shap_values.npy"), shap_values)

    return importance_df


# ---------------------------------------------------------------------
# Section 4: SHAP for Model C
# ---------------------------------------------------------------------

def build_sequences(df, feature_cols):
    X_list, y_list, split_list = [], [], []
    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        features = tdf[feature_cols].values
        labels = tdf["y"].values
        splits = tdf["split"].values
        for i in range(LOOKBACK_DAYS - 1, len(tdf)):
            seq = features[i - LOOKBACK_DAYS + 1: i + 1]
            if np.isnan(seq).any():
                continue
            X_list.append(seq)
            y_list.append(labels[i])
            split_list.append(splits[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(split_list)


def run_shap_model_c(df):
    print("\n" + "=" * 60)
    print("SECTION 4: SHAP for Model C (Attention-LSTM, GradientExplainer)")
    print(
        f"Using reduced background sample (n={SHAP_BACKGROUND_SIZE})\n")

    model = keras.models.load_model(
        os.path.join(MODELS_DIR, "model_c.keras"),
        custom_objects={"AdditiveAttention": AdditiveAttention},
    )

    X, y, split = build_sequences(df, ALL_FEATURE_COLS)
    train_mask = split == "train"
    test_mask = split == "test"

    n_feat = X.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_feat))

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(shape)

    X_train_scaled = scale(X[train_mask])
    X_test_scaled = scale(X[test_mask])

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train_scaled), size=min(
        SHAP_BACKGROUND_SIZE, len(X_train_scaled)), replace=False)
    background = X_train_scaled[bg_idx]

    # Explain a sample of the test set (not all 407) to keep runtime bounded
    explain_idx = rng.choice(len(X_test_scaled), size=min(
        100, len(X_test_scaled)), replace=False)
    X_explain = X_test_scaled[explain_idx]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_explain)
    # shape: (n_explain, timesteps, n_features, n_classes)

    # Average over samples, timesteps, and classes for a global ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1, 3))

    importance_df = pd.DataFrame({
        "feature": ALL_FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("Top 15 features by mean |SHAP value| (Model C):")
    print(importance_df.head(15).to_string(index=False))

    importance_df.to_csv(os.path.join(
        EVAL_DIR, "model_c_shap_importance.csv"), index=False)
    np.save(os.path.join(EVAL_DIR, "model_c_shap_values.npy"), shap_values)

    return importance_df


def main():
    path = os.path.join(DATA_PROCESSED, "features_full.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = df["label"].map(LABEL_MAP)

    ablation_results = run_ablation(df)
    regime_results, vix_median = run_regime_analysis(df)
    shap_d = run_shap_model_d(df)
    shap_c = run_shap_model_c(df)

    # Save consolidated results
    summary = {
        "ablation": ablation_results,
        "regime": {k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                       for kk, vv in v.items()} for k, v in regime_results.items()},
        "vix_median_threshold": float(vix_median),
        "shap_model_d_top10": shap_d.head(10).to_dict("records"),
        "shap_model_c_top10": shap_c.head(10).to_dict("records"),
    }
    summary_path = os.path.join(EVAL_DIR, "phase7_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved full summary to {summary_path}")
    print("Individual SHAP importance CSVs and raw .npy values also saved to src/eval/")


if __name__ == "__main__":
    main()
