"""
Generates static prediction artifacts for the Streamlit
dashboard. Rebuilds each model's test-set predictions (ticker, date, true
label, predicted label, per-class confidence) and saves them as CSVs.

Doing this as a separate precompute step - rather than having the
dashboard re-run TensorFlow/XGBoost inference live.

Usage:
    python generate_dashboard_data.py
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
DASHBOARD_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "dashboard", "data")
os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)

SEED = 42
LOOKBACK_DAYS = 10

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


def build_sequences(df, feature_cols):
    """Same logic as the training scripts - must match exactly so the
    refit scaler and resulting predictions are consistent with training."""
    X_list, y_list, split_list, date_list, ticker_list = [], [], [], [], []
    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        features = tdf[feature_cols].values
        labels = tdf["y"].values
        splits = tdf["split"].values
        dates = tdf["date"].values
        for i in range(LOOKBACK_DAYS - 1, len(tdf)):
            seq = features[i - LOOKBACK_DAYS + 1: i + 1]
            if np.isnan(seq).any():
                continue
            X_list.append(seq)
            y_list.append(labels[i])
            split_list.append(splits[i])
            date_list.append(dates[i])
            ticker_list.append(ticker)
    return (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32),
            np.array(split_list), np.array(date_list), np.array(ticker_list))


def generate_for_lstm(model_path, feature_cols, source_csv, model_label):
    print(f"Generating predictions for {model_label}...")
    df = pd.read_csv(os.path.join(DATA_PROCESSED, source_csv))
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = df["label"].map(LABEL_MAP)

    X, y, split, dates, tickers = build_sequences(df, feature_cols)
    train_mask = split == "train"
    test_mask = split == "test"

    n_feat = X.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_feat))

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(shape)

    X_test_scaled = scale(X[test_mask])
    y_test = y[test_mask]
    dates_test = dates[test_mask]
    tickers_test = tickers[test_mask]

    custom_objects = {
        "AdditiveAttention": AdditiveAttention} if "attention" in model_label.lower() else {}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    probs = model.predict(X_test_scaled, verbose=0)
    preds = np.argmax(probs, axis=1)

    out = pd.DataFrame({
        "ticker": tickers_test,
        "date": pd.to_datetime(dates_test),
        "true_label": [LABEL_NAMES[i] for i in y_test],
        "pred_label": [LABEL_NAMES[i] for i in preds],
        "prob_down": probs[:, 0],
        "prob_flat": probs[:, 1],
        "prob_up": probs[:, 2],
        "confidence": probs.max(axis=1),
    })
    return out


def generate_for_xgboost(model_path, feature_cols, source_csv):
    print("Generating predictions for Model D (XGBoost)...")
    df = pd.read_csv(os.path.join(DATA_PROCESSED, source_csv))
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = df["label"].map(LABEL_MAP)

    test_df = df[df["split"] == "test"].copy()
    X_test = test_df[feature_cols].values

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    out = pd.DataFrame({
        "ticker": test_df["ticker"].values,
        "date": test_df["date"].values,
        "true_label": test_df["label"].values,
        "pred_label": [LABEL_NAMES[i] for i in preds],
        "prob_down": probs[:, 0],
        "prob_flat": probs[:, 1],
        "prob_up": probs[:, 2],
        "confidence": probs.max(axis=1),
    })
    return out


def main():
    # Model A - technical only, 12mo
    df_a = generate_for_lstm(
        os.path.join(MODELS_DIR, "model_a.keras"),
        TECHNICAL_FEATURE_COLS, "features_technical_only.csv", "model_a",
    )
    df_a.to_csv(os.path.join(DASHBOARD_DATA_DIR,
                "predictions_a.csv"), index=False)
    print(f"  Saved {len(df_a)} rows for Model A\n")

    # Model B - technical only, 3yr
    df_b = generate_for_lstm(
        os.path.join(MODELS_DIR, "model_b.keras"),
        TECHNICAL_FEATURE_COLS, "features_technical_3yr.csv", "model_b",
    )
    df_b.to_csv(os.path.join(DASHBOARD_DATA_DIR,
                "predictions_b.csv"), index=False)
    print(f"  Saved {len(df_b)} rows for Model B\n")

    # Model C - multimodal attention
    df_c = generate_for_lstm(
        os.path.join(MODELS_DIR, "model_c.keras"),
        ALL_FEATURE_COLS, "features_full.csv", "model_c_attention",
    )
    df_c.to_csv(os.path.join(DASHBOARD_DATA_DIR,
                "predictions_c.csv"), index=False)
    print(f"  Saved {len(df_c)} rows for Model C\n")

    # Model D - multimodal XGBoost
    df_d = generate_for_xgboost(
        os.path.join(MODELS_DIR, "model_d.json"),
        ALL_FEATURE_COLS, "features_full.csv",
    )
    df_d.to_csv(os.path.join(DASHBOARD_DATA_DIR,
                "predictions_d.csv"), index=False)
    print(f"  Saved {len(df_d)} rows for Model D\n")

    print("All dashboard prediction data generated.")


if __name__ == "__main__":
    main()
