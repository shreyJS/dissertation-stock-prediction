"""
Model C - multimodal Attention-LSTM with conformal prediction.

Architecture: same LSTM backbone family as Models A/B, plus an additive
(Bahdanau-style) attention layer pooling over the LSTM's per-timestep
outputs into a context vector, feeding the classification head. Trained
on the full 40-feature multimodal set (technical + sentiment + peer +
macro).

Conformal prediction: MAPIE 1.5.0's SplitConformalClassifier, using the
'calibration' split (untouched during training/early-stopping - that
uses 'validation') to conformalise a wrapped, pre-fitted Keras model.

Known limitation: conformal prediction's exchangeability assumption is
violated by financial time-series (returns are not i.i.d. - volatility
clustering, regime shifts). Coverage is reported empirically on the
test set.

Usage:
    python train_attention_lstm_conformal.py
"""

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from sklearn.utils.class_weight import compute_class_weight
from mapie.classification import SplitConformalClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LOOKBACK_DAYS = 10  # kept consistent with Models A/B
SEED = 42
# 90% target coverage - report empirical coverage
CONFIDENCE_LEVEL = 0.90

# Must match build_features.py exactly (copied rather than imported, to
# keep this script runnable standalone).
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


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


class AdditiveAttention(layers.Layer):
    """Bahdanau-style additive attention pooling over LSTM timesteps.
    Produces a single context vector per sequence, weighted by learned
    relevance of each timestep, rather than just using the last hidden
    state."""

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="att_W", shape=(d, d),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(d,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(d,),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, d)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)  # (batch, timesteps)
        weights = tf.nn.softmax(ait, axis=1)     # attention over timesteps
        weighted = inputs * tf.expand_dims(weights, axis=-1)
        context = tf.reduce_sum(weighted, axis=1)  # (batch, d)
        return context, weights

    def get_config(self):
        return super().get_config()


def build_sequences(df: pd.DataFrame, feature_cols):
    X_list, y_list, split_list = [], [], []
    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        features = tdf[feature_cols].values
        labels = tdf["label"].map(LABEL_MAP).values
        splits = tdf["split"].values

        for i in range(LOOKBACK_DAYS - 1, len(tdf)):
            seq = features[i - LOOKBACK_DAYS + 1: i + 1]
            if np.isnan(seq).any():
                continue
            X_list.append(seq)
            y_list.append(labels[i])
            split_list.append(splits[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    split = np.array(split_list)
    return X, y, split


def build_model(n_timesteps, n_features, n_classes=3):
    # THIRD AND FINAL configuration attempted for Model C, deliberately a
    # middle ground between the two prior runs:
    #   Run 1 (64/32 units, no L2, dropout 0.2, lr 1e-3): OVERFIT -
    #     train acc 63% vs val ~46%, test MCC +0.004 (~chance).
    #   Run 2 (24/16 units, L2 1e-3, dropout 0.3, lr 5e-4): UNDERFIT -
    #     train acc never exceeded ~38%, test MCC -0.021 (~chance, if
    #     anything slightly below).
    # Both converged to essentially MCC=0 on a 407-example test set, so
    # this run is treated as the final word on Model C's architecture
    # regardless of outcome.
    l2 = keras.regularizers.l2(3e-4)

    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = layers.Dropout(0.2)(inputs)
    lstm_out = layers.LSTM(
        40, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.1,
        kernel_regularizer=l2, recurrent_regularizer=l2,
    )(x)
    lstm_out = layers.LSTM(
        24, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.1,
        kernel_regularizer=l2, recurrent_regularizer=l2,
    )(lstm_out)

    context, attn_weights = AdditiveAttention(name="attention")(lstm_out)

    x = layers.Dense(12, activation="relu", kernel_regularizer=l2)(context)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    # for later interpretability (SHAP/attention viz)
    attn_model = keras.Model(inputs=inputs, outputs=attn_weights)

    model.compile(
        # moderate - between run1's 1e-3 and run2's 5e-4
        optimizer=keras.optimizers.Adam(learning_rate=7e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, attn_model


class KerasProbaWrapper:
    """Minimal sklearn-style wrapper so MAPIE's SplitConformalClassifier
    (prefit=True) can call predict_proba/predict on a fitted Keras model."""

    _estimator_type = "classifier"

    def __init__(self, keras_model, classes):
        self.keras_model_ = keras_model
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        return self.keras_model_.predict(X, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def main():
    print("Training Model C (multimodal Attention-LSTM + conformal prediction)...")
    print(
        f"Lookback window: {LOOKBACK_DAYS} days (consistent with Models A/B)\n")

    set_seeds()

    path = os.path.join(DATA_PROCESSED, "features_full.csv")
    if not os.path.exists(path):
        print(
            f"ERROR: {path} not found. Run build_features.py first.")
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print("Building sequences (40 features)...")
    X, y, split = build_sequences(df, ALL_FEATURE_COLS)
    print(f"Total sequences: {len(X)} (shape {X.shape})")

    train_mask = split == "train"
    calib_mask = split == "calibration"
    val_mask = split == "validation"
    test_mask = split == "test"
    print(f"  train: {train_mask.sum()}, calibration: {calib_mask.sum()}, "
          f"validation: {val_mask.sum()}, test: {test_mask.sum()}\n")

    n_train, n_time, n_feat = X[train_mask].shape
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_feat))

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(shape)

    X_train, y_train = scale(X[train_mask]), y[train_mask]
    X_calib, y_calib = scale(X[calib_mask]), y[calib_mask]
    X_val, y_val = scale(X[val_mask]), y[val_mask]
    X_test, y_test = scale(X[test_mask]), y[test_mask]

    majority_class = pd.Series(y_train).mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    print(f"Majority-class baseline (test set): {baseline_acc:.4f} "
          f"(always predicting '{LABEL_NAMES[majority_class]}')\n")

    print("Building and training Attention-LSTM...")
    model, attn_model = build_model(n_time, n_feat)
    model.summary()

    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}
    print(
        f"Class weights (balanced): {dict(zip(LABEL_NAMES, class_weight_values.round(2)))}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    # --- Standard classification evaluation (same sanity checks as A/B) ---
    print("\nEvaluating classification performance on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred)

    print(
        f"Test accuracy: {test_acc:.4f} (majority baseline: {baseline_acc:.4f})")
    print(f"Macro F1: {test_f1_macro:.4f}")
    print(f"MCC: {test_mcc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # --- Conformal prediction (MAPIE, calibration split) ---
    print("=" * 60)
    print("Conformal prediction calibration (MAPIE SplitConformalClassifier)")
    print(f"Target confidence level: {CONFIDENCE_LEVEL}")
    print(
        "NOTE: financial returns violate the i.i.d./exchangeability "
        "assumption conformal prediction relies on (volatility clustering, "
        "regime shifts). Coverage below is reported empirically.\n"
    )

    wrapper = KerasProbaWrapper(model, classes=[0, 1, 2])
    mapie_clf = SplitConformalClassifier(
        estimator=wrapper, confidence_level=CONFIDENCE_LEVEL, prefit=True
    )
    mapie_clf.conformalize(X_calib, y_calib)

    conformal_preds, pred_sets = mapie_clf.predict_set(X_test)
    pred_sets_bool = pred_sets[:, :, 0] if pred_sets.ndim == 3 else pred_sets

    covered = np.array([y_test[i] in np.where(pred_sets_bool[i])[0]
                       for i in range(len(y_test))])
    empirical_coverage = covered.mean()
    avg_set_size = pred_sets_bool.sum(axis=1).mean()

    print(f"Empirical coverage on test set: {empirical_coverage:.4f} "
          f"(target: {CONFIDENCE_LEVEL})")
    print(
        f"Average prediction set size: {avg_set_size:.2f} (out of 3 classes)")

    if empirical_coverage < CONFIDENCE_LEVEL - 0.05:
        print(
            "\nCAUTION: empirical coverage falls meaningfully short of the "
            "target - consistent with the exchangeability violation."
        )
    else:
        print("\nCoverage is close to or above target - reasonable given "
              "the known caveats about financial time-series exchangeability.")

    # Set-size distribution - useful context for the "confidence-utility"
    # analysis planned for SHAP/ablation/regime analysis.
    set_size_counts = pd.Series(pred_sets_bool.sum(
        axis=1)).value_counts().sort_index()
    print(f"\nPrediction set size distribution:\n{set_size_counts}")

    # --- Save everything ---
    model_path = os.path.join(MODELS_DIR, "model_c.keras")
    model.save(model_path)
    attn_model_path = os.path.join(MODELS_DIR, "model_c_attention.keras")
    attn_model.save(attn_model_path)

    metrics = {
        "model": "Model C (Attention-LSTM, multimodal)",
        "lookback_days": LOOKBACK_DAYS,
        "n_train": int(train_mask.sum()),
        "n_calibration": int(calib_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "majority_baseline_acc": float(baseline_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro),
        "test_mcc": float(test_mcc),
        "conformal_confidence_level": CONFIDENCE_LEVEL,
        "conformal_empirical_coverage": float(empirical_coverage),
        "conformal_avg_set_size": float(avg_set_size),
    }
    metrics_path = os.path.join(MODELS_DIR, "model_c_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save prediction sets + true labels for later SHAP/regime analysis
    np.savez(
        os.path.join(MODELS_DIR, "model_c_conformal_predictions.npz"),
        pred_sets=pred_sets_bool, y_test=y_test, y_pred=y_pred,
    )

    print(f"\nSaved model to {model_path}")
    print(f"Saved attention-weights model to {attn_model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved conformal predictions to model_c_conformal_predictions.npz")


if __name__ == "__main__":
    main()
