"""
Phase 4: Models A & B - shared price-only LSTM architecture.

Model A = this architecture trained on the 12-month technical-only set.
Model B = the SAME architecture trained on the 3-year technical-only set.
(Per the proposal: A/B differ by training data VOLUME, not architecture
or lookback window.)

Default assumption: LOOKBACK_DAYS = 10. Sequences are built per-ticker so no
sequence crosses a ticker boundary. A sequence at row t uses features
from [t-LOOKBACK+1, t] to predict the label at row t (which already
represents t -> t+1 movement, per the label-leakage fix). Sequences
whose target falls in the validation/test split are allowed to draw
their LOOKBACK history from an earlier split's rows - this is non-leaking, since at real prediction time for date t you always
have access to the preceding LOOKBACK days regardless of which "split"
they nominally belong to.

Usage:
    python train_price_lstm.py A     # trains Model A (12-month data)
    python train_price_lstm.py B     # trains Model B (3-year data)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LOOKBACK_DAYS = 10  # DEFAULT
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

LABEL_MAP = {"DOWN": 0, "FLAT": 1, "UP": 2}
LABEL_NAMES = ["DOWN", "FLAT", "UP"]

DATASET_CONFIG = {
    "A": {"file": "features_technical_only.csv", "name": "Model A (12-month)"},
    "B": {"file": "features_technical_3yr.csv", "name": "Model B (3-year)"},
}


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_sequences(df: pd.DataFrame):
    """
    Per-ticker sequence construction. Returns X (n_samples, LOOKBACK, 18),
    y (n_samples,), split (n_samples,) aligned arrays.
    """
    X_list, y_list, split_list, date_list, ticker_list = [], [], [], [], []

    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        features = tdf[TECHNICAL_FEATURE_COLS].values
        labels = tdf["label"].map(LABEL_MAP).values
        splits = tdf["split"].values
        dates = tdf["date"].values

        for i in range(LOOKBACK_DAYS - 1, len(tdf)):
            seq = features[i - LOOKBACK_DAYS + 1: i + 1]
            if np.isnan(seq).any():
                continue  # shouldn't happen post-dropna, but safe guard
            X_list.append(seq)
            y_list.append(labels[i])
            split_list.append(splits[i])
            date_list.append(dates[i])
            ticker_list.append(ticker)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    split = np.array(split_list)
    return X, y, split


def build_model(n_timesteps, n_features, n_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(n_timesteps, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("A", "B"):
        print("Usage: python train_price_lstm.py [A|B]")
        sys.exit(1)

    model_key = sys.argv[1]
    config = DATASET_CONFIG[model_key]

    print(f"Training {config['name']}...")
    print(
        f"Lookback window: {LOOKBACK_DAYS} days (default)\n")

    set_seeds()

    path = os.path.join(DATA_PROCESSED, config["file"])
    if not os.path.exists(path):
        print(
            f"ERROR: {path} not found. Run the Phase 3 feature scripts first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print("Building sequences...")
    X, y, split = build_sequences(df)
    print(f"Total sequences: {len(X)} (shape {X.shape})")

    train_mask = split == "train"
    calib_mask = split == "calibration"
    val_mask = split == "validation"
    test_mask = split == "test"

    print(f"  train: {train_mask.sum()}, calibration: {calib_mask.sum()}, "
          f"validation: {val_mask.sum()}, test: {test_mask.sum()}\n")

    # Fit scaler on TRAIN ONLY, apply to all splits - avoids leakage
    # (val/test statistics must never influence the scaling).
    n_train, n_time, n_feat = X[train_mask].shape
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_feat))

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(shape)

    X_train, y_train = scale(X[train_mask]), y[train_mask]
    X_val, y_val = scale(X[val_mask]), y[val_mask]
    X_test, y_test = scale(X[test_mask]), y[test_mask]

    # Majority-class baseline:
    # every model must beat this, or something upstream is broken.
    majority_class = pd.Series(y_train).mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    print(f"Majority-class baseline (test set): {baseline_acc:.4f} "
          f"(always predicting '{LABEL_NAMES[majority_class]}')\n")

    print("Building and training model...")
    model = build_model(n_time, n_feat)
    model.summary()

    # Class weights - without this, the model has no incentive to ever
    # predict FLAT (the minority class, ~21% of data) and collapses to
    # UP/DOWN only, as seen in the first training run.
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}
    print(
        f"Class weights (balanced): {dict(zip(LABEL_NAMES, class_weight_values.round(2)))}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    print("\nEvaluating on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Majority-class baseline: {baseline_acc:.4f}")
    if test_acc <= baseline_acc:
        print("WARNING: model does NOT beat the majority-class baseline. "
              "Something upstream is "
              "likely broken.")
    else:
        print(f"PASS: model beats the majority-class baseline "
              f"(+{(test_acc - baseline_acc)*100:.1f} points).")

    print(f"\nMacro F1: {test_f1_macro:.4f}")
    print(f"MCC: {test_mcc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Save model + metrics + scaler params
    model_path = os.path.join(MODELS_DIR, f"model_{model_key.lower()}.keras")
    model.save(model_path)

    metrics = {
        "model": config["name"],
        "lookback_days": LOOKBACK_DAYS,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "majority_baseline_acc": float(baseline_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro),
        "test_mcc": float(test_mcc),
        "beats_baseline": bool(test_acc > baseline_acc),
    }
    metrics_path = os.path.join(
        MODELS_DIR, f"model_{model_key.lower()}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
