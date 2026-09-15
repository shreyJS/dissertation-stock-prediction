"""
Streamlit dashboard for the multimodal stock movement prediction system.

Run locally:
    streamlit run app.py

Reads only precomputed, static artifacts (predictions, SHAP importances,
raw OHLCV) - no live model inference. Run
src/models/generate_dashboard_data.py first and SHAP script to produce the files this app reads.
"""

import os
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Multimodal Stock Movement Prediction", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA_DIR = os.path.join(BASE_DIR, "src", "dashboard", "data")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
EVAL_DIR = os.path.join(BASE_DIR, "src", "eval")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM",
    "AVGO", "NOW", "MU",
]

MODEL_INFO = {
    "Model A (price-only, 12mo)": {"file": "predictions_a.csv", "shap": None},
    "Model B (price-only, 3yr)": {"file": "predictions_b.csv", "shap": None},
    "Model C (multimodal Attention-LSTM)": {"file": "predictions_c.csv", "shap": "model_c_shap_importance.csv"},
    "Model D (multimodal XGBoost)": {"file": "predictions_d.csv", "shap": "model_d_shap_importance.csv"},
}

LABEL_COLORS = {"UP": "#2ecc71", "DOWN": "#e74c3c", "FLAT": "#95a5a6"}


@st.cache_data
def load_predictions(filename):
    path = os.path.join(DASHBOARD_DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_shap(filename):
    if filename is None:
        return None
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_ohlcv(ticker):
    path = os.path.join(DATA_RAW_DIR, f"{ticker}_ohlcv.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[date_col, "Close"]).rename(
        columns={date_col: "date"})
    return df.sort_values("date")


@st.cache_data
def load_metrics_summary():
    metrics = {}
    for label, key in [("A", "model_a_metrics.json"), ("B", "model_b_metrics.json"),
                       ("C", "model_c_metrics.json"), ("D", "model_d_metrics.json")]:
        path = os.path.join(BASE_DIR, "models", key)
        if os.path.exists(path):
            with open(path) as f:
                metrics[label] = json.load(f)
    return metrics


# --- Sidebar ---
st.sidebar.title("Controls")
ticker = st.sidebar.selectbox("Stock", TICKERS)
model_name = st.sidebar.selectbox("Model", list(MODEL_INFO.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Predictions shown are precomputed on the held-out test period "
    "(chronologically the most recent ~10% of the dataset), not live "
    "inference."
)

# --- Header ---
st.title("Multimodal Stock Movement Prediction")
st.caption(f"{ticker} — {model_name}")

# --- Load data ---
preds = load_predictions(MODEL_INFO[model_name]["file"])
shap_df = load_shap(MODEL_INFO[model_name]["shap"])
ohlcv = load_ohlcv(ticker)
all_metrics = load_metrics_summary()

if preds is None:
    st.error(
        f"No prediction data found for {model_name}. Run "
        "`python src/models/generate_dashboard_data.py` first."
    )
    st.stop()

ticker_preds = preds[preds["ticker"] == ticker].sort_values("date")

if ticker_preds.empty:
    st.warning(
        f"No test-period predictions available for {ticker} under this model.")
    st.stop()

# --- Top row: latest prediction + confidence ---
latest = ticker_preds.iloc[-1]
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Most recent test-period prediction", latest["pred_label"])
with col2:
    st.metric("Confidence", f"{latest['confidence']*100:.1f}%")
with col3:
    correct = latest["pred_label"] == latest["true_label"]
    st.metric("Actual outcome", latest["true_label"],
              delta="Correct" if correct else "Incorrect")

st.caption(f"As of {latest['date'].date()}")

# --- Price chart with prediction overlay ---
st.subheader("Price history and predictions")

if ohlcv is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ohlcv["date"], y=ohlcv["Close"], mode="lines",
        name="Close price", line=dict(color="#3498db", width=1.5),
    ))

    for label, color in LABEL_COLORS.items():
        subset = ticker_preds[ticker_preds["pred_label"] == label]
        if subset.empty:
            continue
        merged = subset.merge(ohlcv[["date", "Close"]], on="date", how="left")
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["Close"], mode="markers",
            name=f"Predicted {label}",
            marker=dict(color=color, size=7, symbol="circle"),
        ))

    fig.update_layout(
        height=420, xaxis_title="Date", yaxis_title="Close price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("OHLCV data not found for this ticker.")

# --- Prediction accuracy over test period ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Confidence over the test period")
    fig2 = px.line(ticker_preds, x="date", y="confidence", markers=True)
    fig2.update_traces(line_color="#8e44ad")
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("Test-period accuracy")
    ticker_acc = (ticker_preds["pred_label"] ==
                  ticker_preds["true_label"]).mean()
    st.metric(f"{ticker} accuracy (this model)", f"{ticker_acc*100:.1f}%")
    st.caption(f"n = {len(ticker_preds)} test-period days")

    st.markdown("**Predicted label distribution**")
    dist = ticker_preds["pred_label"].value_counts()
    st.bar_chart(dist)

# --- SHAP feature importance ---
if shap_df is not None:
    st.subheader("Top features by importance (SHAP, global)")
    top15 = shap_df.head(15)
    fig3 = px.bar(
        top15.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature", orientation="h",
    )
    fig3.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Global importance across the test set (not specific to the "
        "selected ticker/date)"
    )
else:
    st.caption(
        "SHAP importance not available for this model (price-only models were not run through SHAP).")

# --- Model comparison ---
st.subheader("All models — test-set summary")
if all_metrics:
    rows = []
    for label, m in all_metrics.items():
        rows.append({
            "Model": label,
            "Test Accuracy": m.get("test_accuracy"),
            "Macro F1": m.get("test_macro_f1"),
            "MCC": m.get("test_mcc"),
        })
    st.dataframe(pd.DataFrame(rows).set_index(
        "Model"), use_container_width=True)
else:
    st.caption("Model metrics JSON files not found in /models.")
