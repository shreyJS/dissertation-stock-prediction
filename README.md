# Multimodal Stock Movement Prediction

MSc dissertation project: predicting next-day UP / FLAT / DOWN price movement
for 15 S&P 500 technology stocks, combining price-based technical indicators
with FinBERT/VADER news sentiment. Four models (A, B, C, D) answer five
research questions.

## Quick summary of results

| Model | Description | Test MCC | Macro F1 |
|---|---|---|---|
| A | Price-only LSTM, 12-month training window | 0.022 | 0.36 |
| B | Price-only LSTM, 3-year training window | 0.044 | 0.31 |
| C | Multimodal Attention-LSTM + conformal prediction | 0.045 | 0.32 |
| D | Multimodal XGBoost | −0.048 | 0.24 |

All four models cluster near MCC ≈ 0. Daily-horizon movement is difficult to
predict from this feature set regardless of architecture or the inclusion of
sentiment.

## Repo structure

```
/data
  /raw            OHLCV + news CSVs per ticker, macro.csv (Phase 1)
  /processed      Engineered feature sets (Phase 3)
/models           Trained model files + metrics JSONs (Phases 4-6)
/notebooks        Colab notebook for FinBERT/VADER sentiment scoring (Phase 2)
/sentiment_output Sentiment scores from the Colab notebook
/src
  /ingestion      Finnhub/yfinance backfill scripts
  /features       Feature engineering (technical, sentiment, peer, macro)
  /models         Model training scripts (A, B, C, D) + dashboard data prep
  /eval           SHAP, ablation, regime analysis (Phase 7)
  /dashboard/data Precomputed predictions read by the Streamlit app
/tests            pytest suite - data validation, feature formulas, model
                  sanity checks, dashboard/pipeline consistency
app.py            Streamlit dashboard entry point
requirements.txt
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You'll also need:
- A free [Finnhub](https://finnhub.io/register) API key, set as the
  `FINNHUB_API_KEY` environment variable, for the news backfill.
- A GitHub Personal Access Token if re-running the Colab sentiment notebook
  against a private repo.

## Running the pipeline in order

Only needed to reproduce from scratch — the repo already contains the
generated data/models/predictions from the original run.

```powershell
# Phase 1: data acquisition
python src\ingestion\backfill_all.py
python src\ingestion\backfill_ohlcv_3yr.py       # Model B's 3-year price history
python src\ingestion\validate_phase1.py

# Phase 2: sentiment scoring - run notebooks/phase2_sentiment_pipeline.ipynb
# in Google Colab (GPU runtime), then copy its output zip's contents into
# /sentiment_output at the repo root.

# Phase 3: feature engineering
python src\features\build_features.py
python src\features\build_features_3yr.py

# Phase 4-6: model training
python src\models\train_price_lstm.py A
python src\models\train_price_lstm.py B
python src\models\train_attention_lstm_conformal.py
python src\models\train_xgboost.py

# Phase 7: SHAP, ablation, regime analysis
python src\eval\shap_ablation_regime.py

# Phase 8: dashboard
python src\models\generate_dashboard_data.py
streamlit run app.py

# Phase 9: tests
pytest tests\ -v
```

## Testing

```powershell
pytest tests\ -v
```

94 tests across four files: feature-formula correctness (including
regression tests for a label-leakage fix), data validation (including a
train/test date-leakage check), model sanity (beats a fair majority-class
baseline; conformal calibration sanity), and dashboard/pipeline consistency.
This suite caught a real bug during development.

## Known limitations

- QCOM and AVGO have ~10-11 months of news coverage vs. 12 for other tickers
  (Finnhub free-tier archive depth for these two lower-profile large caps).
- Model C's conformal prediction coverage (85-88% vs. a 90% target) reflects
  a known, expected limitation: financial time-series violate the
  exchangeability assumption conformal prediction relies on.