# Boiler Component Weight Estimator — Streamlit App (Phase 2)

An interactive, dark-mode decision-support tool for estimating boiler
component weights at the conceptual/proposal engineering stage, built on
the Phase 1 synthetic dataset and a Gradient Boosting model trained per
component.

## What's included

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (UI, charts, insights) |
| `train_model.py` | Trains 33 Gradient Boosting models (one per component) and saves them |
| `models/boiler_weight_models.joblib` | Pre-trained model bundle (models + metrics + metadata) |
| `data/boiler_synthetic_dataset.csv` | Phase 1 training data |
| `.streamlit/config.toml` | Dark theme configuration |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt

# (Only needed once, or after regenerating the dataset — a trained model
#  bundle is already included in models/)
python3 train_model.py

streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Deploying

This app is ready for **Streamlit Community Cloud**: push this folder to a
GitHub repo and point Streamlit Cloud at `app.py` — `requirements.txt` and
`.streamlit/config.toml` (dark theme) are picked up automatically. It will
also run unmodified on any host that can run `streamlit run app.py`
(Docker, a VM, Hugging Face Spaces, etc.).

## Model summary

- **Architecture**: one `GradientBoostingRegressor` per output component
  (33 independent models) — chosen over a single multi-output model for
  per-component interpretability (individual R²/feature importance).
- **Performance**: average R² ≈ 0.95, average MAPE ≈ 7.6% on a held-out
  20% test split. The MAPE closely tracks the ±5–10% engineering
  variability injected into the Phase 1 synthetic data — i.e., the model
  is explaining essentially all of the *learnable* signal, and the
  residual error is the intentional noise, not underfitting.
- **Inputs**: Boiler Type (one-hot), Capacity (MW), Steam Flow (TPH),
  Rated Pressure (kg/cm²), Rated Steam Temperature (°C).
- **Outputs**: all 33 component weights (kg) plus total.

## App features

- **Overview** — total weight, category bar chart, category donut chart,
  top-5 heaviest components.
- **Category Breakdown** — expandable detail per system category
  (Pressure Parts, Structural, Thermal, Mechanical, Instrumentation,
  Electrical, Chemical, Miscellaneous).
- **Component Detail** — full 33-row table with per-component model R²,
  CSV export for procurement/costing use.
- **Insights** — automated, decision-oriented callouts: dominant weight
  category, heaviest single item (logistics/crane flag), comparison vs.
  the boiler-type fleet average, type-specific engineering notes, and
  two what-if sensitivity charts (capacity sweep, boiler-type comparison).
- **Model Performance** — global feature importance and per-component
  accuracy table, for transparency on estimate reliability.

## Retraining on new data

If the Phase 1 dataset is regenerated or extended with real project data,
just re-run `python3 train_model.py` — it will overwrite
`models/boiler_weight_models.joblib` and the app will pick up the new
model bundle on next launch (cached with `st.cache_resource`, so restart
the app after retraining).
