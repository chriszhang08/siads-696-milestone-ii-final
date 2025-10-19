# Volatility Regimes — Unsupervised + LSTM Supervised + Interactive Dashboard

Discover and monitor **volatility regimes** from options data (WRDS OPTIONM/CRSP), evaluate clustering stability, and serve an **interactive Plotly dashboard** backed by PostgreSQL. Includes an **LSTM** module for supervised forecasting (e.g., 30-day IV).

---

## Contents
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Unsupervised (econclust)](#unsupervised-econclust)
- [Dashboard (Flask + Plotly + Postgres)](#dashboard-flask--plotly--postgres)
- [Data Uploader (Parquet → Postgres)](#data-uploader-parquet--postgres)
- [LSTM Supervised Module](#lstm-supervised-module)
- [Data Tools & Pipelines](#data-tools--pipelines)
- [Configuration & Environment](#configuration--environment)
- [Ethical Considerations](#ethical-considerations)

---

## Architecture

```
.
├─ dashboard_web/
│  ├─ dashboard.py               # Flask server: API + Plotly UI
│  └─ lazy_parquet_to_db.py      # Uploader (date filter, downsample, COPY streaming)
│
├─ lstm-supervised/
│  ├─ train_lstm_presplit.py     # Train on pre-split parquet (GPU-aware)
│  ├─ model_factory.py           # LSTM/GRU/Seq2Seq + sklearn baselines
│  └─ cv_folds.py                # Expanding-window CV + nested CV utilities
│
├─ unsupervised/
│  ├─ econclust/
│  │  ├─ __init__.py  config.py  features.py  models.py  scan.py  storage.py  viz.py
│  ├─ outputs/
│  │  ├─ frames/                 # clustered parquet outputs
│  │  └─ plots/                  # metric scans, dendrograms, stability plots
│  └─ scripts/pipeline.py        # main unsupervised pipeline entrypoint
│
├─ data_tools/
│  ├─ append_features.py                 # ATM/skew/curvature (streaming)
│  ├─ consolidate_historical_options_data.py
│  ├─ words_stock_prc_generator.py       # CRSP streaming + Fibonacci price diffs
│  ├─ parquet_collator.py                # Concatenate / sample parquet
│  └─ schema_definitions.py              # Polars schemas for WRDS tables
└─ README.md
```

---

## Quickstart

### 1) Install
```bash
python -m venv venv
source venv/bin/activate      # (Windows) venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Environment (dashboard)
Create a `.env` (or export these):
```bash
POSTGRES_URI=postgresql+psycopg2://user:pass@host:5432/dbname
SCHEMA=public
TABLE_NAME=sampled_regimes
LABEL_COL=cluster
API_ROWS_CAP=50000
```

### 3) Load data to Postgres
Prefer **last 5 years** + **daily downsample** to keep size in check:
```bash
python ./dashboard_web/lazy_parquet_to_db.py \
  --parquet ./data/SPY_pca_kmeans.parquet \
  --postgres-uri "postgresql+psycopg2://USER:PASS@HOST:PORT/DB" \
  --table sampled_regimes \
  --if-exists replace \
  --create-index "date,cluster" \
  --last-years 5 \
  --downsample-pct 10
```

### 4) Run dashboard locally
```bash
python ./dashboard_web/dashboard.py
# open http://127.0.0.1:5000
```

---

## Unsupervised (econclust)

The **econclust** pipeline discovers geometry-aware market *regimes* from implied volatility (IV) surfaces and Greeks. It combines flat clustering clarity with hierarchical interpretability.

### Methods
- **K-Means + Wasserstein evaluation (Primary)**  
  Train K-Means on standardized features (raw + PCA). For each *k*, compute a Wasserstein dispersion proxy across features/clusters to incorporate distributional geometry. Scan metrics per *k*: inertia, Silhouette, Calinski–Harabasz (CH), Davies–Bouldin (DB), Wasserstein proxy. Select *k★* via an ensemble + bootstrap stability (Adjusted Rand Index, ARI).
- **Ward Scan (Secondary)**  
  Ward’s minimum variance linkage; scan distance thresholds *t*. Evaluate Silhouette/CH/DB + a compactness proxy, then pick a consensus threshold with bootstrap ARI.

### Inputs
- Options frame with:  
  `impl_volatility, delta, theta, vega, moneyness, volume, open_interest, vol, volume_ma5, prc, price_diff_1d, price_diff_2d, price_diff_3d, price_diff_5d, price_diff_8d, price_diff_34d`  
  plus any `pca_component_*`, and `date`.

### Example config
```json
{
  "data": {
    "input_parquet": "path/to/options_frame.parquet",
    "date_col": "date",
    "feature_cols": [
      "impl_volatility","delta","theta","vega","moneyness",
      "volume","open_interest","vol","volume_ma5","prc",
      "price_diff_1d","price_diff_2d","price_diff_3d",
      "price_diff_5d","price_diff_8d","price_diff_34d"
    ],
    "include_pca": true,
    "pca_variance": 0.95
  },
  "kmeans": {
    "k_grid": [2,3,4,5,6,8,10,12,14,16,18,20],
    "stability": {"alpha": 0.25, "beta": 0.25, "bootstraps": 5, "sample_frac": 0.7}
  },
  "ward": {
    "thresholds": [30,40,50,60,70,80,90,100],
    "stability": {"alpha": 0.25, "beta": 0.25, "bootstraps": 4, "sample_frac": 0.5}
  },
  "output": {
    "frames_dir": "unsupervised/outputs/frames",
    "plots_dir": "unsupervised/outputs/plots"
  }
}
```

---

## Dashboard (Flask + Plotly + Postgres)

Features:
- REST API (`/api/data`) with pagination from Postgres
- Auto-detected numeric/datetime columns for XY plotting
- Cluster split by `LABEL_COL` (default `cluster`)
- Default XY preference: `x=impl_volatility`, `y=pca_component_n` (if present)

---

*Questions or contributions? Open an issue or PR.*

