# Volatility Regimes — Unsupervised + LSTM Supervised + Interactive Dashboard

Discover and monitor **volatility regimes** from options data (WRDS OPTIONM/CRSP), evaluate clustering stability, and serve an **interactive Plotly dashboard** backed by PostgreSQL. Includes an **LSTM supervised** module for forecasting targets (e.g., 30‑day IV).

---

## Table of Contents
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Dashboard (Flask + Plotly + Postgres)](#dashboard-flask--plotly--postgres)
- [Data Uploader (Parquet → Postgres)](#data-uploader-parquet--postgres)
- [Unsupervised Pipeline](#unsupervised-pipeline)
- [LSTM Supervised Module](#lstm-supervised-module)
- [Data Tools & Pipelines](#data-tools--pipelines)
- [Configuration & Environment](#configuration--environment)
- [License](#license)

---

## Architecture

```
.
├─ dashboard_web/
│  ├─ dashboard.py               # Flask server: API + Plotly UI
│  └─ lazy_parquet_to_db.py      # Uploader (last-N-years filter, downsample, COPY streaming)
│
├─ unsupervised/
│  ├─ econclust/
│  │  ├─ __init__.py
│  │  ├─ config.py
│  │  ├─ features.py
│  │  ├─ models.py               # K-Means (W-dist evaluated), Ward scan, metrics
│  │  ├─ scan.py                 # k/threshold scans, stability, ARI bootstraps
│  │  ├─ storage.py
│  │  └─ viz.py
│  ├─ lake/                      # (optional) data cache for remote compute
│  ├─ local_out/                 # local copies of outputs
│  ├─ outputs/
│  │  ├─ frames/                 # parquet datasets
│  │  └─ plots/                  # figures
│  ├─ scripts/
│  │  └─ pipeline.py             # entrypoint for unsupervised pipeline
│  └─ tests/
│
├─ lstm-supervised/
│  ├─ train_lstm_presplit.py     # Train on pre-split parquet (GPU-aware)
│  ├─ model_factory.py           # LSTM/GRU/Seq2Seq + sklearn baselines
│  └─ cv_folds.py                # Expanding-window CV + nested CV utilities
│
├─ data_engineering/
│  ├─ append_features.py         # ATM/skew/curvature (streaming, memory-safe)
│  ├─ consolidate_historical_options_data.py
│  ├─ stock_price_diffs.py       # CRSP streaming + Fibonacci forward diffs
│  ├─ combine_parquet_files.py   # Concatenate/sample parquet sets
│  └─ schema_definitions.py      # Canonical Polars schemas for WRDS
└─ README.md
```

---

## Quickstart

### 1) Install

```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Environment

Create a `.env` (or export these) for the dashboard:

```ini
POSTGRES_URI=postgresql+psycopg2://user:pass@host:5432/dbname
SCHEMA=public
TABLE_NAME=sampled_regimes
LABEL_COL=cluster
API_ROWS_CAP=50000
```

### 3) Load data → Postgres

Prefer **last 5 years** + **daily downsample** to keep size modest.

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

### 4) Launch dashboard (local)

```bash
python ./dashboard_web/dashboard.py
# open http://127.0.0.1:5000
```

---

## Dashboard (Flask + Plotly + Postgres)

- REST API at `/api/data` with limit/offset and schema awareness.
- Auto-detect numeric/datetime columns for X/Y plotting.
- Cluster segmentation by `LABEL_COL` (default `cluster`).
- **Default axes**: `x=impl_volatility`, `y=pca_component_n` (n per chart “window”).
- Front-end: multi-plot grid, color-coded by cluster, sticky X/Y selector panel.

**Render deployment tips**

- Use **Web Service** (not Static Site).
- Start command: `gunicorn dashboard_web.dashboard:app`
- Ensure `gunicorn` is in `requirements.txt`.
- Configure env vars (`POSTGRES_URI`, etc.) in the service settings.

---

## Data Uploader (Parquet → Postgres)

`dashboard_web/lazy_parquet_to_db.py` supports:

- **Column whitelist**: only the following + any PCA components + `date` + `cluster`:

  ```yaml
  Features:
    - Impl_Volatility
    - Delta
    - Theta
    - Vega
    - Moneyness
    - Volume
    - Open_Interest
    - Vol
    - Volume_Ma5
    - Prc
    - Price_Diff_1d
    - Price_Diff_2d
    - Price_Diff_3d
    - Price_Diff_5d
    - Price_Diff_8d
    - Price_Diff_34d
  + any pca_component_* columns
  ```

- **Time filter**: `--last-years N` (e.g., 5) keeps only recent history.
- **Daily downsample**: `--downsample-pct M` samples *within each day*.
- **COPY streaming** for large uploads; `to_sql` for smaller tables.

**Common flags**

```bash
--parquet PATH
--table NAME
--schema public
--if-exists [replace|append]
--create-index "date,cluster"
--last-years 5
--downsample-pct 10
--copy-threshold 500000
--copy-chunksize 200000
--to-sql-chunksize 50000
```

---

## Unsupervised Pipeline

**Goal**: Identify latent volatility regimes using options-derived features, with stability‑driven model selection and hierarchical validation.

### Methods (summary)
- **K‑Means (Primary)** trained on standardized features; **Wasserstein dispersion** is computed per candidate solution (k-scan) for **geometry‑aware model selection** while keeping training fast.
- **Ward Hierarchical Scan (Secondary)** builds a dendrogram (variance‑minimizing linkage) and scans distance thresholds to expose multi‑level structure.

**Model selection**
- For each `k`/threshold: Silhouette, Calinski–Harabasz, Davies–Bouldin, Wasserstein/compactness proxies.
- **Stability** via bootstrap **Adjusted Rand Index (ARI)**; weights α, β tune the tradeoff between compactness and separation.
- Typical winners: `k*` in ~`[7, 17]`; Ward thresholds around `[30, 100]` (dataset-dependent).

**Features**
- Raw Greeks/IV structure plus engineered signals (e.g., `moneyness`, `volume_ma5`, `price_diff_*`).
- **PCA** (~95% variance) for denoising/runtime savings.
- Standardization (`z`‑score) to avoid vega/gamma dominance.

**Interpretation**
- Clusters map to intuitive regimes (e.g., low‑vol carry, transitions, stress).
- Ward confirms **nested** relations (low/med/high) and persistence across time.
- Validation references (no “ground truth”): VIX levels, SPX/ETF drawdowns, realized vol episodes.

---

## LSTM Supervised Module

Located in `lstm-supervised/`. Provides **GPU‑aware** training on pre‑split parquet and a full **nested CV** suite to compare deep learning with traditional baselines.

### Train on pre‑split parquet

`train_lstm_presplit.py`

- **GPU config**: memory growth, visible devices, test compute op.
- **Scaling**: `StandardScaler` (features), `MinMaxScaler` (targets) fit on **train** only.
- **Shapes**: if inputs are tabular rows, code reshapes to `(batch, 1, features)`.
- **Artifacts**: `lstm_model.h5`, `feature_scaler.pkl`, `target_scaler.pkl`, `training_history.csv`, `metrics.csv`, `predictions.csv`.

> **Example** (edit in-file constants or adapt to CLI):

```bash
python lstm-supervised/train_lstm_presplit.py
```

### Model zoo (`model_factory.py`)

- **Deep learning**: `build_lstm`, `build_gru`, `build_seq2seq`.
- **Traditional baselines**: Linear/Ridge/Lasso/ElasticNet, RandomForest, GradientBoosting.

### Cross‑validation & Model selection (`cv_folds.py`)

- **Expanding‑window CV** anchored at distinct market periods.
- **Nested CV**:
  - Traditional models: `GridSearchCV` (inner), custom time splits (outer).
  - LSTM: grid over `units/depth/dropout/learning_rate/W`, evaluated across folds.
- **Comparison**: aggregate RMSE per model; anchor‑specific summaries.

**Typical workflow**

1. Build sequences with `(W, H)` and features → `(N, W, F)` and `(N, H)`.
2. Split by time (train/val/test) or use custom CV folds.
3. Tune via nested CV; track generalization across anchors/regimes.
4. Export metrics/plots; pick best model for downstream forecasting (e.g., 30‑day IV).

---

## Data Tools & Pipelines

- `consolidate_historical_options_data.py` — merge per‑year parquet into a single history (Zstd, row group stats).
- `append_features.py` — streaming computation of **ATM IV**, **skew**, **curvature** using group‑by + joins (memory‑safe).
- `stock_price_diffs.py` — CRSP streaming + **Fibonacci** forward diffs (`price_diff_{n}d`).
- `combine_parquet_files.py` — concatenate/slice large parquet sets; optional sampling.
- `schema_definitions.py` — canonical Polars schemas for WRDS tables.

---

## Configuration & Environment

### Dashboard

```ini
POSTGRES_URI=postgresql+psycopg2://user:pass@host:5432/dbname
SCHEMA=public
TABLE_NAME=sampled_regimes
LABEL_COL=cluster
API_ROWS_CAP=50000
```

### WRDS Access (tools)

- SSL Postgres to `wrds-pgdata.wharton.upenn.edu`.
- Credentials via `PGUSER`/`PGPASSWORD` or `~/.pgpass`.


---

## License

Proprietary / Academic use by project contributors unless otherwise noted. Contact maintainers for reuse permissions.
