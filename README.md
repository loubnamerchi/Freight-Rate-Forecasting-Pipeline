# 🚚 Freight Rate Forecasting Pipeline

An end-to-end **regression ML pipeline** that predicts freight load rates (`posted_rate`) from route, weight, equipment, and market signals — with a chronological (time-aware) train/validation/test methodology, automated hyperparameter tuning, and a dedicated fixed-scenario forecasting mode for isolating date-driven seasonality.

> **Two trained model families ship in this repo:** a full-feature LightGBM model used to score the 12,000-load `validation.csv` submission, and a second, reduced-feature LightGBM model (trained via a separate config) used specifically to generate the fixed December 2025 forecast, since the December scenario only supplies a subset of the original features (route, weight, equipment, date — no live market signals).

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Step-by-Step: What Was Built](#step-by-step-what-was-built)
  - [Step 0 — Notebooks: Exploratory Prototyping](#step-0--notebooks-exploratory-prototyping)
  - [Step 1 — Exploratory Data Analysis](#step-1--exploratory-data-analysis)
  - [Step 2 — Data Ingestion & Preprocessing](#step-2--data-ingestion--preprocessing)
  - [Step 3 — Feature Engineering](#step-3--feature-engineering)
  - [Step 4 — Chronological Train / Validation / Test Split](#step-4--chronological-train--validation--test-split)
  - [Step 5 — Model Training with Optuna](#step-5--model-training-with-optuna)
  - [Step 6 — Model Selection & Evaluation](#step-6--model-selection--evaluation)
  - [Step 7 — Validation Set Scoring](#step-7--validation-set-scoring)
  - [Step 8 — Fixed December Forecast](#step-8--fixed-december-forecast)
- [Model Results](#model-results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Project Overview

| Property | Value |
|---|---|
| **Domain** | Freight / Logistics Rate Prediction |
| **Task** | Regression (predicting `posted_rate` in USD) |
| **Dataset** | 48,000 historical loads (`data/train-test.csv`), Jan 1 – Oct 31, 2025 |
| **Scoring target** | 12,000 unlabeled loads (`data/validation.csv`) |
| **Models compared** | XGBoost, LightGBM, Random Forest (all Optuna-tuned) |
| **Selected model** | LightGBM (lowest validation RMSE, confirmed by test RMSE/MAE/R²) |
| **Split strategy** | Chronological (time-aware), 64% / 16% / 20% |
| **Special feature** | Fixed-scenario December 2025 forecast isolating pure date/seasonality signal |

---

## Architecture

```
notebooks/                                 config.yaml / dec_config.yaml
01_eda.ipynb                                       │
02_preprocessing_&_feature_engineering.ipynb        │
(exploration & prototyping)                         │
         │                                          │
         └──────────────────┬───────────────────────┘
                             ▼
                  data/train-test.csv (48,000 rows)
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                         │
│  DataIngestion → sort by date → chronological split →      │
│  DataPreprocessor (fit on train only) →                    │
│  FeatureEngineer (fit on train only)                       │
│              data/processed/*.parquet                      │
└──────────────────────────┬───────────────────────────────--┘
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                       │
│  Optuna HPO (50 trials, TimeSeriesSplit CV) →               │
│  XGBoost / LightGBM / Random Forest →                       │
│  Final fit on train → evaluate on val & test →               │
│  select best model by validation RMSE                       │
└──────────────────────────┬───────────────────────────────--┘
                            ▼
              ┌─────────────────────────────┐
              │      Saved Artifacts        │
              │  *_regressor.joblib          │
              │  preprocessor.pkl            │
              │  feature_engineer.pkl        │
              │  model_comparison.json       │
              └──────────────┬──────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
  predict_validation.py                predict_december.py
  scores validation.csv                scores the 31-day fixed
  → validation_predictions.csv         December template (dec_* artifacts)
                                        → december_chart_inputs.csv
                                                  │
                                                  ▼
                                             score.py
                                    validates + plots the chart
                                                  │
                                                  ▼
                              scorer_results/candidate_december.png
```

---

## Project Structure

```
freight-rate-prediction/
│
├── config.yaml                     # Main pipeline config (all features, validation.csv scoring)
├── dec_config.yaml                 # December-scenario config (reduced feature set)
├── score.py                        # Provided scorer: validates predictions + plots December chart
├── requirements.txt
│
├── data/
│   ├── train-test.csv              # 48,000 labeled loads (development data)
│   ├── validation.csv              # 12,000 loads requiring predictions
│   ├── validation-predictions-template.csv
│   ├── validation_predictions.csv  # Final submission (load_id, predicted_rate)
│   ├── december-chart-inputs.csv   # 31-day fixed-scenario template (empty predicted_rate)
│   ├── december_chart_inputs.csv   # Filled December predictions
│   ├── interim/                    # Intermediate ingestion output
│   └── processed/                  # Fitted-and-transformed train/val/test parquet files
│
├── artifacts/
│   ├── models/                     # Trained *_regressor.joblib and *_model_dec.joblib files
│   ├── preprocessing/              # Fitted DataPreprocessor + scaler (main and December)
│   ├── feature_engineering/        # Fitted FeatureEngineer + feature_names.json (main and December)
│   └── metrics/                    # model_comparison.json, dec_model_comparison.json
│
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory analysis, 14 EDA figures
│   ├── 02_preprocessing_&_feature_engineering.ipynb  # Prototyping before modularization
│   └── reports/figures/                    # Saved EDA plots (distributions, correlations, maps)
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py       # CSV loading + dataset overview report
│   │   └── data_preprocessing.py   # Dtype fixing, train-only-fitted missing-value imputation
│   ├── features/
│   │   └── build_features.py       # FeatureEngineer: time features, log(distance), encoding
│   ├── models/
│   │   ├── model_factory.py        # Instantiates XGBoost / LightGBM / Random Forest
│   │   ├── train_optuna.py         # Optuna hyperparameter search (TPE, TimeSeriesSplit CV)
│   │   ├── evaluate_model.py       # Cross-validation metrics reporting
│   │   └── compare_models.py       # Selects best model by validation RMSE
│   ├── pipelines/
│   │   ├── data_pipeline.py        # Chronological split → preprocessing → feature engineering
│   │   ├── training_pipeline.py    # Full training entry point (config.yaml)
│   │   └── december_pipeline.py    # December-scenario data pipeline (dec_config.yaml)
│   ├── predictions/
│   │   ├── predict_validation.py   # Scores validation.csv → validation_predictions.csv
│   │   └── predict_december.py     # Scores the December template → december_chart_inputs.csv
│   ├── visualization/
│   │   └── eda_plots.py            # Generates the 14 EDA figures
│   └── utils/
│       ├── config.py                # YAML config loader
│       └── logger.py                # Structured logging setup
│
└── scorer_results/
    └── candidate_december.png       # Chart produced by score.py
```

---

## Step-by-Step: What Was Built

### Step 0 — Notebooks: Exploratory Prototyping

Before writing any production code, `notebooks/01_eda.ipynb` was used to profile the raw 48,000-row dataset — distributions, missing values, duplicates, outlier screening, correlations, and geographic/categorical breakdowns — generating 14 saved figures now in `notebooks/reports/figures/`. `notebooks/02_preprocessing_&_feature_engineering.ipynb` prototyped the cleaning and feature-construction logic before it was refactored into the reusable `DataPreprocessor` and `FeatureEngineer` classes in `src/`.

### Step 1 — Exploratory Data Analysis

Key findings that shaped downstream decisions:

- **Missing values** in `weight` (300 rows) and `market_index` (374 rows) → required imputation.
- **Zero duplicate rows.**
- **Negative and zero weight values** (292 rows, minimum -47,500) flagged as a data-quality issue — documented but not removed in the current pipeline (see [Known Limitations](#known-limitations)).
- **64 unique pickup and 64 unique delivery locations**, and **3 equipment types** → required categorical encoding.
- **Distance is right-skewed** (max 3,439 miles); values judged plausible and retained.
- **`quote_signal` has 1,956 IQR-flagged points**, but they show a systematic, non-linear (inverted-U) relationship with `posted_rate` — judged meaningful, not erroneous, and retained without treatment.
- **Geographic coordinates** fall within valid lat/lon ranges — no treatment needed.
- **Strong linear relationship between `distance` and `posted_rate`** (r ≈ 0.91).
- Data has clear **temporal structure**, motivating a chronological split and time-based features.

### Step 2 — Data Ingestion & Preprocessing

`DataIngestion` logs a full overview (shape, dtypes, missing counts, duplicate count, descriptive stats) at load time. `DataPreprocessor`:

- Casts `date` to a proper datetime type.
- Imputes missing values — **median for numeric columns, mode for categorical columns** — fitted with `.fit_transform()` on the training split only; validation, test, and December data are only ever passed through `.transform()`.

### Step 3 — Feature Engineering

`FeatureEngineer` derives, from `date`: `year`, `month`, `day_of_week`, `is_weekend` (then drops the raw date column). Distance is transformed with `log1p`. Categorical encoding is applied by cardinality: `equipment` (3 categories) → one-hot; `pickup` / `delivery` (64 categories each) → frequency encoding. All encoders are fitted on the training split only and reused via `.transform()` on validation, test, and December data.

### Step 4 — Chronological Train / Validation / Test Split

The split is **chronological, not random** — appropriate because `posted_rate` is time-varying and the ultimate task (the December forecast) requires extrapolating into a future period. The dataset is sorted by date, then split with two sequential 80/20 cuts, executed *before* any preprocessing or feature engineering is fitted:

```
Raw Data (48,000 rows, sorted by date)
        │
        ▼
Cut 1: first 80% / last 20% ───────────► Test (20% · 9,600 rows)
        │
        ▼
Cut 2 (on remaining 80%): first 80% / last 20%
        │                         │
        ▼                         ▼
   Train (64% · 30,720)   Validation (16% · 7,680)
```

### Step 5 — Model Training with Optuna

Three regression candidates — **XGBoost, LightGBM, Random Forest** — are each tuned with **Optuna** (TPE sampler, 50 trials, seed 42; XGBoost/LightGBM additionally use median pruning). Each trial is scored on the **mean RMSE from a 5-fold `TimeSeriesSplit` cross-validation**, computed on the training split only — validation and test are never touched during tuning.

### Step 6 — Model Selection & Evaluation

After tuning, each model is re-evaluated with a fresh 5-fold `TimeSeriesSplit` CV for reporting, then retrained once on the full training split with its tuned hyperparameters. The model with the **lowest validation RMSE** is selected as final:

| Model | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 665.11 | 230.15 | 0.803 | 651.56 | 220.03 | 0.817 |
| **LightGBM (best)** | **643.47** | **138.15** | **0.816** | **633.00** | **124.21** | **0.828** |
| Random Forest | 656.41 | 160.33 | 0.808 | 634.74 | 129.16 | 0.827 |

LightGBM wins on validation RMSE and is also best on every test metric, so the choice isn't an artifact of optimizing a single number.

### Step 7 — Validation Set Scoring

`predict_validation.py` loads the fitted preprocessor, feature engineer, and best model, scores all 12,000 rows of `data/validation.csv`, and writes `validation_predictions.csv` with the required `load_id,predicted_rate` columns.

### Step 8 — Fixed December Forecast

A separate configuration, `dec_config.yaml`, drops six columns unavailable in the December scenario (`pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`, `market_index`, `quote_signal`) and re-runs the **same** pipeline (`run_pipeline()`, chronological split, preprocessing, feature engineering, Optuna tuning, model comparison) end-to-end on the reduced feature set, saving its own artifacts (`dec_*`) so the main model files are never overwritten. `predict_december.py` scores the 31-day fixed template — every field constant except `date` — with the resulting December-specific LightGBM model, and `score.py` (provided) validates the output and renders the chart:

![December 2025 Predicted Load Rate](scorer_results/candidate_december.png)

The repeating weekly pattern (rates climbing through the work week, dropping over the weekend) reflects the `day_of_week` / `is_weekend` seasonality the model has learned for this fixed route.

---

## Model Results

```
Dataset: 48,000 freight loads, Jan 1 – Oct 31, 2025
Train/Val/Test split: 64% / 16% / 20% (chronological)
Scoring set: 12,000 loads (data/validation.csv)
```

| Metric | Value |
|---|---|
| Best model | **LightGBM** |
| Validation RMSE | **643.47** |
| Validation MAE | **138.15** |
| Validation R² | **0.816** |
| Test RMSE | **633.00** |
| Test MAE | **124.21** |
| Test R² | **0.828** |
| Optuna trials per model | **50** |
| CV strategy | **5-fold TimeSeriesSplit** |

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML / Data** | Python, XGBoost, LightGBM, scikit-learn, Optuna, pandas, NumPy, pyarrow |
| **Feature Engineering** | Custom sklearn-style `DataPreprocessor` / `FeatureEngineer` classes |
| **Visualization** | matplotlib, seaborn |
| **Configuration** | YAML config files (`config.yaml`, `dec_config.yaml`) |
| **Scoring / Validation** | Provided `score.py` scorer |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full main pipeline: data prep → training → model comparison
python -m src.pipelines.training_pipeline

# 3. Score the validation set (writes validation_predictions.csv)
python -m src.predictions.predict_validation

# 4. Run the December-scenario data + training pipeline
python -m src.pipelines.december_pipeline

# 5. Score the fixed December template (writes december_chart_inputs.csv)
python -m src.predictions.predict_december

# 6. Validate submission + generate the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs land in `data/validation_predictions.csv` and `scorer_results/candidate_december.png`.

---

## Known Limitations

- **Negative/zero weight values are not corrected.** A row-removal function exists in `DataPreprocessor` but is intentionally disabled, since applying it would change the row count expected by `validation.csv`. These rows (min weight = -47,500) currently flow into training unmodified.
- **Coordinate-range and date-validity checks** were verified manually during EDA but are not enforced as automated checks in the production pipeline.
- **Two distinct LightGBM models exist** — one trained on the full feature set (used for `validation_predictions.csv`) and one trained on a reduced feature set (used only for the December forecast) — because the December scenario doesn't supply every original feature. This is a deliberate design choice, not an inconsistency, but worth knowing when comparing metrics across the two `model_comparison.json` files.

---

## Future Work

- Add an automated weight-validity check (configurable threshold) instead of leaving it disabled.
- Enforce coordinate-range and date-validity checks in the production pipeline, not just the EDA notebook.
- Add feature importance / SHAP analysis to explain individual rate predictions.
- Extend the fixed-scenario evaluation to additional routes beyond Lexington → Fort Wayne.
