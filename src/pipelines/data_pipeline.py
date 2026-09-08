
from __future__ import annotations

import time

from src.data.data_ingestion     import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.build_features  import FeatureEngineer
from src.utils.logger             import get_logger
from src.utils.config             import load_config

logger = get_logger(__name__)


def run_pipeline(config_path: str = "config.yaml") -> dict:
    t0  = time.time()
    cfg = load_config(config_path)

    # ── 1. Ingest ────────────────────────────────────────────────
    logger.info("\n[1/4] DATA INGESTION")
    ingestor   = DataIngestion(config_path)
    df_raw     = ingestor.load()
    
    if cfg["data"].get("drop_optional_features", False):
        df_raw = df_raw.drop(
            columns=[
                "pickup_lat",
                "pickup_lon",
                "delivery_lat",
                "delivery_lon",
                "market_index",
                "quote_signal",
            ],
            errors="ignore",
        )
        logger.info("Optional features dropped")
    else:
        logger.info("Optional features kept")  
     
    ingestor.print_overview(df_raw)


    # ── 2. SPLIT FIRST — before ANY fitting ──────────────────────
    logger.info("\n[2/4] SPLIT FIRST (before fitting anything)")
    target    = cfg["data"]["target_column"]
    id_column = cfg["data"]["id_column"]

    df = df_raw.sort_values("date").reset_index(drop=True)
    
    test_split_idx = int(len(df) * 0.80)

    train_val_df = df.iloc[:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    val_split_idx = int(len(train_val_df) * 0.80)

    train_df = train_val_df.iloc[:val_split_idx].copy()
    val_df = train_val_df.iloc[val_split_idx:].copy()

    X_train = train_df.drop(columns=[target,id_column])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target,id_column])
    y_val = val_df[target]

    X_test = test_df.drop(columns=[target,id_column])
    y_test = test_df[target]


    logger.info(f"Train: {X_train.shape} | "
                f"Validation: {X_val.shape} | "
                f"Test: {X_test.shape}")

    # ── 3. Preprocess — fit on train ONLY ────────────────────────
    logger.info("\n[3/4] DATA PREPROCESSING")

    preprocessor = DataPreprocessor(config_path)
    train_clean  = preprocessor.fit_transform(X_train)  # FIT on train only
    val_clean    = preprocessor.transform(X_val)         # APPLY to val
    test_clean   = preprocessor.transform(X_test)        # APPLY to test
    preprocessor.save_preprocessor(cfg["paths"]["preprocessor"])
    logger.info("Preprocessor fitted on train only — applied to val and test")


    # ── 4. Feature Engineering — fit on train ONLY ───────────────
    logger.info("\n[4/4] FEATURE ENGINEERING")

    fe          = FeatureEngineer(config_path)
    X_tr_eng    = fe.fit_transform(train_clean)   # FIT on train only
    X_v_eng     = fe.transform(val_clean)        # APPLY to val
    X_te_eng    = fe.transform(test_clean)       # APPLY to test

    fe.save_scaler(cfg["paths"]["scaler"])
    fe.save_feature_names(cfg["paths"]["feature_names"])
    fe.save_feature_engineer(cfg["paths"]["feature_engineer"])
    fe.save_parquets(X_tr_eng,X_v_eng,X_te_eng,y_train,y_val,y_test)

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    
    return {"X_train": X_tr_eng,
        "X_val": X_v_eng,
        "X_test": X_te_eng,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": fe.feature_names_out_,
        "preprocessor": preprocessor,
        "feature_engineer": fe,
        "elapsed_seconds": elapsed,
    }
    
    
if __name__ == "__main__":
    run_pipeline()