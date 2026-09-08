from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib

from src.models.train_optuna import run_optuna
from src.models.evaluate_model import cross_validate_model
from src.models.model_factory import create_model
from src.models.model_factory import get_default_params
from src.models.compare_models import train_and_compare_models

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def train_model(config_path: str = "config.yaml",):

    cfg = load_config(config_path)

    models = cfg.get("models",["xgboost","lightgbm","random_forest",],)
    random_state = cfg["training"].get("random_state", 42,)
    
    logger.info( "================ STARTING MODEL TRAINING =====================")
    logger.info("Models: %s",models,)
    
    # 1. LOAD DATA

    logger.info("\n[1/6] LOAD PROCESSED DATA")

    X_train = pd.read_parquet(cfg["paths"]["processed_train"])
    X_val = pd.read_parquet(cfg["paths"]["processed_val"])
    X_test = pd.read_parquet(cfg["paths"]["processed_test"])
    y_train = pd.read_parquet(cfg["paths"]["processed_y_train"]).squeeze()
    y_val = pd.read_parquet(cfg["paths"]["processed_y_val"]).squeeze()
    y_test = pd.read_parquet(cfg["paths"]["processed_y_test"]).squeeze()
    
    logger.info("Train: %s",X_train.shape,)
    logger.info("Validation: %s",X_val.shape,)
    logger.info("Test: %s",X_test.shape,)

    feature_names = X_train.columns.tolist()

    logger.info("Number of features: %d",len(feature_names),)

    
    # 2. OPTUNA
   
    logger.info( "\n[2/6] OPTUNA HYPERPARAMETER TUNING")

    best_params = {}
    optuna_results = {}

    for model_name in models:

        logger.info("\n----------------------------------------")
        logger.info("OPTUNA: %s",model_name.upper(),)

        if cfg["optuna"].get("enabled",True,):
            model_best_params,best_cv_rmse,study = run_optuna(model_name=model_name,X_train=X_train,
                                                                 y_train=y_train,cfg=cfg,)
            best_params[model_name] = model_best_params
            optuna_results[model_name] = {"best_cv_rmse": float(best_cv_rmse),
                                          "best_params":model_best_params,
                                          "n_trials":len(study.trials),}
            logger.info("%s best CV RMSE: %.4f",model_name,best_cv_rmse,)
            logger.info("%s best params: %s",model_name,model_best_params,)

        else:
            logger.info("Optuna disabled for %s",model_name,)
            best_params[model_name] = get_default_params(model_name,cfg,)


    # 3. CROSS VALIDATION
    
    logger.info("\n[3/6] FINAL TIME-SERIES CROSS-VALIDATION")

    cv_results = {}
    n_splits = cfg["cross_validation"].get("n_splits",5,)

    for model_name in models:
        logger.info("\nCV: %s",model_name.upper(),)
        model = create_model(model_name=model_name,params=best_params[model_name],
                             random_state=random_state,)
        cv_stats = cross_validate_model(model=model,X=X_train,y=y_train,n_splits=n_splits,)
        cv_results[model_name] = cv_stats

        logger.info("%s CV RMSE: %.4f ± %.4f",model_name,cv_stats["cv_rmse_mean"],cv_stats["cv_rmse_std"],)
        logger.info("%s CV MAE: %.4f ± %.4f",model_name,cv_stats["cv_mae_mean"],cv_stats["cv_mae_std"],)
        logger.info("%s CV R²: %.4f ± %.4f",model_name,cv_stats["cv_r2_mean"],cv_stats["cv_r2_std"],)

    # 4. FINAL TRAINING + COMPARISON
    
    logger.info("\n[4/6] FINAL MODEL TRAINING")

    trained_models,comparison_results = train_and_compare_models(models=models,X_train=X_train,y_train=y_train,
                                                                    X_val=X_val,y_val=y_val,
                                                                    X_test=X_test,y_test=y_test,
                                                                    best_params=best_params,cfg=cfg,)

    # 5. ADD CV + OPTUNA RESULTS
  
    logger.info("\n[5/6] BUILD FINAL RESULTS")

    for model_name in models:
        comparison_results[model_name]["cv"] = cv_results[model_name]
        comparison_results[model_name]["optuna"] = optuna_results.get(model_name,{},)

    best_model_name = comparison_results["best_model"]
   
    # 6. SAVE EVERYTHING

    logger.info("\n[6/6] SAVE MODELS AND METRICS")

    models_dir = Path(cfg["artifacts"]["models_dir"])
    models_dir.mkdir(parents=True,exist_ok=True,)
    
    # Save every model

    for model_name, model in trained_models.items():
        model_path = models_dir/ f"{model_name}{cfg['artifacts']['model_suffix']}"
        joblib.dump(model,model_path,)
        
        logger.info("%s saved to %s",model_name,model_path,)
   
    # Save complete comparison
    
    comparison_path = Path(cfg["artifacts"]["comparison_metrics"])
    comparison_path.parent.mkdir(parents=True,exist_ok=True,)

    with open(comparison_path,"w",) as f:
        json.dump(comparison_results,f,indent=2,)
   
    # Save feature names

    feature_names_path = Path(cfg["artifacts"]["feature_names"])
    feature_names_path.parent.mkdir(parents=True,exist_ok=True,)

    with open(feature_names_path,"w",) as f:
        json.dump(feature_names,f,indent=2,)
        
    logger.info("================= FINAL MODEL COMPARISON ======================= ")

    for model_name in models:
        val_rmse = comparison_results[model_name]["validation"]["rmse"]
        test_rmse = comparison_results[model_name]["test"]["rmse"]
        logger.info("%s | Val RMSE: %.4f | Test RMSE: %.4f",model_name,val_rmse,test_rmse,)

    logger.info("BEST MODEL: %s",best_model_name,)

    return {
        "models": trained_models,
        "best_model": trained_models[best_model_name],
        "best_model_name":best_model_name,
        "best_params":best_params,
        "cv_results": cv_results,
        "optuna_results": optuna_results,
        "comparison_results":comparison_results,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names":feature_names,}


if __name__ == "__main__":
    train_model()