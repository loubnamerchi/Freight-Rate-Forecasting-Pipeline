from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.model_factory import create_model
from src.utils.logger import get_logger


logger = get_logger(__name__)

def evaluate_model(model,X,y,):

    predictions = model.predict(X)
    rmse = mean_squared_error(y,predictions,) ** 0.5
    mae = mean_absolute_error(y,predictions,)
    r2 = r2_score(y,predictions,)

    return {"rmse": float(rmse),"mae": float(mae),"r2": float(r2),}


def train_and_compare_models(models,X_train,y_train, X_val,y_val,X_test,y_test,best_params,cfg,):

    random_state = cfg["training"].get("random_state",42,)
    results = {}
    trained_models = {}

    for model_name in models:
        logger.info("\nTraining final %s model...", model_name)
        model = create_model(model_name=model_name,
                             params=best_params[model_name],
                             random_state=random_state,)
        
        # Final training

        if model_name == "xgboost":
            model.fit(X_train,y_train,eval_set=[(X_train, y_train),(X_val, y_val),],verbose=False,)

        elif model_name == "lightgbm":
            model.fit(X_train,y_train,eval_set=[(X_train, y_train),(X_val, y_val),],callbacks=[],)

        else:
            model.fit(X_train,y_train,)
       
        # Validation

        val_metrics = evaluate_model(model,X_val,y_val,)
        
        # Test

        test_metrics = evaluate_model(model,X_test,y_test,)
        results[model_name] = {"validation": val_metrics,
                               "test": test_metrics,
                               "best_params": best_params[model_name],}

        trained_models[model_name] = model

        logger.info("%s | Val RMSE: %.4f | Test RMSE: %.4f",model_name,val_metrics["rmse"],test_metrics["rmse"],)
 
    # SAVE COMPARISON

    comparison_path = Path(cfg["artifacts"]["comparison_metrics"])
    comparison_path.parent.mkdir(parents=True,exist_ok=True,)
    with open(comparison_path,"w",) as f:
        json.dump(results,f,indent=2,)

    # BEST MODEL

    best_model_name = min(results, key=lambda name:results[name]["validation"]["rmse"],)
    results["best_model"] = best_model_name
    
    logger.info("\n========================================")
    logger.info("BEST MODEL: %s", best_model_name)
    logger.info("Validation RMSE: %.4f",results[best_model_name]["validation"]["rmse"],)
    logger.info("Test RMSE: %.4f",results[best_model_name]["test"]["rmse"],)
    logger.info("========================================")

    return (trained_models,results,)