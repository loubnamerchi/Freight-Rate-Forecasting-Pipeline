from __future__ import annotations

from typing import Any

import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.models.model_factory import create_model


def run_optuna(model_name: str,X_train,y_train,cfg: dict,):

    model_name = model_name.lower()
    n_splits = cfg["cross_validation"].get("n_splits",5,)
    n_trials = cfg["optuna"].get("n_trials",50,)
    random_state = cfg["training"].get("random_state",42,)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    optuna_cfg = cfg["optuna"][model_name]

    def objective(trial):

        # XGBOOST
        
        if model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_cfg["learning_rate_min"],
                    optuna_cfg["learning_rate_max"],
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    optuna_cfg["subsample_min"],
                    optuna_cfg["subsample_max"],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    optuna_cfg["colsample_bytree_min"],
                    optuna_cfg["colsample_bytree_max"],
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    optuna_cfg["min_child_weight_min"],
                    optuna_cfg["min_child_weight_max"],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_cfg["reg_alpha_min"],
                    optuna_cfg["reg_alpha_max"],
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_cfg["reg_lambda_min"],
                    optuna_cfg["reg_lambda_max"],
                    log=True,
                ),
            }
 
        # LIGHTGBM

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    optuna_cfg["num_leaves_min"],
                    optuna_cfg["num_leaves_max"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_cfg["learning_rate_min"],
                    optuna_cfg["learning_rate_max"],
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    optuna_cfg["subsample_min"],
                    optuna_cfg["subsample_max"],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    optuna_cfg["colsample_bytree_min"],
                    optuna_cfg["colsample_bytree_max"],
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    optuna_cfg["min_child_samples_min"],
                    optuna_cfg["min_child_samples_max"],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_cfg["reg_alpha_min"],
                    optuna_cfg["reg_alpha_max"],
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_cfg["reg_lambda_min"],
                    optuna_cfg["reg_lambda_max"],
                    log=True,
                ),
            }

        # RANDOM FOREST

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    optuna_cfg["min_samples_split_min"],
                    optuna_cfg["min_samples_split_max"],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    optuna_cfg["min_samples_leaf_min"],
                    optuna_cfg["min_samples_leaf_max"],
                ),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    optuna_cfg["max_features_options"],
                ),
            }

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # TIME-SERIES CV

        fold_rmse = []
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_train)):
            X_tr = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[valid_idx]

            model = create_model(model_name=model_name,params=params,random_state=random_state,)

            # XGBoost
            
            if model_name == "xgboost":
                pruning_callback = (XGBoostPruningCallback(trial,"validation_0-rmse",))
                model.set_params(callbacks=[pruning_callback])
                model.fit(X_tr,y_tr,eval_set=[(X_fold_val, y_fold_val)],verbose=False,)

            # LightGBM

            elif model_name == "lightgbm":
                pruning_callback = (LightGBMPruningCallback(trial,"rmse",))
                model.fit(X_tr,y_tr,eval_set=[(X_fold_val, y_fold_val)],callbacks=[pruning_callback],)

            # Random Forest
            
            else:
                model.fit(X_tr,y_tr,)

            predictions = model.predict(X_fold_val)
            rmse = mean_squared_error(y_fold_val,predictions,) ** 0.5
            fold_rmse.append(rmse)
            intermediate_rmse = sum(fold_rmse) / len(fold_rmse)
            
            if model_name in ["xgboost", "lightgbm"]:
                trial.report(intermediate_rmse,step=fold,)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return sum(fold_rmse) / len(fold_rmse)

    # OPTUNA STUDY
    
    if model_name in ["xgboost", "lightgbm"]:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}_regression",
            sampler=optuna.samplers.TPESampler(
                seed=random_state
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
            ),
        )  
    else:  
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}_regression",
            sampler=optuna.samplers.TPESampler(
                seed=random_state
            ),
        )
        
    study.optimize(objective,n_trials=n_trials,show_progress_bar=True,)

    return (study.best_params,study.best_value,study,)