from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def create_model(model_name: str,params: dict[str, Any],random_state: int = 42,):

    model_name = model_name.lower()

    if model_name == "xgboost":

        return XGBRegressor(**params,objective="reg:squarederror",
                            eval_metric="rmse",random_state=random_state,
                            n_jobs=-1,)

    elif model_name == "lightgbm":

        return LGBMRegressor(**params,objective="regression",metric="rmse",
                             random_state=random_state,n_jobs=-1,verbosity=-1,)

    elif model_name == "random_forest":

        return RandomForestRegressor(**params,random_state=random_state,n_jobs=-1,)

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Supported models: xgboost, lightgbm, random_forest.")

def get_default_params(model_name: str,cfg: dict,) -> dict[str, Any]:

    model_name = model_name.lower()

    if model_name not in cfg["training"]:
        raise ValueError(f"No training configuration found for {model_name}")

    return cfg["training"][model_name].copy()