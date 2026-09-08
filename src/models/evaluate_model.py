from __future__ import annotations

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def _fit_fold_model(model,X_train,y_train,X_val,y_val,):

    if isinstance(model, XGBRegressor):

        model.fit(X_train,y_train,eval_set=[(X_val, y_val)],verbose=False,)

    elif isinstance(model, LGBMRegressor):

        model.fit(X_train,y_train,eval_set=[(X_val, y_val)],callbacks=[],)

    else:

        # Random Forest do not support eval_set.
        model.fit(X_train,y_train,)

    return model


def cross_validate_model(model,X,y,n_splits=5,):
    """
    Time-series cross-validation.

    The data must already be sorted chronologically.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X),start=1,):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Fresh model for each fold
        fold_model = clone(model)
        fold_model = _fit_fold_model(fold_model,X_tr,y_tr,X_val,y_val,)
        predictions = fold_model.predict(X_val)
        rmse = mean_squared_error(y_val,predictions,) ** 0.5
        mae = mean_absolute_error(y_val,predictions,)
        r2 = r2_score(y_val,predictions,)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return {
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores)),
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores)),
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
    }