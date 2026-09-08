
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from pathlib import Path 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,    
    StandardScaler,

)

from src.utils.logger import get_logger

from src.utils.config import load_config


logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg      = load_config(config_path)
        self.target   = self.cfg["data"]["target_column"]
        self.paths_cfg = self.cfg["paths"]

        self._scaler: Optional[object]           = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scale_cols: List[str]              = []
        self._engineered_cols: List[str]         = []
        self.feature_names_out_: List[str]       = []



    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        
        logger.info("=== Feature Engineering FIT ===")
        X = X.copy()

        # 1. Build new features first (no fitting needed)
        X = self._time_features(X)
        X = self._distance_features(X)
        #X = self._weight_features(X)    here i cant use it because i didnt remove negative weights 
        
        #if self.fe_cfg.get("interaction_features", False):       # ill use it for linear models i didnt decide yet whcih models ill use
            #X = self._interaction_features(X)

        # 2. Encode categoricals
        X = self._encode_categoricals(X, fit=True)

        # 3. Fit scaler on designated columns
        ## am not gonna use it for XGBoost because tree-based models are insensitive to feature scale 
        #self._fit_scaler(X)

        self.feature_names_out_ = X.columns.tolist()
        logger.info(f"Feature engineering fitted. Output features: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        
        logger.info("=== Feature Engineering TRANSFORM ===")
        X = X.copy()

        X = self._time_features(X)
        X = self._distance_features(X)
        #X = self._weight_features(X)
        
        #if self.fe_cfg.get("interaction_features", False):            # for Linear models
            #X = self._interaction_features(X)
            
        X = self._encode_categoricals(X, fit=False)
        
        #X = self._apply_scaler(X)

        # Reorder to match fitted order
        for col in self.feature_names_out_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[[c for c in self.feature_names_out_ if c in X.columns]]
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
    

    ##----------- Feature builders --------------------
    

    def _time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        
        if "date" not in X.columns:
            return X
    
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["day_of_week"] = X["date"].dt.dayofweek
        X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
        
        new_cols = ["year",
                "month",
                "day_of_week",
                "is_weekend"
            ]
    
        self._engineered_cols.extend(new_cols)
    
        # Drop raw date after extracting useful information
        X = X.drop(columns=["date"])
    
        logger.info(f"Time features created: {new_cols}")
    
        return X
    
    def _distance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        
        if "distance" not in X.columns:
            return X
    
        # Log transformation reduces the effect of very large distances
        X["distance_log"] = np.log1p(X["distance"])
    
        new_cols = ["distance_log"]
    
        self._engineered_cols.extend(new_cols)
    
        # Drop raw distance
        X = X.drop(columns=["distance"])
    
        logger.info(f"Distance features created: {new_cols}")
    
        return X
    
    def _weight_features(self, X: pd.DataFrame) -> pd.DataFrame:
        
        if "weight" not in X.columns:
            return X
    
        # Log transformation reduces the effect of very large weights
        X["weight_log"] = np.log1p(X["weight"])
    
        new_cols = ["weight_log"]
    
        self._engineered_cols.extend(new_cols)
    
        # Drop raw weight
        X = X.drop(columns=["weight"])
    
        logger.info(f"Weight features created: {new_cols}")
    
        return X
    
        
    def _interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fe_cfg.get("interaction_features", True):
            return X
    
        new_cols = []
    
        # Distance × Weight
        if "Distance" in X.columns and "Weight" in X.columns:
            X["distance_weight_interaction"] = (
                X["Distance"] * X["Weight"]
            )
            new_cols.append("distance_weight_interaction")
    
        # Distance × Market Index
        if "Distance" in X.columns and "market_index" in X.columns:
            X["distance_market_interaction"] = (
                X["Distance"] * X["market_index"]
            )
            new_cols.append("distance_market_interaction")
    
        self._engineered_cols.extend(new_cols)
    
        logger.info(f"Interaction features created: {new_cols}")
    
        return X
    
    ## -------------------- Encoding ---------------------
    

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return X

        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique == 2:
                # Binary → label encode
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self._label_encoders[col] = le
                else:
                    le = self._label_encoders.get(col)
                    if le:
                        X[col] = le.transform(X[col].astype(str))
            elif n_unique <= 15:
                # Low cardinality → one-hot encode
                if fit:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    self._label_encoders[f"_ohe_{col}"] = dummies.columns.tolist()
                else:
                    expected_cols = self._label_encoders.get(f"_ohe_{col}", [])
                    dummies       = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    for ec in expected_cols:
                        if ec not in dummies.columns:
                            dummies[ec] = 0
                    X = pd.concat([X.drop(columns=[col]), dummies[expected_cols]], axis=1)
            else:
                # High cardinality → frequency encoding
                if fit:
                    freq = X[col].value_counts(normalize=True).to_dict()
                    self._label_encoders[f"_freq_{col}"] = freq
                else:
                    freq = self._label_encoders.get(f"_freq_{col}", {})
                X[col] = X[col].map(freq if fit else self._label_encoders.get(f"_freq_{col}", {})).fillna(0)

        logger.info(f"Categorical encoding applied to {len(cat_cols)} column(s).")
        return X


    ## ------------------------ Scaling ------------------------------------

    def _fit_scaler(self, X: pd.DataFrame) -> None:  # for train X 
        scale_cols = [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
        ]
    
        # Keep only columns that actually exist in X
        scale_cols = [c for c in scale_cols if c in X.columns]
    
        self._scale_cols = scale_cols
    
        if not scale_cols:
            return
    
        self._scaler = StandardScaler()
        self._scaler.fit(X[scale_cols])
    
        logger.info(f"StandardScaler fitted on: {scale_cols}")

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:  # for val/test X 
        if self._scaler is not None and self._scale_cols:
            cols_present = [c for c in self._scale_cols if c in X.columns]
    
            X[cols_present] = self._scaler.transform(X[cols_present])
    
        return X
    

    ## ------------------ Persistence -----------------------

    def save_scaler(self, path: str = "data/processed/scaler.joblib") -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        logger.info(f"Scaler saved → {path}")

    def save_feature_names(self, path: str = "artifacts/feature_engineering/feature_names.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.feature_names_out_, f, indent=2)
        logger.info(f"Feature names saved → {path}")
        
    def save_feature_engineer(self, path: str = "artifacts/feature_engineering/feature_engineer.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved → {path}") 
    
    def save_parquets(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> None:
        
        datasets = {
            self.paths_cfg["processed_train"]: X_train,
            self.paths_cfg["processed_val"]: X_val,
            self.paths_cfg["processed_test"]: X_test,
            self.paths_cfg["processed_y_train"]: y_train,
            self.paths_cfg["processed_y_val"]: y_val,
            self.paths_cfg["processed_y_test"]: y_test,
        }
        
        for file_path, obj in datasets.items():
            path = Path(file_path)

            # Create parent directory if it does not exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Feature DataFrames
            if isinstance(obj, pd.DataFrame):
                obj.to_parquet(path, index=False)
                logger.info(
                    f"Saved {path.name} | shape={obj.shape}"
                )

            # Numpy arrays returned by fit_resample
            elif isinstance(obj, np.ndarray):
                df = pd.DataFrame(
                    obj,
                    columns=self.selected_features_
                )
                
                df.to_parquet(path, index=False)
                logger.info(
                    f"Saved {path.name} | shape={df.shape}"
                )
                
            # Target Series
            else:
                pd.Series(obj, name=self.target).to_frame().to_parquet(
                    path,
                    index=False,
                )
                logger.info(
                    f"Saved {path.name} | shape={(len(obj), 1)}"
                )
                
                
            
        

        
                 
                
            
            
            
                 
    
    
    
    

    

    
        

    
       
            