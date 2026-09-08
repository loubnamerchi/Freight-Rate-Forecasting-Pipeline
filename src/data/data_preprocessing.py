

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pathlib
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger 

from src.utils.config import load_config


logger = get_logger(__name__)


class DataPreprocessor:

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg      = load_config(config_path)
        self.target   = self.cfg["data"]["target_column"]
        
        self.cleaning_report: Dict = {}


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
      
        logger.info("=== Data Preprocessing START ===")
        original_shape = df.shape
        df = df.copy()

        df = self._fix_dtypes(df)
        #df = self._remove_invalid_values(df)  i commented it because in  validation.csv it should be 12000 no removed rows i cant consider negaative weights positive, i dont have much informations about the dataset 
        df = self._handle_missing(df, fit=True)

        self.cleaning_report = {
            "original_shape": original_shape,
            "clean_shape":    df.shape,
            "rows_removed":   original_shape[0] - df.shape[0],
        }
        logger.info(
            f"Preprocessing complete. "
            f"{original_shape[0]:,} → {df.shape[0]:,} rows "
            f"({self.cleaning_report['rows_removed']:,} removed)"
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df = self._fix_dtypes(df)
        #df = self._remove_invalid_values(df)
        df = self._handle_missing(df, fit=False)
        return df


    ## -------------------- Fix dtypes ------------------------
    
    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df["date"] = pd.to_datetime(df["date"])
        
        logger.info("date Dtype fixes applied.")
        
        return df

    ## ---------------- Remove invalid values ---------------------
    
    def _remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
            
        invalid_weight = (
            (df["weight"] <= 0) |
            (~np.isfinite(df["weight"]))
            )
        
        df = df.loc[~invalid_weight].copy()
                       
        logger.info(f"Removed {invalid_weight.sum():,} rows with invalid weight values.")
         
        return df
    
    
    ## ---------------- Handle missing values -------------------
    
    """
    def _handle_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        
        missing_cols = ["weight", "market_index"]
        
        if fit:
            self._imputer = SimpleImputer(strategy="median")
            df[missing_cols]  = self._imputer.fit_transform(df[missing_cols])
                        
        else:
            if self._imputer is None:
                raise RuntimeError("Imputer not fitted — call fit_transform first.")
            df[missing_cols] = self._imputer.transform(df[missing_cols])
        
        logger.info(f"Imputed missing values in columns: {missing_cols} "f"using median strategy.")

        return df
    """
    
    def _handle_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        null_counts = df.isnull().sum()
        total_missing = null_counts.sum()
    
        if total_missing == 0:
            logger.info("No missing values — skipping imputation.")
            return df
    
        logger.info("Handling %s missing values ""(numerical: median, categorical: mode)",f"{total_missing:,}" )
    
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    
        # Numerical columns → Median
        if num_cols:
            if fit:
                self._imputer = SimpleImputer(strategy="median")
                df[num_cols] = self._imputer.fit_transform(df[num_cols])
            else:
                if self._imputer is None:
                    raise RuntimeError("Imputer not fitted — call fit_transform first.")
    
                df[num_cols] = self._imputer.transform(df[num_cols])
    
        # Categorical columns → Mode
        if cat_cols:
            if fit:
                self._cat_modes = {}
    
                for col in cat_cols:
                    mode = df[col].mode()
    
                    if not mode.empty:
                        self._cat_modes[col] = mode.iloc[0]
                    else:
                        self._cat_modes[col] = "UNKNOWN"
    
                    df[col] = df[col].fillna(self._cat_modes[col])
    
            else:
                if not hasattr(self, "_cat_modes"):
                    raise RuntimeError("Categorical imputation values not fitted.")
    
                for col in cat_cols:
                    if col in self._cat_modes:
                        df[col] = df[col].fillna(self._cat_modes[col])
    
        return df
    
    
    ## ------------------- Persistence -------------------
        
    def save_preprocessor(self,path: str = "artifacts/preprocessing/preprocessor.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"DataPreprocessor saved → {path}")
            
    
        
    
