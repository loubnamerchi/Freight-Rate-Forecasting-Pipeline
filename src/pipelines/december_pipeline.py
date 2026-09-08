
from __future__ import annotations

import time

from src.data.data_ingestion     import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.build_features  import FeatureEngineer
from src.utils.logger             import get_logger
from src.pipelines.data_pipeline  import run_pipeline
from src.pipelines.training_pipeline  import train_model

logger = get_logger(__name__)


def december_pipeline(config_path: str = "dec_config.yaml") -> dict:
  
    logger.info("=============== DECEMBER PIPELINE ===============")

    # 1. DATA PIPELINE
    logger.info("Running December data pipeline...")
    data_result = run_pipeline(config_path)

    # 2. TRAINING PIPELINE
    logger.info("Running December training pipeline...")
    training_result = train_model(config_path)

    logger.info("December pipeline completed")

    return {"data": data_result,"training": training_result,}

if __name__ == "__main__":
    december_pipeline()