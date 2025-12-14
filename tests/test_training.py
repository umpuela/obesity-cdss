from pathlib import Path

import pandas as pd
import pytest

from obesity_cdss.config import settings
from obesity_cdss.ml.train import ModelTrainer


def test_training_pipeline_smoke(tmp_path: Path, sample_data: pd.DataFrame) -> None:
    fake_csv = tmp_path / "fake_data.csv"
    sample_data.to_csv(fake_csv, index=False)

    fake_model_path = tmp_path / "model.joblib"
    fake_processor_path = tmp_path / "data_processor.joblib"
    fake_metrics_path = tmp_path / "metrics.json"

    settings.raw_data_path = fake_csv
    settings.model_path = fake_model_path
    settings.processor_path = fake_processor_path
    settings.metrics_path = fake_metrics_path
    settings.grid_search_cv_folds = 2
    settings.model_grids = {
        "LogisticRegression": {"C": [1.0]},
        "RandomForest": {"n_estimators": [2], "max_depth": [2]},
        "XGBoost": {"n_estimators": [2], "max_depth": [2]},
    }

    trainer = ModelTrainer()

    try:
        trainer.run()
    except Exception as e:
        pytest.fail(f"Training pipeline failed: {e}")

    assert fake_model_path.exists(), "The model file was not generated."
    assert fake_processor_path.exists(), "The encoder file was not generated."
    assert fake_metrics_path.exists(), "The metrics file was not generated."

    assert trainer.best_model is not None
