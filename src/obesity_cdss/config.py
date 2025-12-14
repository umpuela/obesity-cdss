from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings.

    Variables are loaded in the following priority order:
    1. System environment variables (OS)
    2. .env file (at project root)
    3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    project_name: str = "Obesity Risk CDSS"
    version: str = "0.1.0"
    log_level: str = "DEBUG"

    # Machine Learning
    target_column: str = "Obesity"
    test_size: float = 0.2
    random_state: int = 239667
    grid_search_cv_folds: int = 5
    model_grids: dict[str, dict[str, Any]] = {
        "LogisticRegression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
            "num_class": [7],
            "objective": ["multi:softmax"],
            "eval_metric": ["mlogloss"],
        },
    }

    # API
    api_base_url: str = "http://127.0.0.1:8000"

    # Paths
    proj_root: Path = Path(__file__).resolve().parents[2]
    data_file_name: str = "Obesity.csv"
    model_file_name: str = "obesity_model.joblib"
    processor_file_name: str = "data_processor.joblib"
    metrics_file_name: str = "evaluation_metrics.json"
    data_dictionary_file_name: str = "data_dictionary.json"

    data_dir: Path = proj_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    models_dir: Path = proj_root / "models"
    reports_dir: Path = proj_root / "reports"
    utils_dir: Path = proj_root / "src/obesity_cdss/utils"
    log_dir: Path = reports_dir / "logs"

    raw_data_path: Path = raw_data_dir / data_file_name
    model_path: Path = models_dir / model_file_name
    processor_path: Path = models_dir / processor_file_name
    metrics_path: Path = reports_dir / metrics_file_name
    data_dictionary_path: Path = utils_dir / data_dictionary_file_name
    log_dir_ml: Path = log_dir / "ml"
    log_dir_api: Path = log_dir / "api"


settings = Settings()
