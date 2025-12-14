import json
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from obesity_cdss.api.schemas import PatientData, PredictionResponse
from obesity_cdss.config import settings
from obesity_cdss.ml.processing import DataProcessor


class ModelService:
    """
    Singleton service for Model Management and Inference.

    Responsible for:
    1. Loading and caching ML artifacts (Model and Encoder) in memory.
    2. Performing predictions on new patient data.
    3. Retrieving model metadata (metrics, dictionary).

    Attributes:
        model (Any, optional): The trained Scikit-Learn/XGBoost estimator.
        processor (DataProcessor, optional): The fitted data processor.
        is_loaded (bool): Flag indicating if artifacts are currently in memory.
    """

    _instance: "ModelService | None" = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None
            cls._instance.is_loaded = False
        return cls._instance

    def load_artifacts(self) -> None:
        """
        Loads the serialized model and encoder from disk into memory.

        Raises:
            FileNotFoundError: If artifact files (.pkl/.joblib) are missing.
            RuntimeError: If loading fails due to deserialization errors.
        """
        if self.is_loaded:
            return

        logger.info("Loading model artifacts into memory...")

        try:
            self.processor = DataProcessor.load_pipeline(settings.processor_path)

            if not settings.model_path.exists():
                raise FileNotFoundError(f"Model file missing at {settings.model_path}")
            self.model = joblib.load(settings.model_path)

            self.is_loaded = True
            logger.success("Model artifacts loaded successfully.")

        except Exception as e:
            logger.critical(f"Failed to load artifacts: {e}")
            raise RuntimeError("Could not initialize model service.") from e

    def get_metrics(self) -> dict[str, Any]:
        """
        Retrieves the model performance metrics from the report file.

        Returns:
            dict[str, Any]: JSON content of the metrics report.

        Raises:
            FileNotFoundError: If the metrics file does not exist.
        """
        file_path = settings.metrics_path
        if not file_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Corrupted metrics file: {e}")
            raise RuntimeError("Failed to read metrics file.") from e

    def get_data_dictionary(self) -> dict[str, Any]:
        """
        Retrieves the Data Dictionary (Schema) for frontend rendering.

        Returns:
            dict[str, Any]: The data dictionary.
        """
        file_path = settings.data_dictionary_path
        if not file_path.exists():
            raise FileNotFoundError(f"Data dictionary file not found at {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse data dictionary: {e}")
            raise RuntimeError("Invalid data dictionary format.") from e

    def predict(self, input_data: PatientData) -> PredictionResponse:
        """
        Runs the inference pipeline for a single patient.

        Args:
            input_data (PatientData): Validated input data from the API request.

        Returns:
            PredictionResponse: The predicted class and confidence score.
        """
        if not self.is_loaded:
            self.load_artifacts()

        try:
            input_dict = input_data.model_dump()
            df = pd.DataFrame([input_dict])

            X_processed = self.processor.transform(df)
            prediction_idx = self.model.predict(X_processed)[0]
            prediction_label = str(self.processor.decode_target(prediction_idx))

            probabilities = self.model.predict_proba(X_processed)[0]
            confidence = float(probabilities[prediction_idx])

            return PredictionResponse(
                label=prediction_label,
                probability=round(confidence, 4),
                model_version=settings.version,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e


class InsightsService:
    """
    Service responsible for Exploratory Data Analysis (EDA) logic.

    It handles loading raw historical data, performing aggregations, and
    returning anonymized statistics for the frontend dashboard.

    This service ensures Privacy by Design by not exposing PII (Personally
    Identifiable Information) to the outer layers.
    """

    def get_distributions(self) -> dict[str, Any]:
        """
        Aggregates data for distribution charts and KPIs.

        Returns:
            dict[str, Any]: A dictionary containing lists of records for visualization.

        Raises:
            FileNotFoundError: If the raw dataset is not found on the server.
        """
        if not settings.raw_data_path.exists():
            logger.error(f"Dataset not found at {settings.raw_data_path}")
            raise FileNotFoundError("Dataset file missing.")

        cols = [
            "Obesity",
            "Gender",
            "Age",
            "Height",
            "Weight",
            "family_history",
            "MTRANS",
            "FCVC",
            "TUE",
            "FAF",
            "FAVC",
            "SMOKE",
            "NCP",
        ]

        try:
            df = pd.read_csv(settings.raw_data_path, usecols=cols)
            processor = DataProcessor()
            df = processor._clean_dataframe(df)

            total_pacients = len(df)
            median_ncp = float(df["ncp"].median())

            obesity_count = df["obesity"].str.contains("obesity", case=False).sum()
            obesity_rate = (
                float(obesity_count / total_pacients) if total_pacients > 0 else 0.0
            )

            dist_obesity = (
                df["obesity"]
                .value_counts()
                .reset_index()
                .rename(
                    columns={
                        "index": "category",
                        "obesity": "category",
                    }
                )
                .to_dict(orient="records")
            )

            fam_hist_data = (
                df.groupby(["family_history", "obesity"])
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            )

            smoke_data = (
                df["smoke"]
                .value_counts(normalize=True)
                .reset_index()
                .rename(
                    columns={
                        "index": "category",
                        "smoke": "category",
                    }
                )
                .to_dict(orient="records")
            )

            sample_df = df.sample(n=min(2000, len(df)), random_state=42)
            scatter_data = sample_df[["age", "weight", "mtrans", "height"]].to_dict(
                orient="records"
            )
            boxplot_data = sample_df[["fcvc", "weight", "obesity"]].to_dict(
                orient="records"
            )
            tech_data = sample_df[["tue", "age", "obesity"]].to_dict(orient="records")
            lifestyle_data = sample_df[["faf", "favc", "weight"]].to_dict(
                orient="records"
            )
            pyramid_data = sample_df[["gender", "age"]].to_dict(orient="records")

            return {
                "kpis": {
                    "total_patients": total_pacients,
                    "median_ncp": median_ncp,
                    "obesity_rate": obesity_rate,
                },
                "charts": {
                    "distribution_obesity": dist_obesity,
                    "family_history": fam_hist_data,
                    "smoke_data": smoke_data,
                    "scatter_data": scatter_data,
                    "boxplot_data": boxplot_data,
                    "tech_data": tech_data,
                    "lifestyle_data": lifestyle_data,
                    "pyramid_data": pyramid_data,
                },
            }

        except Exception as e:
            logger.error(f"Failed to process insights data: {e}")
            raise RuntimeError("Could not process dataset.") from e
