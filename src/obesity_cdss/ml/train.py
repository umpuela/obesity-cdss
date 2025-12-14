import json
import time
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from obesity_cdss.config import settings
from obesity_cdss.ml.processing import DataProcessor
from obesity_cdss.utils.logging import setup_logging


class ModelTrainer:
    """
    Orchestrates the end-to-end training pipeline for the Obesity Risk model.

    This class encapsulates the logic for:
    1. Loading and validating raw data.
    2. Splitting data into stratified train/test sets.
    3. Fitting the DataProcessor (feature engineering).
    4. Hyperparameter tuning (GridSearch) across multiple algorithms.
    5. Selecting the best model based on accuracy.
    6. Generating comprehensive metrics (confusion matrix, feature importance).
    7. Serializing artifacts (model, encoder, metrics) for production.

    Attributes:
        processor (DataProcessor): The data preprocessing engine.
        best_model (Any): The winning Scikit-Learn/XGBoost estimator.
        best_metrics (dict[str, Any]): Dictionary containing performance
            metrics of the champion model.
        best_model_name (str): The identifier of the winning algorithm.
    """

    def __init__(self) -> None:
        self.processor: DataProcessor = DataProcessor()
        self.best_model: Any = None
        self.best_metrics: dict[str, Any] = {}
        self.best_model_name: str = ""

    def load_data(self) -> pd.DataFrame:
        """
        Loads and validates the dataset from the configured path.

        Returns:
            pd.DataFrame: The raw dataframe loaded from CSV.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not settings.raw_data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {settings.raw_data_path}")

        logger.info(f"Loading data from {settings.raw_data_path}...")
        return pd.read_csv(settings.raw_data_path)

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataframe into training and testing sets with stratification.

        Args:
            df (pd.DataFrame): The DataFrame to be split.

        Returns:
            tuple: A tuple containing the training and testing sets.
        """
        X = df.drop(columns=[settings.target_column])
        y = df[settings.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y,
        )

        return X_train, X_test, y_train, y_test

    def prepare_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepares the input data for the model.

        Args:
            X_train (pd.DataFrame): Training data.
            X_test (pd.DataFrame): Test data.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.

        Returns:
            tuple: A tuple containing the preprocessed training and testing data.
        """
        logger.info("Processing features and encoding target...")

        X_train_processed = self.processor.fit_transform(X_train, y_train)
        X_test_processed = self.processor.transform(X_test)

        y_train_processed = self.processor.encode_target(y_train)
        y_test_processed = self.processor.encode_target(y_test)

        return X_train_processed, X_test_processed, y_train_processed, y_test_processed

    def grid_search_and_train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Runs GridSearchCV for defined models and selects the best one based on accuracy.
        Updates self.best_model and self.best_model_name with the winner.

        Args:
            X_train (pd.DataFrame): Processed training data.
            y_train (np.ndarray): Processed training labels.
        """
        models_to_train = {
            "LogisticRegression": LogisticRegression(
                random_state=settings.random_state
            ),
            "RandomForest": RandomForestClassifier(random_state=settings.random_state),
            "XGBoost": XGBClassifier(random_state=settings.random_state),
        }

        best_global_score = 0.0

        for name, model in models_to_train.items():
            logger.info(f"Starting GridSearch for {name}...")

            grid_params = settings.model_grids.get(name, {})

            clf = GridSearchCV(
                estimator=model,
                param_grid=grid_params,
                cv=settings.grid_search_cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )

            clf.fit(X_train, y_train)

            best_estimator = clf.best_estimator_
            logger.debug(f"Best params for {name}: {clf.best_params_}")

            if clf.best_score_ > best_global_score:
                best_global_score = clf.best_score_
                self.best_model = best_estimator
                self.best_model_name = name

    def _get_feature_importances(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float]:
        """
        Safely extracts and aggregates feature importances from the model,
        whether it is tree-based or linear.

        Args:
            model (Any): The trained estimator.
            feature_names (list[str]): List of feature names corresponding
                to the model input.

        Returns:
            dict[str, float]: Top 10 most important aggregated features
                and their scores.
        """
        raw_importances = []
        if hasattr(model, "feature_importances_"):
            raw_importances = model.feature_importances_.astype(float)
        elif hasattr(model, "coef_"):
            raw_importances = np.mean(np.abs(model.coef_), axis=0)

        if len(raw_importances) == 0:
            return {}

        aggregated_importances: dict[str, float] = {}

        for name, score in zip(feature_names, raw_importances, strict=True):
            parent_feature = name
            for original in self.processor.categorical_features:
                if name.startswith(f"{original}"):
                    parent_feature = original
                    break
            aggregated_importances[parent_feature] = (
                aggregated_importances.get(parent_feature, 0.0) + score
            )

        return dict(
            sorted(
                aggregated_importances.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        )

    def evaluate_best_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        """
        Generates detailed metrics for the champion model and updates self.best_metrics.
        Includes Classification Report, Confusion Matrix, and Feature Importance.

        Args:
            X_test (pd.DataFrame): Processed test data.
            y_test (np.ndarray): Processed test labels.
        """
        if not self.best_model:
            return

        logger.info("Generating detailed metrics for champion model..")
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        target_names = [str(c) for c in self.processor.target_encoder.classes_]
        report = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)

        feature_names = list(X_test.columns)
        importances = self._get_feature_importances(self.best_model, feature_names)

        auc_score_val = "N/A"
        try:
            auc_score_val = roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="macro"
            )
        except ValueError as e:
            logger.warning(f"Não foi possível calcular o ROC AUC: {e}")

        self.best_metrics.update(
            {
                "model_name": self.best_model_name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "best_params": self.best_model.get_params(),
                "feature_importances": importances,
                "classification_report": report,
                "confusion_matrix": df_cm.to_dict(),
                "roc_auc_macro_ovr": auc_score_val,
            }
        )

    def save_artifacts(self) -> None:
        """Serializes the best model, the fitted processor, and the metrics to disk."""
        if not self.best_model:
            logger.error("No model to save.")
            return

        settings.processor_path.parent.mkdir(parents=True, exist_ok=True)
        self.processor.save_pipeline(settings.processor_path)

        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, settings.model_path)

        with open(settings.metrics_path, "w") as f:
            json.dump(self.best_metrics, f, indent=4, default=str)

        logger.success(f"Artifacts saved successfully at {settings.model_path.parent}.")

    def run(self) -> None:
        """Executes the full training pipeline."""
        logger.info("Starting training pipeline...")
        start_time = time.time()

        try:
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.split_data(df)

            X_train_processed, X_test_processed, y_train_processed, y_test_processed = (
                self.prepare_features(X_train, X_test, y_train, y_test)
            )
            self.grid_search_and_train(X_train_processed, y_train_processed)

            if self.best_model:
                self.evaluate_best_model(X_test_processed, y_test_processed)

                elapsed_time = time.time() - start_time
                logger.success(
                    f"Winner: {self.best_model_name} | "
                    f"Acc: {self.best_metrics['classification_report']['accuracy']:.4f}"
                    f" | Time: {elapsed_time:.2f}s"
                )
                self.save_artifacts()
            else:
                logger.error("Failed to select a model.")

        except Exception as e:
            logger.exception("Critical error during training pipeline.")
            raise e


if __name__ == "__main__":
    setup_logging(log_dir=settings.log_dir_ml)
    trainer = ModelTrainer()
    trainer.run()
