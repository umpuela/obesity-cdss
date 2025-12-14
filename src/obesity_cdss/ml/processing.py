from pathlib import Path
from typing import Self, cast

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


class DataProcessor:
    """
    Handles data preprocessing for features (X) and target (y) in the pipeline.

    This class encapsulates the logic for:
    1. Cleaning data.
    2. Imputing missing values.
    3. Encoding categorical variables (OneHot and Ordinal).
    4. Scaling numerical variables.
    5. Encoding/Decoding the target variable.

    Attributes:
        discrete_features (List[str]): Discrete numerical features.
        continuous_features (List[str]): Continuous numerical features.
        numeric_features (List[str]): All numerical features.
        cat_nominal_features (List[str]): Categorical features without order.
        cat_ordinal_features (List[str]): Categorical features with intrinsic order.
        ordinal_categories (List[List[str]]): Categories for ordinal encoding.
        pipeline (ColumnTransformer): The main Scikit-Learn transformation pipeline.
        target_encoder (LabelEncoder): The encoder for the target variable.
        is_fitted (bool): Flag indicating if the processor has been fitted.
    """

    def __init__(self) -> None:
        self.discrete_features: list[str] = [
            "fcvc",
            "ncp",
            "ch2o",
            "faf",
            "tue",
        ]
        self.continuous_features: list[str] = ["age"]
        self.numeric_features: list[str] = (
            self.discrete_features + self.continuous_features
        )

        self.cat_nominal_features: list[str] = [
            "gender",
            "family_history",
            "favc",
            "smoke",
            "scc",
            "mtrans",
        ]
        self.cat_ordinal_features: list[str] = ["caec", "calc"]
        self.categorical_features: list[str] = (
            self.cat_nominal_features + self.cat_ordinal_features
        )

        self.ordinal_categories: list[list[str]] = [
            ["no", "sometimes", "frequently", "always"],
            ["no", "sometimes", "frequently", "always"],
        ]

        self.pipeline: ColumnTransformer = self._build_feature_pipeline()
        self.target_encoder: LabelEncoder = LabelEncoder()
        self.is_fitted: bool = False

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame. Renames columns and its values to lowercase,
        rounds up and converts discrete features to integers.

        Args:
            df (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned and standardized DataFrame.
        """
        df_clean = df.copy()

        df_clean.columns = df_clean.columns.str.lower().str.strip()

        for col in self.categorical_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower()

        for col in self.discrete_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].round().astype(int)

        logger.success("Data cleaning complete.")
        return df_clean

    def _build_feature_pipeline(self) -> ColumnTransformer:
        """
        Constructs the Scikit-Learn transformation pipeline.

        Builds a ColumnTransformer that applies:
        - SimpleImputer and StandardScaler to numerical features.
        - SimpleImputer and OrdinalEncoder to ordinal features.
        - SimpleImputer and OneHotEncoder to nominal features.

        Returns:
            ColumnTransformer: The configured preprocessing pipeline ready to be fitted.
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        numeric_transformer.set_output(transform="pandas")

        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=self.ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        ordinal_transformer.set_output(transform="pandas")

        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        nominal_transformer.set_output(transform="pandas")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    self.numeric_features,
                ),
                ("ord", ordinal_transformer, self.cat_ordinal_features),
                ("nom", nominal_transformer, self.cat_nominal_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        preprocessor.set_output(transform="pandas")

        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the feature pipeline and optionally the target encoder.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series, optional): The target variable. If provided,
                the target encoder will also be fitted.

        Returns:
            DataProcessor: The instance itself (self), allowing method chaining.
        """
        logger.info("Fitting DataProcessor...")

        X_clean = self._clean_dataframe(X)
        self.pipeline.fit(X_clean)

        if y is not None:
            logger.info("Fitting target encoder...")
            self.target_encoder.fit(y.astype(str))

        self.is_fitted = True
        logger.info("DataProcessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted pipeline to transform input features.

        Args:
            X (pd.DataFrame): The input features to be transformed.

        Returns:
            pd.DataFrame: The processed data with scaled numbers and encoded categories.

        Raises:
            NotFittedError: If the processor has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError("DataProcessor must be fitted before transform.")

        X_clean = self._clean_dataframe(X)
        logger.debug(f"Transforming data shape: {X_clean.shape}")
        X_transformed = self.pipeline.transform(X_clean)
        return cast(pd.DataFrame, X_transformed)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Fits the pipeline to the data and returns the transformed version.

        This is a convenience method that calls `fit` followed by `transform`.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            pd.DataFrame: The processed features.
        """
        self.fit(X, y)
        return self.transform(X)

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Converts string class labels into integer codes.

        Args:
            y (pd.Series): The target variable containing string labels
                (e.g., 'Normal_Weight').

        Returns:
            np.ndarray: An array of integers representing the encoded classes.

        Raises:
            NotFittedError: If the target encoder has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError("DataProcessor must be fitted before transform.")

        y_encoded = self.target_encoder.transform(y.astype(str))
        return cast(np.ndarray, y_encoded)

    def decode_target(self, y_encoded: int | np.ndarray) -> str | np.ndarray:
        """
        Converts integer class codes back into string labels.

        Args:
            y_encoded (int | np.ndarray): An integer or array of integers
                representing the predicted classes.

        Returns:
            str | np.ndarray: The original string label(s) corresponding
                to the input code(s).

        Raises:
            NotFittedError: If the target encoder has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError("Processor not fitted.")

        if isinstance(y_encoded, (int, np.integer)):
            return self.target_encoder.inverse_transform([y_encoded])[0]

        return self.target_encoder.inverse_transform(y_encoded)

    def save_pipeline(self, filepath: Path) -> None:
        """
        Serializes the entire DataProcessor instance to disk.

        Args:
            filepath (str | Path): The destination path for the .joblib file.

        Raises:
            NotFittedError: If the processor has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError("Cannot save an unfitted pipeline.")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load_pipeline(filepath: Path) -> "DataProcessor":
        """
        Deserializes a DataProcessor instance from disk.

        Args:
            filepath (Path): The path to the saved processor.

        Returns:
            DataProcessor: The loaded processor.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found at {filepath}")

        processor = joblib.load(filepath)
        logger.info(f"Processor loaded from {filepath}")
        return processor
