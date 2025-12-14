import numpy as np
import pandas as pd

from obesity_cdss.ml.processing import DataProcessor


def test_processor_fit_transform(sample_data: pd.DataFrame) -> None:
    processor = DataProcessor()

    processed = processor.fit_transform(sample_data)

    assert isinstance(processed, pd.DataFrame)

    assert processed.isna().sum().sum() == 0

    assert "gender" not in processed.columns

    assert processed.shape[1] >= sample_data.shape[1]


def test_target_encoding(sample_data: pd.DataFrame) -> None:
    processor = DataProcessor()
    y_raw = pd.Series(["Normal_Weight", "Obesity_Type_I"])

    processor.fit(sample_data, y_raw)

    y_enc = processor.encode_target(y_raw)
    assert isinstance(y_enc, np.ndarray)
    assert len(y_enc) == 2

    y_dec = processor.decode_target(y_enc)
    assert y_dec[0] == "Normal_Weight"
