from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obesity_cdss.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Returns a test client for the API (simulates HTTP requests)."""
    return TestClient(app)


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Returns a valid JSON payload to send to the API."""
    return {
        "gender": "male",
        "age": 25,
        "height": 1.75,
        "weight": 80.0,
        "family_history": "yes",
        "favc": "yes",
        "fcvc": 2.0,
        "ncp": 3.0,
        "caec": "sometimes",
        "smoke": "no",
        "ch2o": 2.0,
        "scc": "no",
        "faf": 1.0,
        "tue": 0.0,
        "calc": "sometimes",
        "mtrans": "public_transportation",
    }


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Returns a DataFrame for unit tests. Mocks raw data from CSV file."""
    data = {
        "Age": [21.0, 26.0, 21.0, 26.0, 22.0, 30.0, 35.0, 40.0, 32.0, 28.0],
        "Height": [1.62, 1.75, 1.62, 1.75, 1.70, 1.60, 1.80, 1.65, 1.72, 1.68],
        "Weight": [60.0, 70.0, 58.0, 72.0, 65.0, 95.0, 105.0, 90.0, 100.0, 92.0],
        "fcvc": [2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0],
        "ncp": [3.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "ch2o": [2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0],
        "faf": [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "tue": [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "Gender": [
            "Female",
            "Male",
            "Female",
            "Male",
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
        ],
        "family_history": [
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
        "favc": ["no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "yes", "yes"],
        "smoke": ["no", "no", "no", "no", "no", "no", "yes", "no", "no", "no"],
        "scc": ["no", "no", "no", "no", "no", "no", "no", "yes", "no", "no"],
        "mtrans": [
            "Public_Transportation",
            "Bike",
            "Walking",
            "Automobile",
            "Bike",
            "Automobile",
            "Automobile",
            "Walking",
            "Motorbike",
            "Bike",
        ],
        "caec": [
            "Sometimes",
            "Sometimes",
            "Sometimes",
            "Always",
            "Sometimes",
            "Sometimes",
            "no",
            "Frequently",
            "Sometimes",
            "Always",
        ],
        "calc": [
            "no",
            "Sometimes",
            "no",
            "Frequently",
            "no",
            "Sometimes",
            "Always",
            "no",
            "Sometimes",
            "no",
        ],
        "Obesity": [
            "Normal_Weight",
            "Normal_Weight",
            "Normal_Weight",
            "Normal_Weight",
            "Normal_Weight",
            "Obesity_Type_I",
            "Obesity_Type_I",
            "Obesity_Type_I",
            "Obesity_Type_I",
            "Obesity_Type_I",
        ],
    }
    return pd.DataFrame(data)
