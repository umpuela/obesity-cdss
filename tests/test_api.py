from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok"


def test_dictionary_endpoint(client: TestClient) -> None:
    response = client.get("/data-dictionary")

    if response.status_code == 200:
        data = response.json()
        assert "gender" in data
        assert data["gender"]["type"] == "cat_nominal"


def test_predict_endpoint(client: TestClient, sample_payload: dict[str, Any]) -> None:
    response = client.post("/predict", json=sample_payload)

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "label" in data
        assert "probability" in data
        assert isinstance(data["probability"], float)

    elif response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        assert (
            response.json()["detail"] == "Modelo de ML não carregado ou não encontrado."
        )

    else:
        pytest.fail(f"Status code inesperado: {response.status_code}")
