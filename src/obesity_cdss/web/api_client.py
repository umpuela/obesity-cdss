from typing import Any

import httpx
import streamlit as st

from obesity_cdss.config import settings


class ApiClient:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_data_dictionary() -> dict[str, Any]:
        """Gets the data dictionary for dynamic rendering."""
        try:
            response = httpx.get(f"{settings.api_base_url}/data-dictionary")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Erro ao conectar com API: {e}")
            return {}

    @staticmethod
    @st.cache_data(ttl=300)
    def get_insights_data() -> dict[str, Any]:
        """Gets aggregated data for EDA."""
        try:
            response = httpx.get(f"{settings.api_base_url}/dashboard-data")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Erro ao buscar insights: {e}")
            return {}

    @staticmethod
    def get_model_metrics() -> dict[str, Any]:
        """Gets model metrics (accuracy, confusion matrix, etc)."""
        try:
            response = httpx.get(f"{settings.api_base_url}/metrics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Erro ao buscar métricas: {e}")
            return {}

    @staticmethod
    def predict(payload: dict[str, Any]) -> dict[str, Any] | None:
        """Sends data for prediction."""
        try:
            response = httpx.post(f"{settings.api_base_url}/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"Erro da API: {e}")
        except Exception as e:
            st.error(f"Erro de conexão: {e}")
        return None
