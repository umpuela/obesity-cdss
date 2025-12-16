from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse
from loguru import logger

from obesity_cdss.api.schemas import PatientData, PredictionResponse
from obesity_cdss.api.services import InsightsService, ModelService
from obesity_cdss.config import settings
from obesity_cdss.utils.theme import CUSTOM_PALETTE

router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """
    Root endpoint with an stylized HTML interface.
    """
    css_vars = "\n".join(f"--{key}: {value};" for key, value in CUSTOM_PALETTE.items())

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Obesity Risk CDSS API</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg" />

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap">
        <link
        rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0"
        />
        <link rel="stylesheet" href="/static/styles.css" />

        <style>
        :root {{
            {css_vars}
        }}
        </style>
    </head>
    <body>
        <div class="container">
        <h1>
            <span
            class="material-symbols-outlined"
            style="font-size: 32px; vertical-align: bottom"
            >body_fat</span
            >
            API para Preditor de Obesidade
        </h1>
        <p>Sistema de Suporte à Decisão Clínica</p>

        <div class="menu">
            <a href="/docs">
            <span class="material-symbols-outlined">api</span>
            Documentação (Swagger)
            </a>

            <a href="/health">
            <span class="material-symbols-outlined">monitor_heart</span>
            Verificação de saúde da API
            </a>

            <a href="/data-dictionary">
            <span class="material-symbols-outlined">menu_book</span>
            Dicionário de dados
            </a>

            <a href="/metrics">
            <span class="material-symbols-outlined">analytics</span>
            Métricas do modelo
            </a>
        </div>

        <div class="footer">v{settings.version} | Modelo de Machine Learning</div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """
    Verifies if the API is online.
    """
    return {"status": "ok", "version": settings.version}


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def get_model_metrics() -> dict[str, Any]:
    """
    Returns the performance metrics of the current model.
    """
    try:
        service = ModelService()
        return service.get_metrics()

    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar métricas.",
        ) from e


@router.get("/data-dictionary", status_code=status.HTTP_200_OK)
async def get_data_dictionary() -> dict[str, Any]:
    """
    Returns the data dictionary for dynamic rendering of the frontend.
    """
    try:
        service = ModelService()
        return service.get_data_dictionary()

    except Exception as e:
        logger.error(f"Failed to read data dictionary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao carregar metadados.",
        ) from e


@router.get("/dashboard-data", status_code=status.HTTP_200_OK)
async def get_distributions() -> dict[str, Any]:
    """
    Returns grouped data to build charts in the dashboard.
    """
    try:
        service = InsightsService()
        return service.get_distributions()

    except FileNotFoundError:
        logger.warning("Service reported dataset missing.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dados base não encontrados no servidor",
        ) from None

    except Exception as e:
        logger.error(f"Unhandled error in insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar dados analíticos.",
        ) from e


@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict(patient: PatientData) -> PredictionResponse:
    """
    Receives patient data and returns the risk of obesity classification.
    """
    logger.info(
        "Receiving prediction request to patient: "
        f"{patient.age} years old, {patient.gender}"
    )

    try:
        service = ModelService()
        result = service.predict(patient)

        logger.info(f"Prediction result: {result.label} ({result.probability:.2%})")
        return result

    except FileNotFoundError as e:
        logger.critical(f"Unavailable service (missing artifacts): {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de ML não carregado ou não encontrado.",
        ) from e
    except Exception as e:
        logger.error(f"Unhandled error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar a predição.",
        ) from e
