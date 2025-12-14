from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from obesity_cdss.api.routers import router
from obesity_cdss.api.services import ModelService
from obesity_cdss.config import settings
from obesity_cdss.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the application lifecycle (Startup and Shutdown events)."""
    setup_logging(log_dir=settings.log_dir_api)
    logger.info("Starting Obesity Risk CDSS API...")

    try:
        ModelService().load_artifacts()
        logger.info("Warm-up complete: Model artifacts loaded into memory.")
    except Exception as e:
        logger.warning(
            f"Warm-up failed during startup: {e}. "
            "System will retry loading on the first request."
        )

    yield

    logger.info("Shutting down API...")


def create_app() -> FastAPI:
    """
    Application Factory Pattern. Creates and configures the FastAPI application
    instance, setting up middleware (CORS), routers, and lifecycle events.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="Clinical Decision Support System API for Obesity Risk Prediction",
        lifespan=lifespan,
    )

    app.mount(
        "/static", StaticFiles(directory="src/obesity_cdss/api/static"), name="static"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()
