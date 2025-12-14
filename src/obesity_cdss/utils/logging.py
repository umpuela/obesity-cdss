import logging
import sys
from pathlib import Path
from types import FrameType
from typing import cast

from loguru import logger

from obesity_cdss.config import settings


class InterceptHandler(logging.Handler):
    """
    Custom logging handler to intercept standard Python logging messages
    and redirect them to Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercepts the standard logging record and redirects it to the Loguru logger.

        Args:
            record (logging.LogRecord): The log record to be processed.
        """
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(log_dir: Path | None = None) -> None:
    """
    Configures the application logging system.

    Args:
        log_dir (Path, optional): Directory path to save log files.
                                  If None, file logging is disabled.
    """
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(settings.log_level)

    logger.remove()

    for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
        _logger = logging.getLogger(_log)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / "app.log"

            logger.add(
                log_file_path,
                rotation="10 MB",
                retention="30 days",
                compression="zip",
                level=settings.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} - {message}",
                enqueue=True,
            )
            logger.info(f"File logging enabled. Logs will be saved to: {log_file_path}")

        except Exception as e:
            logger.error(f"Failed to configure file logging at {log_dir}: {e}")

    logger.info(
        f"Logging initialized. Project: {settings.project_name} v{settings.version}"
    )
