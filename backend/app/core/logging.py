import logging
import sys
from backend.app.core.config import settings


def setup_logging() -> None:
    """
    Configure application-wide structured logging.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
    else:
        root_logger.handlers = [console_handler]

    # Silence noisy external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Helper to retrieve a logger with standard formatting.
    """
    return logging.getLogger(name)
