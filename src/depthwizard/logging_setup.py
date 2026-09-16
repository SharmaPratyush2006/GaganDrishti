"""Structured logging for DepthWizard.

Two output shapes are supported, chosen by ``logging.format`` in the config:

``text``
    Human-readable single line, with any structured fields appended as
    ``key=value`` pairs::

        2026-09-16 10:31:04 | INFO     | depthwizard.ingest.synthetic | wrote raster | path=... gsd_m=0.5

``json``
    One JSON object per line, suitable for ingestion by log tooling::

        {"ts": "...", "level": "INFO", "logger": "...", "message": "...", "gsd_m": 0.5}

Structured fields are passed through the standard ``extra=`` mechanism::

    log = get_logger(__name__)
    log.info("wrote raster", extra={"path": str(p), "gsd_m": 0.5})

Anything in ``extra`` that is not a standard :class:`logging.LogRecord`
attribute is treated as a structured field and included in both formats.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from depthwizard.config import LoggingConfig

__all__ = ["setup_logging", "get_logger", "JsonFormatter", "TextFormatter"]

ROOT_LOGGER_NAME = "depthwizard"

# Attributes present on every LogRecord. Anything else came from `extra=`.
_STANDARD_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


def _structured_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the user-supplied ``extra=`` fields attached to ``record``."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
    }


def _isoformat(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")


class TextFormatter(logging.Formatter):
    """Human-readable formatter that appends structured fields as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = _isoformat(record)
        base = f"{stamp} | {record.levelname:<8} | {record.name} | {record.getMessage()}"
        extras = _structured_fields(record)
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _isoformat(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_structured_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str keeps Path / numpy scalars from breaking serialisation.
        return json.dumps(payload, default=str)


def setup_logging(config: LoggingConfig | Mapping[str, Any] | None = None) -> logging.Logger:
    """Configure and return the ``depthwizard`` root logger.

    Safe to call more than once: existing handlers are replaced rather than
    stacked, so repeated calls never produce duplicated log lines.
    """
    if config is None:
        config = LoggingConfig()
    elif isinstance(config, Mapping):
        config = LoggingConfig(**config)

    formatter: logging.Formatter = (
        JsonFormatter() if config.format == "json" else TextFormatter()
    )

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    logger.addHandler(handler)
    logger.setLevel(config.level)
    # Keep our records out of the global root logger's handlers.
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``depthwizard`` namespace.

    ``get_logger(__name__)`` from inside the package returns that module's
    logger unchanged; any other name is nested under ``depthwizard.``.
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
