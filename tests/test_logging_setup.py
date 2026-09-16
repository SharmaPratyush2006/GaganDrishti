"""Tests for the structured logging setup."""

from __future__ import annotations

import json
import logging

import pytest

from depthwizard.config import LoggingConfig
from depthwizard.logging_setup import ROOT_LOGGER_NAME, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    """Leave the depthwizard logger as we found it."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    logger.handlers = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_get_logger_namespaces_under_depthwizard() -> None:
    assert get_logger("thing").name == "depthwizard.thing"
    assert get_logger("depthwizard.ingest").name == "depthwizard.ingest"
    assert get_logger(None).name == ROOT_LOGGER_NAME


def test_setup_logging_is_idempotent() -> None:
    first = setup_logging(LoggingConfig(level="DEBUG"))
    assert len(first.handlers) == 1
    second = setup_logging(LoggingConfig(level="DEBUG"))
    assert first is second
    assert len(second.handlers) == 1, "repeated setup must not duplicate log lines"


def test_text_format_includes_structured_fields(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(level="INFO", format="text"))
    get_logger("test").info("wrote raster", extra={"gsd_m": 0.5, "scene": "s1"})
    err = capsys.readouterr().err
    assert "wrote raster" in err
    assert "gsd_m=0.5" in err
    assert "scene=s1" in err
    assert "INFO" in err


def test_json_format_emits_one_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(level="INFO", format="json"))
    get_logger("test").info("wrote raster", extra={"gsd_m": 0.5})
    line = capsys.readouterr().err.strip()
    payload = json.loads(line)
    assert payload["message"] == "wrote raster"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "depthwizard.test"
    assert payload["gsd_m"] == 0.5
    assert "ts" in payload


def test_level_is_respected(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(level="WARNING", format="text"))
    log = get_logger("test")
    log.info("should be hidden")
    log.warning("should be shown")
    err = capsys.readouterr().err
    assert "should be hidden" not in err
    assert "should be shown" in err
