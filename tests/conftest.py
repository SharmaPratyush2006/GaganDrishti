"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from depthwizard.config import AppConfig, SceneConfig, load_config
from depthwizard.ingest.synthetic import GeneratedFixture, generate_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


@pytest.fixture(scope="session")
def app_config(default_config_path: Path) -> AppConfig:
    return load_config(default_config_path)


@pytest.fixture(scope="session")
def scene(app_config: AppConfig) -> SceneConfig:
    return app_config.scene


@pytest.fixture(scope="session")
def generated(scene: SceneConfig, tmp_path_factory: pytest.TempPathFactory) -> GeneratedFixture:
    """Generate the default fixture once per test session, into a temp dir."""
    out_dir = tmp_path_factory.mktemp("synthetic")
    return generate_fixture(scene, out_dir, run_id="pytest")
