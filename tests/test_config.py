"""Tests for the YAML + dataclass configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from depthwizard.config import (
    AppConfig,
    BuildingSpec,
    ConfigError,
    GroundConfig,
    LoggingConfig,
    RasterConfig,
    SunConfig,
    load_config,
)


def test_default_config_loads(app_config: AppConfig) -> None:
    assert app_config.project.name == "depthwizard"
    assert isinstance(app_config.paths.raw, Path)
    assert isinstance(app_config.logging, LoggingConfig)
    assert isinstance(app_config.scene.raster, RasterConfig)
    assert isinstance(app_config.scene.sun, SunConfig)
    assert isinstance(app_config.scene.ground, GroundConfig)
    assert len(app_config.scene.buildings) > 0
    assert all(isinstance(b, BuildingSpec) for b in app_config.scene.buildings)


def test_default_config_values(app_config: AppConfig) -> None:
    raster = app_config.scene.raster
    assert raster.crs == "EPSG:32643"
    assert raster.gsd_m == pytest.approx(0.5)
    assert raster.width_m == pytest.approx(raster.width_px * raster.gsd_m)
    assert app_config.scene.sun.elevation_deg == pytest.approx(45.0)
    assert app_config.scene.sun.azimuth_deg == pytest.approx(135.0)


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_unknown_key_is_rejected(default_config_path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    raw["scene"]["sun"]["elevaton_deg"] = 30.0  # deliberate typo
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(bad)


def test_missing_required_key_is_rejected(default_config_path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    del raw["scene"]["raster"]["gsd_m"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(bad)


@pytest.mark.parametrize("elevation", [0.0, -5.0, 91.0])
def test_invalid_sun_elevation_rejected(elevation: float) -> None:
    with pytest.raises(ConfigError, match="elevation_deg"):
        SunConfig(elevation_deg=elevation, azimuth_deg=180.0)


def test_azimuth_is_wrapped_into_0_360() -> None:
    assert SunConfig(elevation_deg=45.0, azimuth_deg=-45.0).azimuth_deg == pytest.approx(315.0)
    assert SunConfig(elevation_deg=45.0, azimuth_deg=405.0).azimuth_deg == pytest.approx(45.0)


def test_invalid_logging_values_rejected() -> None:
    with pytest.raises(ConfigError, match="logging.level"):
        LoggingConfig(level="LOUD")
    with pytest.raises(ConfigError, match="logging.format"):
        LoggingConfig(format="xml")


def test_shadow_must_be_darker_than_ground() -> None:
    with pytest.raises(ConfigError, match="darker"):
        GroundConfig(reflectance=0.3, shadow_reflectance=0.4)


def test_building_requires_positive_height() -> None:
    with pytest.raises(ConfigError, match="height_m"):
        BuildingSpec(name="x", x_m=0, y_m=0, width_m=10, depth_m=10, height_m=0)


def test_duplicate_building_names_rejected(default_config_path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    raw["scene"]["buildings"][1]["name"] = raw["scene"]["buildings"][0]["name"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate building name"):
        load_config(bad)
