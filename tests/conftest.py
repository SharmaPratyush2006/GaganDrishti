"""Shared pytest fixtures.

Phase 1 tests run against the Phase 0 synthetic GeoTIFF, plus a family of
deliberately awkward variants -- no sun tags, no CRS, a geographic CRS, a
Sentinel-style zenith angle, a Maxar-style .IMD sidecar -- so that metadata
discovery is exercised on more than the one format DepthWizard itself writes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from depthwizard.config import AppConfig, SceneConfig, load_config
from depthwizard.ingest.geotiff import (
    MetadataTags,
    read_raster,
    transform_from_raster_config,
    write_single_band,
)
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
    """Generate the default 512x512 fixture once per session, into a temp dir."""
    out_dir = tmp_path_factory.mktemp("synthetic")
    return generate_fixture(scene, out_dir, run_id="pytest")


@pytest.fixture(scope="session")
def fixture_tif(generated: GeneratedFixture) -> Path:
    """The Phase 0 synthetic GeoTIFF: EPSG:32643, 0.5 m GSD, sun 45/135."""
    return generated.image_path


# ---------------------------------------------------------------------------
# A larger scene, so tiling has more than one tile to work with
# ---------------------------------------------------------------------------

LARGE_WIDTH_PX = 1200
LARGE_HEIGHT_PX = 900


@pytest.fixture(scope="session")
def large_fixture(scene: SceneConfig, tmp_path_factory: pytest.TempPathFactory) -> GeneratedFixture:
    """A 1200x900 scene, which tiles to a 2x3 grid at 512/64."""
    big = replace(
        scene,
        name="synthetic_large",
        raster=replace(scene.raster, width_px=LARGE_WIDTH_PX, height_px=LARGE_HEIGHT_PX),
    )
    return generate_fixture(big, tmp_path_factory.mktemp("synthetic_large"), run_id="pytest")


# ---------------------------------------------------------------------------
# Plain images (always relative mode)
# ---------------------------------------------------------------------------


def _write_image(path: Path, array: np.ndarray) -> Path:
    """Encode via imencode+tofile, the Windows-safe path-agnostic pattern."""
    ok, buffer = cv2.imencode(path.suffix, array)
    assert ok, f"failed to encode {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
    return path


@pytest.fixture(scope="session")
def png_path(fixture_tif: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture's pixels as a plain greyscale PNG, stripped of all metadata."""
    array = read_raster(fixture_tif).array
    return _write_image(tmp_path_factory.mktemp("images") / "scene.png", array)


@pytest.fixture(scope="session")
def rgb_png_path(fixture_tif: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3-band PNG with distinguishable channels, to check band ordering."""
    grey = read_raster(fixture_tif).array
    # BGR on disk: blue ramp, green constant, red = the scene.
    blue = np.linspace(0, 255, grey.shape[1], dtype=np.uint8)[None, :].repeat(grey.shape[0], 0)
    green = np.full_like(grey, 40)
    bgr = np.dstack([blue, green, grey])
    return _write_image(tmp_path_factory.mktemp("images_rgb") / "scene_rgb.png", bgr)


@pytest.fixture(scope="session")
def jpg_path(fixture_tif: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    array = read_raster(fixture_tif).array
    return _write_image(tmp_path_factory.mktemp("images_jpg") / "scene.jpg", array)


# ---------------------------------------------------------------------------
# Awkward GeoTIFF variants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tif_without_sun(
    generated: GeneratedFixture, scene: SceneConfig, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Georeferenced, but carrying no sun angles anywhere."""
    source = read_raster(generated.image_path)
    return write_single_band(
        tmp_path_factory.mktemp("no_sun") / "no_sun.tif",
        source.array,
        crs=scene.raster.crs,
        transform=transform_from_raster_config(scene.raster),
        tags={MetadataTags.SCENE_NAME: "no_sun", "NOTE": "sun angles deliberately absent"},
    )


@pytest.fixture(scope="session")
def tif_without_crs(
    generated: GeneratedFixture, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Sun angles present, but no CRS and no real geotransform."""
    import rasterio
    from rasterio.transform import Affine

    source = read_raster(generated.image_path)
    path = tmp_path_factory.mktemp("no_crs") / "no_crs.tif"
    profile = {
        "driver": "GTiff",
        "height": source.array.shape[0],
        "width": source.array.shape[1],
        "count": 1,
        "dtype": source.array.dtype.name,
        "crs": None,
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(source.array, 1)
        dst.update_tags(SUN_ELEVATION_DEG="45.0", SUN_AZIMUTH_DEG="135.0")
    return path


@pytest.fixture(scope="session")
def sentinel_style_tif(
    generated: GeneratedFixture, scene: SceneConfig, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Sentinel-2 style: zenith instead of elevation, so 90 - 32 = 58 deg."""
    source = read_raster(generated.image_path)
    return write_single_band(
        tmp_path_factory.mktemp("sentinel") / "sentinel_style.tif",
        source.array,
        crs=scene.raster.crs,
        transform=transform_from_raster_config(scene.raster),
        tags={
            "MEAN_SUN_ZENITH_ANGLE": "32.0",
            "MEAN_SUN_AZIMUTH_ANGLE": "160.5",
        },
    )


@pytest.fixture(scope="session")
def imd_scene(
    generated: GeneratedFixture, scene: SceneConfig, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Maxar style: no sun tags in the TIFF, a .IMD sidecar beside it."""
    directory = tmp_path_factory.mktemp("imd")
    source = read_raster(generated.image_path)
    tif = write_single_band(
        directory / "maxar_scene.tif",
        source.array,
        crs=scene.raster.crs,
        transform=transform_from_raster_config(scene.raster),
        tags={MetadataTags.SCENE_NAME: "maxar_scene"},
    )
    (directory / "maxar_scene.IMD").write_text(
        "\n".join(
            [
                "BEGIN_GROUP = IMAGE_1",
                '\tsatId = "WV03";',
                "\tmeanSunAz = 151.3;",
                "\tmeanSunEl = 62.7;",
                "\tmeanSatAz = 24.1;",
                "\tmeanOffNadirViewAngle = 12.4;",
                "END_GROUP = IMAGE_1",
                "END;",
            ]
        ),
        encoding="utf-8",
    )
    return tif


@pytest.fixture(scope="session")
def geographic_tif(
    generated: GeneratedFixture, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """EPSG:4326, where pixel size is in degrees and needs converting to metres."""
    from rasterio.transform import Affine

    source = read_raster(generated.image_path)
    # ~0.5 m at the equator-ish latitude used here.
    degrees_per_pixel = 4.5e-6
    transform = Affine(degrees_per_pixel, 0.0, 77.2, 0.0, -degrees_per_pixel, 28.6)
    return write_single_band(
        tmp_path_factory.mktemp("geographic") / "geographic.tif",
        source.array,
        crs="EPSG:4326",
        transform=transform,
        tags={"SUN_ELEVATION": "50.0", "SUN_AZIMUTH": "120.0"},
    )
