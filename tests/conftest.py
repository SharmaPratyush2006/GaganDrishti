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


# ---------------------------------------------------------------------------
# Phase 2 helpers: analytic shadow masks
# ---------------------------------------------------------------------------
#
# The synthetic GeoTIFF is rendered at sun azimuth 135 deg, which is the worst
# case for a square pixel grid: the shadow runs exactly along the diagonal, so
# consecutive pixel centres along every ray are sqrt(2) px apart *and every ray
# shares the same phase*. The representable shadow lengths are therefore the
# multiples of sqrt(2), and no amount of averaging over rays recovers a length
# that falls between two of them.
#
# To test the measurement geometry itself, rather than that quantization, these
# helpers build a shadow mask analytically from a known length, at an azimuth
# the caller chooses. At an axis-aligned azimuth the ray step is exactly 1 px
# and the geometry is exactly representable, so recovery is exact.


def shadow_ray_step_px(sun_azimuth_deg: float) -> float:
    """Spacing between consecutive pixel centres along the shadow ray.

    1.0 for an axis-aligned shadow, sqrt(2) for a 45 degree diagonal. This is
    the grid's resolution limit for a single ray, and therefore the scale of
    the only error an exact measurement can still make.
    """
    from depthwizard.physics.sun import shadow_direction_pixels

    d_row, d_col = shadow_direction_pixels(sun_azimuth_deg)
    return 1.0 / max(abs(d_row), abs(d_col))


def build_analytic_shadow_mask(
    shape: tuple[int, int],
    *,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    sun_azimuth_deg: float,
    shadow_length_px: float,
) -> np.ndarray:
    """A shadow mask built analytically from a rectangular footprint.

    The shadow of a flat-roofed box is its footprint swept from the base to the
    tip, so a pixel centre ``p`` is shadow exactly when ``p - t * direction``
    lies inside the footprint for some ``t`` in ``[0, shadow_length_px]``. The
    footprint itself is cleared: a roof is not its own shadow.

    That condition is solved in closed form rather than by sampling ``t``. For
    each axis the constraint is an interval in ``t``; the pixel is shadow when
    the row interval, the column interval and ``[0, L]`` share a point. So the
    mask has no sweep-resolution error of its own, and the only quantization
    left is the pixel grid -- which is the thing under test.
    """
    from depthwizard.physics.sun import shadow_direction_pixels

    d_row, d_col = shadow_direction_pixels(sun_azimuth_deg)
    length = float(shadow_length_px)
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]

    # Feasible t interval, intersected axis by axis, starting from [0, L].
    low = np.zeros(shape, dtype=np.float64)
    high = np.full(shape, length, dtype=np.float64)

    for coordinate, delta, lower, upper in (
        (rows, d_row, row_min - 0.5, row_max - 0.5),
        (cols, d_col, col_min - 0.5, col_max - 0.5),
    ):
        if abs(delta) < 1e-15:
            # No travel along this axis: the pixel must already be in range.
            inside = (coordinate >= lower) & (coordinate <= upper)
            high = np.where(inside, high, -np.inf)
            continue
        # lower <= coordinate - t * delta <= upper, solved for t.
        bound_a = (coordinate - upper) / delta
        bound_b = (coordinate - lower) / delta
        low = np.maximum(low, np.minimum(bound_a, bound_b))
        high = np.minimum(high, np.maximum(bound_a, bound_b))

    mask = np.broadcast_to(low <= high, shape).copy()
    mask[row_min:row_max, col_min:col_max] = False
    return mask


@pytest.fixture
def analytic_shadow_mask():
    """Factory fixture wrapping :func:`build_analytic_shadow_mask`."""
    return build_analytic_shadow_mask
