"""Tests for GeoTIFFLoader, ImageLoader and the mode router.

This file carries the Phase 1 acceptance checks: load the Phase 0 synthetic
GeoTIFF and confirm its CRS, GSD and sun angles match the fixture metadata;
load a PNG/JPG and confirm it routes to Mode.RELATIVE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthwizard.config import SceneConfig
from depthwizard.ingest.loaders import GeoTIFFLoader, ImageLoader
from depthwizard.ingest.metadata import MissingMetadataError, UnsupportedInputError
from depthwizard.ingest.router import load_scene, read_metadata, route
from depthwizard.mode import Mode


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------


def test_modes_exist_and_describe_themselves() -> None:
    assert Mode.ABSOLUTE.value == "absolute"
    assert Mode.RELATIVE.value == "relative"
    assert Mode.ABSOLUTE.is_metric
    assert not Mode.RELATIVE.is_metric
    assert Mode("absolute") is Mode.ABSOLUTE


# ---------------------------------------------------------------------------
# ACCEPTANCE: the synthetic GeoTIFF loads in absolute mode with correct values
# ---------------------------------------------------------------------------


def test_geotiff_loader_matches_fixture_metadata(fixture_tif: Path, scene: SceneConfig) -> None:
    metadata = GeoTIFFLoader().read_metadata(fixture_tif)

    assert metadata.mode is Mode.ABSOLUTE
    assert metadata.crs.to_string() == scene.raster.crs
    assert metadata.gsd_m == pytest.approx(scene.raster.gsd_m)
    assert metadata.sun_elevation_deg == pytest.approx(scene.sun.elevation_deg)
    assert metadata.sun_azimuth_deg == pytest.approx(scene.sun.azimuth_deg)
    assert metadata.downgrade_reason is None


def test_geotiff_loader_reads_geotransform(fixture_tif: Path, scene: SceneConfig) -> None:
    geo = GeoTIFFLoader().read_metadata(fixture_tif).georeference
    assert geo is not None
    assert geo.transform.a == pytest.approx(scene.raster.gsd_m)
    assert geo.transform.e == pytest.approx(-scene.raster.gsd_m)
    assert geo.transform.c == pytest.approx(scene.raster.origin_easting_m)
    assert geo.transform.f == pytest.approx(scene.raster.origin_northing_m)


def test_geotiff_loader_reads_pixels_as_hwc(fixture_tif: Path, scene: SceneConfig) -> None:
    loaded = GeoTIFFLoader().load(fixture_tif)
    assert loaded.array.shape == (scene.raster.height_px, scene.raster.width_px, 1)
    assert loaded.array.dtype == np.uint8
    assert loaded.band(0).shape == (scene.raster.height_px, scene.raster.width_px)
    assert loaded.mode is Mode.ABSOLUTE


def test_sun_values_carry_provenance(fixture_tif: Path) -> None:
    """Every value must be traceable to a source and a key."""
    sun = GeoTIFFLoader().read_metadata(fixture_tif).sun
    assert sun is not None
    assert sun.elevation_provenance.source == "gdal:default"
    assert sun.elevation_provenance.key == "SUN_ELEVATION_DEG"
    assert sun.elevation_provenance.raw_value == "45.0"


def test_summary_reports_the_acceptance_values(fixture_tif: Path) -> None:
    text = GeoTIFFLoader().read_metadata(fixture_tif).summary()
    assert "EPSG:32643" in text
    assert "0.5" in text
    assert "45" in text and "135" in text


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GeoTIFFLoader().read_metadata(tmp_path / "nope.tif")


# ---------------------------------------------------------------------------
# ACCEPTANCE: PNG/JPG route to relative mode
# ---------------------------------------------------------------------------


def test_png_routes_to_relative(png_path: Path) -> None:
    decision = route(png_path)
    assert decision.mode is Mode.RELATIVE
    assert decision.loader == "ImageLoader"
    assert decision.metadata.crs is None
    assert decision.metadata.gsd_m is None
    assert decision.metadata.sun is None


def test_jpg_routes_to_relative(jpg_path: Path) -> None:
    decision = route(jpg_path)
    assert decision.mode is Mode.RELATIVE
    assert decision.loader == "ImageLoader"


def test_image_loader_preserves_pixels(png_path: Path, fixture_tif: Path) -> None:
    """PNG is lossless, so greyscale pixels must survive the round trip."""
    from depthwizard.ingest.geotiff import read_raster

    loaded = ImageLoader().load(png_path)
    assert loaded.array.shape[2] == 1
    assert np.array_equal(loaded.band(0), read_raster(fixture_tif).array)


def test_image_loader_returns_rgb_not_bgr(rgb_png_path: Path) -> None:
    """OpenCV decodes BGR; the loader must hand back RGB."""
    loaded = ImageLoader().load(rgb_png_path)
    assert loaded.band_count == 3
    # The fixture wrote a left-to-right ramp into the BLUE channel, and a flat
    # 40 into green. After BGR->RGB the ramp must be band 2, not band 0.
    assert loaded.band(1).min() == loaded.band(1).max() == 40
    ramp = loaded.band(2)
    assert ramp[0, 0] < ramp[0, -1]


def test_image_loader_never_invents_georeferencing(png_path: Path) -> None:
    metadata = ImageLoader().read_metadata(png_path)
    assert metadata.georeference is None
    assert metadata.rpc is None
    assert "no geospatial metadata" in metadata.probed_sources[0]
    assert "relative" in metadata.downgrade_reason


def test_relative_scene_cannot_be_forced_absolute(png_path: Path) -> None:
    with pytest.raises(MissingMetadataError) as excinfo:
        route(png_path, require=Mode.ABSOLUTE)
    assert "sun_elevation_deg" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Failing loudly rather than inventing values
# ---------------------------------------------------------------------------


def test_geotiff_without_sun_raises_by_default(tif_without_sun: Path) -> None:
    with pytest.raises(MissingMetadataError) as excinfo:
        GeoTIFFLoader().read_metadata(tif_without_sun)

    message = str(excinfo.value)
    assert "sun_elevation_deg" in message and "sun_azimuth_deg" in message
    assert "probed sources" in message
    assert "gdal:default" in message
    # The message must name real alias keys, so the user knows what to supply.
    assert "meanSunEl" in message and "MEAN_SUN_ZENITH_ANGLE" in message
    assert "hint" in message


def test_geotiff_without_sun_can_be_explicitly_downgraded(tif_without_sun: Path) -> None:
    metadata = GeoTIFFLoader(require_absolute=False).read_metadata(tif_without_sun)
    assert metadata.mode is Mode.RELATIVE
    assert metadata.sun is None
    assert "no sun elevation/azimuth" in metadata.downgrade_reason
    # Georeferencing is still available; only the sun is missing.
    assert metadata.crs is not None
    assert metadata.gsd_m == pytest.approx(0.5)


def test_router_downgrades_loudly(tif_without_sun: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="depthwizard"):
        decision = route(tif_without_sun)

    assert decision.mode is Mode.RELATIVE
    assert decision.was_downgraded
    assert any("downgraded to relative mode" in record.message for record in caplog.records)


def test_router_can_refuse_to_downgrade(tif_without_sun: Path) -> None:
    with pytest.raises(MissingMetadataError):
        route(tif_without_sun, require=Mode.ABSOLUTE)


def test_geotiff_without_crs_downgrades(tif_without_crs: Path) -> None:
    metadata = GeoTIFFLoader(require_absolute=False).read_metadata(tif_without_crs)
    assert metadata.mode is Mode.RELATIVE
    assert metadata.georeference is None
    assert metadata.sun is not None, "sun tags are present even without a CRS"
    assert "no CRS" in metadata.downgrade_reason


def test_require_absolute_on_metadata_object(tif_without_sun: Path) -> None:
    metadata = GeoTIFFLoader(require_absolute=False).read_metadata(tif_without_sun)
    with pytest.raises(MissingMetadataError):
        metadata.require_absolute()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_geotiff_routes_to_absolute(fixture_tif: Path) -> None:
    decision = route(fixture_tif)
    assert decision.mode is Mode.ABSOLUTE
    assert decision.loader == "GeoTIFFLoader"
    assert not decision.was_downgraded


def test_imd_sidecar_scene_reaches_absolute(imd_scene: Path) -> None:
    """Sun angles found only in the sidecar are still enough for absolute mode."""
    metadata = read_metadata(imd_scene)
    assert metadata.mode is Mode.ABSOLUTE
    assert metadata.sun_elevation_deg == pytest.approx(62.7)
    assert metadata.sun.elevation_provenance.source.startswith("imd:")


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="unsupported input"):
        route(path)


@pytest.mark.parametrize("suffix", [".tif", ".TIF", ".tiff"])
def test_geotiff_suffixes_are_case_insensitive(suffix: str) -> None:
    assert GeoTIFFLoader.handles(Path(f"scene{suffix}"))


@pytest.mark.parametrize("suffix", [".png", ".PNG", ".jpg", ".jpeg"])
def test_image_suffixes_are_case_insensitive(suffix: str) -> None:
    assert ImageLoader.handles(Path(f"scene{suffix}"))


def test_load_scene_dispatches_to_the_right_loader(fixture_tif: Path, png_path: Path) -> None:
    assert load_scene(fixture_tif).mode is Mode.ABSOLUTE
    assert load_scene(png_path).mode is Mode.RELATIVE


def test_load_scene_gives_both_inputs_the_same_array_rank(
    fixture_tif: Path, png_path: Path
) -> None:
    """One shape contract regardless of source: (rows, cols, bands)."""
    tif_scene = load_scene(fixture_tif)
    png_scene = load_scene(png_path)
    assert tif_scene.array.ndim == png_scene.array.ndim == 3
    assert tif_scene.array.shape[:2] == png_scene.array.shape[:2]
