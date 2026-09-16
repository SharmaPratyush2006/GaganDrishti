"""Tests for provider-agnostic metadata discovery."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import rasterio
from rasterio.transform import Affine

from depthwizard.config import SceneConfig
from depthwizard.ingest.metadata import (
    MetadataCandidates,
    MetadataError,
    MissingMetadataError,
    SunMetadata,
    collect_candidates,
    compute_gsd_m,
    find_sidecars,
    parse_imd,
    read_georeference,
    resolve_sun,
)


def _candidates(path: Path) -> MetadataCandidates:
    with rasterio.open(path) as src:
        return collect_candidates(src, path)


# ---------------------------------------------------------------------------
# The Phase 0 fixture
# ---------------------------------------------------------------------------


def test_resolves_depthwizard_native_tags(fixture_tif: Path, scene: SceneConfig) -> None:
    sun = resolve_sun(_candidates(fixture_tif), fixture_tif)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(scene.sun.elevation_deg)
    assert sun.azimuth_deg == pytest.approx(scene.sun.azimuth_deg)
    assert sun.elevation_provenance.key == "SUN_ELEVATION_DEG"
    assert sun.elevation_provenance.source == "gdal:default"
    assert sun.elevation_provenance.conversion is None


def test_georeference_from_fixture(fixture_tif: Path, scene: SceneConfig) -> None:
    with rasterio.open(fixture_tif) as src:
        geo = read_georeference(src)
    assert geo is not None
    assert geo.crs.to_string() == scene.raster.crs
    assert geo.gsd_m == pytest.approx(scene.raster.gsd_m)
    assert geo.is_north_up
    west, south, east, north = geo.bounds
    assert west == pytest.approx(scene.raster.origin_easting_m)
    assert north == pytest.approx(scene.raster.origin_northing_m)
    assert east == pytest.approx(scene.raster.origin_easting_m + scene.raster.width_m)
    assert south == pytest.approx(scene.raster.origin_northing_m - scene.raster.height_m)


def test_probed_sources_are_recorded(fixture_tif: Path) -> None:
    candidates = _candidates(fixture_tif)
    assert "gdal:default" in candidates.source_labels


# ---------------------------------------------------------------------------
# Alternative provider conventions
# ---------------------------------------------------------------------------


def test_sentinel_style_zenith_is_converted(sentinel_style_tif: Path) -> None:
    """Sentinel-2 reports zenith; elevation = 90 - zenith."""
    sun = resolve_sun(_candidates(sentinel_style_tif), sentinel_style_tif)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(58.0)
    assert sun.azimuth_deg == pytest.approx(160.5)
    assert sun.elevation_provenance.key == "MEAN_SUN_ZENITH_ANGLE"
    assert "90 - zenith" in sun.elevation_provenance.conversion


def test_imd_sidecar_is_found_and_parsed(imd_scene: Path) -> None:
    sidecars = find_sidecars(imd_scene)
    assert [p.name for p in sidecars] == ["maxar_scene.IMD"]

    sun = resolve_sun(_candidates(imd_scene), imd_scene)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(62.7)
    assert sun.azimuth_deg == pytest.approx(151.3)
    assert sun.elevation_provenance.source == "imd:maxar_scene.IMD"
    assert sun.elevation_provenance.key == "meanSunEl"


def test_parse_imd_handles_groups_quotes_and_semicolons() -> None:
    parsed = parse_imd(
        "\n".join(
            [
                "BEGIN_GROUP = IMAGE_1",
                '\tsatId = "WV03";',
                "\tmeanSunEl = 62.7;",
                "\tabsCalFactor = 1.234e-2;",
                "END_GROUP = IMAGE_1",
                "END;",
            ]
        )
    )
    assert parsed["satId"] == "WV03"
    assert parsed["meanSunEl"] == "62.7"
    assert parsed["absCalFactor"] == "1.234e-2"
    assert "BEGIN_GROUP" not in parsed


@pytest.mark.parametrize(
    "elevation_key, azimuth_key",
    [
        ("SUN_ELEVATION", "SUN_AZIMUTH"),  # Landsat / generic GDAL
        ("sun_elevation", "sun_azimuth"),  # Planet
        ("SUNEL", "SUNAZ"),  # WorldView TRE
        ("SolarElevation", "SolarAzimuth"),  # mixed case
        ("solar-elevation", "solar-azimuth"),  # punctuation variant
    ],
)
def test_key_aliases_are_matched_case_and_punctuation_insensitively(
    elevation_key: str, azimuth_key: str
) -> None:
    candidates = MetadataCandidates(
        tables=(("test", {elevation_key: "37.5", azimuth_key: "210.25"}),)
    )
    sun = resolve_sun(candidates)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(37.5)
    assert sun.azimuth_deg == pytest.approx(210.25)


def test_values_with_units_are_parsed() -> None:
    candidates = MetadataCandidates(
        tables=(("test", {"SUN_ELEVATION": "45.5 degrees", "SUN_AZIMUTH": "135.0 deg"}),)
    )
    sun = resolve_sun(candidates)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(45.5)


def test_earlier_source_wins_over_later() -> None:
    """Priority order is source order: native tags beat sidecars."""
    candidates = MetadataCandidates(
        tables=(
            ("gdal:default", {"SUN_ELEVATION_DEG": "45.0", "SUN_AZIMUTH_DEG": "135.0"}),
            ("imd:other.IMD", {"meanSunEl": "10.0", "meanSunAz": "20.0"}),
        )
    )
    sun = resolve_sun(candidates)
    assert sun is not None
    assert sun.elevation_deg == pytest.approx(45.0)
    assert sun.elevation_provenance.source == "gdal:default"


# ---------------------------------------------------------------------------
# Nothing is invented
# ---------------------------------------------------------------------------


def test_absent_sun_resolves_to_none_not_a_default(tif_without_sun: Path) -> None:
    assert resolve_sun(_candidates(tif_without_sun), tif_without_sun) is None


def test_half_present_sun_is_not_enough() -> None:
    """Elevation without azimuth must not yield a partially-invented result."""
    candidates = MetadataCandidates(tables=(("test", {"SUN_ELEVATION": "45.0"}),))
    assert resolve_sun(candidates) is None


def test_empty_values_are_ignored() -> None:
    candidates = MetadataCandidates(
        tables=(
            ("empty", {"SUN_ELEVATION": "", "SUN_AZIMUTH": "  "}),
            ("real", {"SUN_ELEVATION": "30.0", "SUN_AZIMUTH": "90.0"}),
        )
    )
    sun = resolve_sun(candidates)
    assert sun is not None
    assert sun.elevation_provenance.source == "real"


def test_out_of_range_elevation_is_rejected_not_clamped() -> None:
    with pytest.raises(MetadataError, match="outside"):
        SunMetadata(
            elevation_deg=120.0,
            azimuth_deg=10.0,
            elevation_provenance=_p(),
            azimuth_provenance=_p(),
        )


def _p():
    from depthwizard.ingest.metadata import Provenance

    return Provenance(source="test", key="k", raw_value="v")


def test_missing_metadata_error_message_is_actionable() -> None:
    error = MissingMetadataError(
        missing=["sun_elevation_deg"],
        probed_sources=["gdal:default", "imd:scene.IMD"],
        tried_keys={"sun_elevation_deg": ["SUN_ELEVATION", "meanSunEl"]},
        path=Path("scene.tif"),
        hint="supply it as a GeoTIFF tag",
    )
    text = str(error)
    assert "sun_elevation_deg" in text
    assert "gdal:default" in text and "imd:scene.IMD" in text
    assert "SUN_ELEVATION" in text and "meanSunEl" in text
    assert "supply it as a GeoTIFF tag" in text


# ---------------------------------------------------------------------------
# Georeferencing edge cases
# ---------------------------------------------------------------------------


def test_missing_crs_yields_no_georeference(tif_without_crs: Path) -> None:
    with rasterio.open(tif_without_crs) as src:
        assert read_georeference(src) is None


def test_identity_transform_is_not_treated_as_georeferenced(tmp_path: Path) -> None:
    """A CRS plus an identity transform would silently imply a 1 m GSD."""
    import numpy as np

    path = tmp_path / "identity.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=Affine.identity(),
    ) as dst:
        dst.write(np.zeros((8, 8), dtype="uint8"), 1)

    with rasterio.open(path) as src:
        assert read_georeference(src) is None


def test_gsd_from_geographic_crs_is_converted_to_metres(geographic_tif: Path) -> None:
    with rasterio.open(geographic_tif) as src:
        geo = read_georeference(src)
    assert geo is not None
    assert geo.crs.is_geographic
    # 4.5e-6 deg at ~28.6N is roughly half a metre, not 4.5e-6 "metres".
    assert 0.3 < geo.gsd_x_m < 0.7
    assert 0.3 < geo.gsd_y_m < 0.7
    assert "geographic" in geo.gsd_note and "approximate" in geo.gsd_note


def test_gsd_handles_rotated_transform() -> None:
    """A rotated transform still has a well-defined pixel size."""
    from rasterio.crs import CRS

    angle = math.radians(30.0)
    gsd = 2.0
    rotated = Affine(
        gsd * math.cos(angle), -gsd * math.sin(angle), 500000.0,
        gsd * math.sin(angle), gsd * math.cos(angle), 4000000.0,
    )
    gsd_x, gsd_y, _ = compute_gsd_m(CRS.from_epsg(32643), rotated, 100, 100)
    assert gsd_x == pytest.approx(gsd)
    assert gsd_y == pytest.approx(gsd)


def test_gsd_converts_non_metre_projected_units(tmp_path: Path) -> None:
    """EPSG:2229 is in US survey feet; a 3 ft pixel is ~0.914 m."""
    from rasterio.crs import CRS

    crs = CRS.from_epsg(2229)
    transform = Affine(3.0, 0.0, 6_500_000.0, 0.0, -3.0, 1_800_000.0)
    gsd_x, gsd_y, note = compute_gsd_m(crs, transform, 100, 100)
    assert gsd_x == pytest.approx(0.9144, rel=1e-3)
    assert gsd_y == pytest.approx(0.9144, rel=1e-3)
    assert "converted from" in note


def test_non_square_pixels_are_rejected(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "rect.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=Affine(1.0, 0.0, 0.0, 0.0, -2.0, 0.0),
    ) as dst:
        dst.write(np.zeros((8, 8), dtype="uint8"), 1)

    with rasterio.open(path) as src:
        geo = read_georeference(src)
    assert geo is not None
    assert geo.gsd_x_m == pytest.approx(1.0)
    assert geo.gsd_y_m == pytest.approx(2.0)
    with pytest.raises(MetadataError, match="non-square"):
        _ = geo.gsd_m
