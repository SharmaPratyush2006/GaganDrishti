"""Acceptance tests for the synthetic fixture generator.

These read the generated GeoTIFF back off disk and verify that its CRS,
transform/GSD and sun metadata survive the round trip, and that the rendered
shadows are geometrically consistent with the configured sun.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS

from depthwizard.config import SceneConfig
from depthwizard.ingest.geotiff import (
    MetadataTags,
    pixel_to_map,
    read_raster,
    read_sun_metadata,
)
from depthwizard.ingest.synthetic import (
    GeneratedFixture,
    footprint_pixels,
    generate_fixture,
    render_scene,
)
from depthwizard.physics.sun import shadow_pixel_offset


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


def test_fixture_files_are_written(generated: GeneratedFixture) -> None:
    assert generated.image_path.is_file()
    assert generated.truth_path.is_file()
    assert generated.height_path is not None and generated.height_path.is_file()


# ---------------------------------------------------------------------------
# Geospatial metadata round trip (the core acceptance criterion)
# ---------------------------------------------------------------------------


def test_crs_round_trips(generated: GeneratedFixture, scene: SceneConfig) -> None:
    result = read_raster(generated.image_path)
    assert result.crs == CRS.from_user_input(scene.raster.crs)
    assert result.crs.is_projected, "GSD is in metres, so the CRS must be projected"


def test_transform_and_gsd_round_trip(generated: GeneratedFixture, scene: SceneConfig) -> None:
    raster = scene.raster
    result = read_raster(generated.image_path)

    assert result.is_north_up
    assert result.gsd_m == pytest.approx(raster.gsd_m)
    # Affine: a = +x pixel size, e = -y pixel size, (c, f) = top-left corner.
    assert result.transform.a == pytest.approx(raster.gsd_m)
    assert result.transform.e == pytest.approx(-raster.gsd_m)
    assert result.transform.c == pytest.approx(raster.origin_easting_m)
    assert result.transform.f == pytest.approx(raster.origin_northing_m)


def test_raster_shape_matches_config(generated: GeneratedFixture, scene: SceneConfig) -> None:
    result = read_raster(generated.image_path)
    assert (result.height, result.width) == (scene.raster.height_px, scene.raster.width_px)
    assert result.array.shape == (scene.raster.height_px, scene.raster.width_px)
    assert result.array.dtype == np.uint8


def test_sun_metadata_round_trips(generated: GeneratedFixture, scene: SceneConfig) -> None:
    sun = read_sun_metadata(generated.image_path)
    assert sun.elevation_deg == pytest.approx(scene.sun.elevation_deg)
    assert sun.azimuth_deg == pytest.approx(scene.sun.azimuth_deg)


def test_descriptive_tags_are_present(generated: GeneratedFixture, scene: SceneConfig) -> None:
    tags = read_raster(generated.image_path).tags
    assert float(tags[MetadataTags.GSD]) == pytest.approx(scene.raster.gsd_m)
    assert tags[MetadataTags.SCENE_NAME] == scene.name
    assert tags[MetadataTags.SYNTHETIC] == "true"
    assert tags[MetadataTags.RUN_ID] == "pytest"
    assert "clockwise from North" in tags[MetadataTags.AZIMUTH_CONVENTION]


def test_georeferencing_places_pixels_where_expected(
    generated: GeneratedFixture, scene: SceneConfig
) -> None:
    """Pixel (0, 0)'s corner is the configured origin; the grid steps by one GSD."""
    raster = scene.raster
    transform = read_raster(generated.image_path).transform

    assert pixel_to_map(transform, 0, 0) == pytest.approx(
        (raster.origin_easting_m, raster.origin_northing_m)
    )
    assert pixel_to_map(transform, 1, 1) == pytest.approx(
        (raster.origin_easting_m + raster.gsd_m, raster.origin_northing_m - raster.gsd_m)
    )
    # Bottom-right corner of the grid.
    assert pixel_to_map(transform, raster.width_px, raster.height_px) == pytest.approx(
        (
            raster.origin_easting_m + raster.width_m,
            raster.origin_northing_m - raster.height_m,
        )
    )


# ---------------------------------------------------------------------------
# Ground truth is preserved
# ---------------------------------------------------------------------------


def test_truth_sidecar_preserves_building_heights(
    generated: GeneratedFixture, scene: SceneConfig
) -> None:
    truth = json.loads(generated.truth_path.read_text(encoding="utf-8"))
    recorded = {b["name"]: b["height_m"] for b in truth["buildings"]}
    configured = {b.name: b.height_m for b in scene.buildings}
    assert recorded == configured


def test_truth_sidecar_records_sun_and_grid(generated: GeneratedFixture, scene: SceneConfig) -> None:
    truth = json.loads(generated.truth_path.read_text(encoding="utf-8"))
    assert truth["sun"]["elevation_deg"] == pytest.approx(scene.sun.elevation_deg)
    assert truth["sun"]["azimuth_deg"] == pytest.approx(scene.sun.azimuth_deg)
    assert truth["raster"]["gsd_m"] == pytest.approx(scene.raster.gsd_m)
    assert truth["raster"]["crs"] == scene.raster.crs
    assert truth["synthetic"] is True


def test_truth_shadow_lengths_follow_the_physics(generated: GeneratedFixture, scene: SceneConfig) -> None:
    """L = h / tan(elevation); at 45 deg that is L == h."""
    truth = json.loads(generated.truth_path.read_text(encoding="utf-8"))
    tan_elev = math.tan(math.radians(scene.sun.elevation_deg))
    for record in truth["buildings"]:
        assert record["expected_shadow_length_m"] == pytest.approx(record["height_m"] / tan_elev)
        assert record["expected_shadow_length_px"] == pytest.approx(
            record["expected_shadow_length_m"] / scene.raster.gsd_m
        )


def test_height_raster_holds_exactly_the_configured_heights(
    generated: GeneratedFixture, scene: SceneConfig
) -> None:
    assert generated.height_path is not None
    heights = read_raster(generated.height_path).array
    assert heights.dtype == np.float32

    for building in scene.buildings:
        fp = footprint_pixels(building, scene.raster)
        patch = heights[fp.row_min : fp.row_max, fp.col_min : fp.col_max]
        assert np.all(patch == pytest.approx(building.height_m))

    # Bare ground is exactly zero, and nothing exceeds the tallest building.
    assert heights.min() == pytest.approx(0.0)
    assert heights.max() == pytest.approx(max(b.height_m for b in scene.buildings))


# ---------------------------------------------------------------------------
# The rendered scene is geometrically consistent
# ---------------------------------------------------------------------------


def test_scene_has_ground_roofs_and_shadows(scene: SceneConfig) -> None:
    render = render_scene(scene)
    assert render.building_mask.any(), "no buildings were rendered"
    assert render.shadow_mask.any(), "no shadows were rendered"
    # Shadows are cast on the ground, never on the roofs that cast them.
    assert not (render.shadow_mask & render.building_mask).any()


def test_shadows_are_darker_and_roofs_brighter_than_ground(scene: SceneConfig) -> None:
    render = render_scene(scene)
    ground_only = ~(render.building_mask | render.shadow_mask)
    assert render.reflectance[render.shadow_mask].max() < render.reflectance[ground_only].min()
    assert render.reflectance[render.building_mask].min() > render.reflectance[ground_only].max()


def test_shadow_tip_lands_where_the_physics_predicts(scene: SceneConfig) -> None:
    """The far end of each swept shadow is shadowed, at the predicted offset.

    This checks length *and* direction at once: the probe point is computed
    purely from h, sun elevation, sun azimuth and GSD - never measured off the
    image.

    The probe is the footprint's *leading* corner (the one furthest along the
    shadow direction) plus the offset, not the centroid. For a building that is
    wider than its shadow is long - ``low_d`` is 25 m wide but only 6 m tall -
    the centroid plus offset still lands on the roof, which is not shadowed.
    """
    render = render_scene(scene)
    n_rows, n_cols = render.shadow_mask.shape

    for building in scene.buildings:
        fp = render.footprints[building.name]
        d_row, d_col = shadow_pixel_offset(
            height_m=building.height_m,
            sun_elevation_deg=scene.sun.elevation_deg,
            sun_azimuth_deg=scene.sun.azimuth_deg,
            gsd_m=scene.raster.gsd_m,
        )
        step_row, step_col = int(round(d_row)), int(round(d_col))
        if step_row == 0 and step_col == 0:
            continue  # sun too high to cast a shadow of even one pixel

        # Leading edge of the footprint along each axis.
        base_row = fp.row_min if step_row <= 0 else fp.row_max - 1
        base_col = fp.col_min if step_col <= 0 else fp.col_max - 1
        tip_row, tip_col = base_row + step_row, base_col + step_col

        assert 0 <= tip_row < n_rows and 0 <= tip_col < n_cols, (
            f"{building.name}: shadow tip falls outside the raster; widen the scene"
        )
        assert not render.building_mask[tip_row, tip_col], (
            f"{building.name}: probe point ({tip_row}, {tip_col}) is on a roof, "
            "so this assertion would be vacuous"
        )
        assert render.shadow_mask[tip_row, tip_col], (
            f"{building.name}: expected shadow at the predicted tip ({tip_row}, {tip_col})"
        )

        # One pixel further along the shadow is beyond the tip: not shadowed.
        beyond_row = tip_row + (1 if step_row > 0 else -1 if step_row < 0 else 0)
        beyond_col = tip_col + (1 if step_col > 0 else -1 if step_col < 0 else 0)
        if 0 <= beyond_row < n_rows and 0 <= beyond_col < n_cols:
            assert not render.shadow_mask[beyond_row, beyond_col], (
                f"{building.name}: shadow extends past the predicted tip"
            )


def test_shadow_falls_on_the_far_side_of_the_sun(scene: SceneConfig) -> None:
    """With the configured SE sun (azimuth 135), shadows must run to the NW."""
    render = render_scene(scene)
    assert scene.sun.azimuth_deg == pytest.approx(135.0), "this test assumes the default sun"

    shadow_rows, shadow_cols = np.nonzero(render.shadow_mask)
    building_rows, building_cols = np.nonzero(render.building_mask)
    # North is a smaller row index, West is a smaller column index.
    assert shadow_rows.mean() < building_rows.mean()
    assert shadow_cols.mean() < building_cols.mean()


def test_taller_buildings_cast_more_shadow(scene: SceneConfig) -> None:
    """Shadow area grows with height for equal-footprint buildings."""
    tallest = max(scene.buildings, key=lambda b: b.height_m)
    shortest = min(scene.buildings, key=lambda b: b.height_m)

    def shadow_area(height_m: float) -> int:
        from dataclasses import replace

        one = replace(tallest, height_m=height_m)
        solo = replace(scene, buildings=(one,))
        return int(render_scene(solo).shadow_mask.sum())

    assert shadow_area(tallest.height_m) > shadow_area(shortest.height_m)


# ---------------------------------------------------------------------------
# Generator behaviour
# ---------------------------------------------------------------------------


def test_generation_is_deterministic(scene: SceneConfig, tmp_path: Path) -> None:
    a = generate_fixture(scene, tmp_path / "a")
    b = generate_fixture(scene, tmp_path / "b")
    assert np.array_equal(read_raster(a.image_path).array, read_raster(b.image_path).array)


def test_height_raster_can_be_skipped(scene: SceneConfig, tmp_path: Path) -> None:
    fixture = generate_fixture(scene, tmp_path / "no_height", write_height_raster=False)
    assert fixture.height_path is None
    assert not (tmp_path / "no_height" / f"{scene.name}_truth_height.tif").exists()


def test_building_outside_the_raster_is_rejected(scene: SceneConfig) -> None:
    from dataclasses import replace

    off_grid = replace(scene.buildings[0], x_m=scene.raster.width_m + 10.0)
    with pytest.raises(ValueError, match="outside"):
        footprint_pixels(off_grid, scene.raster)


def test_overhead_sun_produces_no_shadow(scene: SceneConfig) -> None:
    from dataclasses import replace

    noon = replace(scene, sun=replace(scene.sun, elevation_deg=90.0))
    render = render_scene(noon)
    assert not render.shadow_mask.any()
    assert render.building_mask.any()
