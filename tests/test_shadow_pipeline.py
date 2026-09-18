"""The end-to-end Phase 2 pipeline, and recovery of the synthetic fixture.

The recovery tests here answer a specific question: how much of the residual
error on the synthetic fixture belongs to DepthWizard, and how much belongs to
the pixel grid?

The answer is *none* and *all of it*. Rendered at an axis-aligned sun azimuth,
where the geometry is exactly representable on a square grid, the whole stack
-- render, detect, measure, invert -- returns the specified heights to machine
epsilon. Rendered at azimuth 135 degrees, where the shadow runs exactly along
the diagonal, the recoverable lengths are the multiples of sqrt(2) and the
residual is exactly the distance to the nearest one. Both are asserted below
against closed forms, so neither can drift unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from depthwizard.config import BuildingSpec, GroundConfig, RasterConfig, SceneConfig, SunConfig
from depthwizard.ingest.geotiff import read_raster
from depthwizard.ingest.synthetic import footprint_pixels, render_scene
from depthwizard.physics import Confidence
from depthwizard.shadows import (
    BuildingFootprint,
    BuildingHeightEstimate,
    ClassicalShadowDetector,
    DetectionContext,
    SceneHeightResult,
    ShadowHeightPipeline,
    estimate_heights_from_mask,
)

from conftest import build_analytic_shadow_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _footprints_from_scene(scene: SceneConfig, shape) -> list[BuildingFootprint]:
    """Footprints are SUPPLIED -- read from the scene spec, never detected."""
    footprints = []
    for building in scene.buildings:
        pixels = footprint_pixels(building, scene.raster)
        footprints.append(
            BuildingFootprint.from_bbox(
                building.name,
                shape=shape,
                row_min=pixels.row_min,
                row_max=pixels.row_max,
                col_min=pixels.col_min,
                col_max=pixels.col_max,
            )
        )
    return footprints


def _render_u8(scene: SceneConfig) -> np.ndarray:
    render = render_scene(scene)
    return np.clip(np.rint(render.reflectance * 255.0), 0, 255).astype(np.uint8)


def _run(scene: SceneConfig) -> SceneHeightResult:
    image = _render_u8(scene)
    footprints = _footprints_from_scene(scene, image.shape)
    context = DetectionContext(
        gsd_m=scene.raster.gsd_m,
        sun_azimuth_deg=scene.sun.azimuth_deg,
        sun_elevation_deg=scene.sun.elevation_deg,
    )
    return ShadowHeightPipeline().run(image, footprints, context)


#: Ground 0.35, shadow 0.08: a fixed threshold must sit strictly between them.
#: 0.35 itself would swallow the whole ground plane, since the cut is ``<=``.
FIXED_THRESHOLD = 0.20


def _centred_building_scene(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    height_m: float = 30.0,
    gsd_m: float = 0.5,
) -> SceneConfig:
    """One building alone in the middle of a raster with room on every side.

    The default fixture's four buildings are laid out for azimuth 135; swing the
    sun elsewhere, or drop it towards the horizon, and their shadows start
    running off the raster or across each other. Both are real effects worth
    handling, but neither is what a direction or elevation sweep is trying to
    measure, so those sweeps get a scene with margin to spare.
    """
    size_px = 400
    centre_m = size_px * gsd_m / 2.0
    footprint_m = 20.0
    return SceneConfig(
        name="centred",
        raster=RasterConfig(
            width_px=size_px,
            height_px=size_px,
            gsd_m=gsd_m,
            crs="EPSG:32643",
            origin_easting_m=700000.0,
            origin_northing_m=3170000.0,
        ),
        sun=SunConfig(elevation_deg=elevation_deg, azimuth_deg=azimuth_deg),
        ground=GroundConfig(),
        buildings=[
            BuildingSpec(
                name="solo",
                x_m=centre_m - footprint_m / 2.0,
                y_m=centre_m - footprint_m / 2.0,
                width_m=footprint_m,
                depth_m=footprint_m,
                height_m=height_m,
            )
        ],
    )


@pytest.fixture(scope="module")
def axis_aligned_scene(request) -> SceneConfig:
    """The default scene with the sun due East, so shadows run due West.

    Everything else is unchanged. At a cardinal azimuth the shadow advances one
    whole pixel per step, so a shadow length that is a whole number of pixels is
    exactly representable -- which is the case the fixture's own geometry gives.
    """
    scene = request.getfixturevalue("scene")
    return replace(scene, sun=SunConfig(elevation_deg=45.0, azimuth_deg=90.0))


# ---------------------------------------------------------------------------
# Exact recovery: the whole stack contributes no error of its own
# ---------------------------------------------------------------------------


def test_axis_aligned_fixture_recovers_every_height_to_machine_epsilon(axis_aligned_scene):
    """Render -> detect -> measure -> invert, with zero accumulated error.

    This is the tightest statement the project can make about its own geometry:
    on a scene whose shadows the grid can represent exactly, the specified
    heights come back to ~1e-12 m. Any sign error, off-by-one, or half-pixel
    convention slip anywhere in the chain would destroy this immediately.
    """
    result = _run(axis_aligned_scene)
    assert len(result) == len(axis_aligned_scene.buildings)
    for estimate, building in zip(result, axis_aligned_scene.buildings):
        assert estimate.building_id == building.name
        assert estimate.height_m is not None
        assert estimate.height_m == pytest.approx(building.height_m, abs=1e-9)


def test_axis_aligned_shadow_lengths_are_exact(axis_aligned_scene):
    """At 45 degrees elevation the shadow length equals the height in metres."""
    result = _run(axis_aligned_scene)
    for estimate, building in zip(result, axis_aligned_scene.buildings):
        assert estimate.shadow_length_m == pytest.approx(building.height_m, abs=1e-9)
        expected_px = building.height_m / axis_aligned_scene.raster.gsd_m
        assert estimate.shadow_length_px == pytest.approx(expected_px, abs=1e-9)


@pytest.mark.parametrize("azimuth_deg", [0.0, 90.0, 180.0, 270.0])
def test_every_cardinal_azimuth_recovers_exactly(azimuth_deg):
    """The result must not depend on which way the shadows happen to fall."""
    cardinal = _centred_building_scene(azimuth_deg=azimuth_deg, elevation_deg=45.0)
    result = _run(cardinal)
    for estimate, building in zip(result, cardinal.buildings):
        assert estimate.height_m == pytest.approx(building.height_m, abs=1e-9)


@pytest.mark.parametrize("elevation_deg", [25.0, 30.0, 35.0, 40.0, 45.0])
def test_recovery_is_exact_across_the_usable_elevation_band(elevation_deg):
    """h = L * tan(theta) must invert the render at every elevation, not just 45.

    Lower elevations stretch the shadow, so the pixel grid resolves the length
    more finely -- but tan(theta) shrinks, so the height error shrinks with it.
    The bound is half a rendered pixel of shadow, converted and amplified.
    """
    tilted = _centred_building_scene(azimuth_deg=90.0, elevation_deg=elevation_deg)
    result = _run(tilted)
    gsd_m = tilted.raster.gsd_m
    for estimate, building in zip(result, tilted.buildings):
        bound = 0.5 * gsd_m * math.tan(math.radians(elevation_deg)) + 1e-9
        assert abs(estimate.height_m - building.height_m) <= bound


# ---------------------------------------------------------------------------
# The committed 135 degree fixture: the residual is the diagonal grid step
# ---------------------------------------------------------------------------


def test_diagonal_fixture_residual_is_exactly_the_grid_quantization(fixture_tif, generated):
    """At azimuth 135 the recoverable lengths are the multiples of sqrt(2).

    Every ray is launched from a pixel centre and steps along the diagonal, so
    consecutive samples are sqrt(2) px apart and every ray shares the same
    phase. The fixture renders its shadow tip at the nearest whole-pixel
    translation, so the measurable length is the nearest multiple of sqrt(2) to
    the geometric one -- and that is asserted here exactly, not approximately.
    """
    image = read_raster(fixture_tif).array
    footprints = _footprints_from_scene(generated.scene, image.shape)
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=135.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline().run(image, footprints, context)

    diagonal_step = math.sqrt(2.0)
    by_id = {e.building_id: e for e in result}
    for building in generated.truth["buildings"]:
        estimate = by_id[building["name"]]
        geometric_px = building["expected_shadow_length_px"]
        representable_px = round(geometric_px / diagonal_step) * diagonal_step
        assert estimate.shadow_length_px == pytest.approx(representable_px, abs=1e-9)


def test_diagonal_fixture_height_error_is_within_the_grid_bound(fixture_tif, generated):
    """Half a diagonal step of shadow, converted to metres and amplified by tan.

    At 0.5 m pixels, 45 degrees elevation and azimuth 135 that is
    ``(sqrt(2) / 2) * 0.5 * tan(45) = 0.354 m``. Nothing in the implementation
    can do better on this scene; the bound is asserted rather than a round
    number chosen to fit.
    """
    image = read_raster(fixture_tif).array
    footprints = _footprints_from_scene(generated.scene, image.shape)
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=135.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline().run(image, footprints, context)

    gsd_m, elevation_deg = 0.5, 45.0
    bound_m = (math.sqrt(2.0) / 2.0) * gsd_m * math.tan(math.radians(elevation_deg))
    truth = {b["name"]: b["height_m"] for b in generated.truth["buildings"]}
    for estimate in result:
        error = abs(estimate.height_m - truth[estimate.building_id])
        assert error <= bound_m + 1e-9, f"{estimate.building_id}: {error:.4f} m > {bound_m:.4f} m"


def test_the_diagonal_residual_beats_the_axis_aligned_case_by_the_grid_alone(scene):
    """Same buildings, same physics: only the azimuth differs, and only the grid.

    Rotating the sun to a cardinal bearing removes the entire residual. That is
    the evidence that the error is sampling and not a modelling mistake.
    """
    diagonal = _run(replace(scene, sun=SunConfig(elevation_deg=45.0, azimuth_deg=135.0)))
    cardinal = _run(replace(scene, sun=SunConfig(elevation_deg=45.0, azimuth_deg=90.0)))
    truth = {b.name: b.height_m for b in scene.buildings}

    diagonal_error = max(abs(e.height_m - truth[e.building_id]) for e in diagonal)
    cardinal_error = max(abs(e.height_m - truth[e.building_id]) for e in cardinal)
    assert cardinal_error < 1e-9
    assert diagonal_error > 100 * cardinal_error


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------


def test_pipeline_result_carries_the_geometry_it_used(axis_aligned_scene):
    result = _run(axis_aligned_scene)
    assert result.sun_azimuth_deg == pytest.approx(90.0)
    assert result.shadow_azimuth_deg == pytest.approx(270.0)
    assert result.sun_elevation_deg == pytest.approx(45.0)
    assert result.gsd_m == pytest.approx(axis_aligned_scene.raster.gsd_m)
    assert result.shadow_mask.shape == (
        axis_aligned_scene.raster.height_px,
        axis_aligned_scene.raster.width_px,
    )


def test_pipeline_refuses_to_invent_a_scale(axis_aligned_scene):
    """Without a GSD there is no metric height, and none is guessed."""
    image = _render_u8(axis_aligned_scene)
    footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    context = DetectionContext(sun_azimuth_deg=90.0, sun_elevation_deg=45.0)
    with pytest.raises(ValueError, match="gsd_m"):
        ShadowHeightPipeline().run(image, footprints, context)


@pytest.mark.parametrize("missing", ["sun_azimuth_deg", "sun_elevation_deg"])
def test_pipeline_refuses_without_the_sun_angles(axis_aligned_scene, missing):
    image = _render_u8(axis_aligned_scene)
    footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    fields = {"gsd_m": 0.5, "sun_azimuth_deg": 90.0, "sun_elevation_deg": 45.0}
    fields[missing] = None
    with pytest.raises(ValueError, match=missing):
        ShadowHeightPipeline().run(image, footprints, DetectionContext(**fields))


def test_run_overrides_take_precedence_over_the_context(axis_aligned_scene):
    """A context built for detection only can still be given the geometry."""
    image = _render_u8(axis_aligned_scene)
    footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    result = ShadowHeightPipeline().run(
        image,
        footprints,
        DetectionContext(),
        gsd_m=axis_aligned_scene.raster.gsd_m,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
    )
    for estimate, building in zip(result, axis_aligned_scene.buildings):
        assert estimate.height_m == pytest.approx(building.height_m, abs=1e-9)


def test_pipeline_accepts_an_alternative_detector(axis_aligned_scene):
    """The detector is swappable; the rest of the pipeline does not change."""
    image = _render_u8(axis_aligned_scene)
    footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline(ClassicalShadowDetector(threshold=FIXED_THRESHOLD)).run(
        image, footprints, context
    )
    assert result.shadow_mask.method == "hsv_value_fixed"
    for estimate, building in zip(result, axis_aligned_scene.buildings):
        assert estimate.height_m == pytest.approx(building.height_m, abs=1e-9)


def test_estimate_heights_from_mask_bypasses_detection():
    """A hand-drawn or ground-truth mask must work without any detector."""
    shape = (200, 200)
    mask = build_analytic_shadow_mask(
        shape,
        row_min=80,
        row_max=120,
        col_min=80,
        col_max=120,
        sun_azimuth_deg=90.0,
        shadow_length_px=40.0,
    )
    footprint = BuildingFootprint.from_bbox(
        "b", shape=shape, row_min=80, row_max=120, col_min=80, col_max=120
    )
    estimates = estimate_heights_from_mask(
        [footprint], mask, gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    assert len(estimates) == 1
    assert estimates[0].height_m == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Confidence and failure
# ---------------------------------------------------------------------------


def test_in_band_scene_is_nominal(axis_aligned_scene):
    result = _run(axis_aligned_scene)
    assert all(e.confidence is Confidence.NOMINAL for e in result)
    assert all(e.reduced_confidence_reasons == () for e in result)


@pytest.mark.parametrize("elevation_deg", [15.0, 20.0, 60.0, 75.0])
def test_out_of_band_scene_is_reduced_but_still_measured(scene, elevation_deg):
    """The band gate flags; it does not stop a height being produced."""
    out_of_band = replace(scene, sun=SunConfig(elevation_deg=elevation_deg, azimuth_deg=90.0))
    result = _run(out_of_band)
    assert len(result) == len(out_of_band.buildings)
    for estimate in result:
        assert estimate.confidence is Confidence.REDUCED
        assert estimate.height_m is not None
        assert estimate.reduced_confidence_reasons


def test_a_building_with_no_shadow_fails_without_a_substituted_height():
    """A failed estimate carries None. There is never a fallback number."""
    shape = (120, 120)
    mask = np.zeros(shape, dtype=bool)
    footprint = BuildingFootprint.from_bbox(
        "unlit", shape=shape, row_min=40, row_max=80, col_min=40, col_max=80
    )
    estimate = estimate_heights_from_mask(
        [footprint], mask, gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )[0]
    assert estimate.confidence is Confidence.FAILED
    assert estimate.height_m is None
    assert estimate.height_uncertainty_m is None
    assert estimate.shadow_length_m is None
    assert estimate.failure_reason


def test_successful_filters_out_the_failures():
    shape = (200, 200)
    mask = build_analytic_shadow_mask(
        shape,
        row_min=40,
        row_max=70,
        col_min=120,
        col_max=150,
        sun_azimuth_deg=90.0,
        shadow_length_px=30.0,
    )
    good = BuildingFootprint.from_bbox(
        "good", shape=shape, row_min=40, row_max=70, col_min=120, col_max=150
    )
    bad = BuildingFootprint.from_bbox(
        "bad", shape=shape, row_min=150, row_max=180, col_min=20, col_max=50
    )
    estimates = estimate_heights_from_mask(
        [good, bad], mask, gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    result = SceneHeightResult(
        estimates=tuple(estimates),
        shadow_mask=ClassicalShadowDetector().detect(
            np.zeros(shape, dtype=np.uint8), DetectionContext()
        ),
        context=DetectionContext(),
        sun_elevation_deg=45.0,
        sun_azimuth_deg=90.0,
        shadow_azimuth_deg=270.0,
        gsd_m=0.5,
    )
    assert len(result) == 2
    assert [e.building_id for e in result.successful] == ["good"]


# ---------------------------------------------------------------------------
# Uncertainty through the pipeline
# ---------------------------------------------------------------------------


def test_supplied_pixel_uncertainty_becomes_a_height_uncertainty(axis_aligned_scene):
    """dL_px -> dL_m -> dh = tan(theta) * dL_m, with one GSD throughout."""
    image = _render_u8(axis_aligned_scene)
    footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline(measurement_uncertainty_px=1.0).run(
        image, footprints, context
    )
    for estimate in result:
        # 1 px * 0.5 m/px * tan(45) = 0.5 m
        assert estimate.height_uncertainty_m == pytest.approx(0.5)


def test_no_uncertainty_supplied_means_none_reported(axis_aligned_scene):
    result = _run(axis_aligned_scene)
    assert all(e.height_uncertainty_m is None for e in result)


def test_estimate_to_dict_is_complete(axis_aligned_scene):
    estimate = _run(axis_aligned_scene).estimates[0]
    record = estimate.to_dict()
    for key in (
        "building_id",
        "confidence_flag",
        "shadow_length_px",
        "shadow_length_m",
        "height_m",
        "tan_theta",
        "equation",
        "sun_band",
    ):
        assert key in record
    assert isinstance(estimate, BuildingHeightEstimate)


def test_scene_result_to_dict_carries_detector_and_buildings(axis_aligned_scene):
    record = _run(axis_aligned_scene).to_dict()
    assert record["sun_azimuth_deg"] == pytest.approx(90.0)
    assert record["shadow_azimuth_deg"] == pytest.approx(270.0)
    assert "detector" in record
    assert len(record["buildings"]) == len(axis_aligned_scene.buildings)


# ---------------------------------------------------------------------------
# Footprints stay supplied inputs
# ---------------------------------------------------------------------------


def test_the_pipeline_never_invents_a_footprint(axis_aligned_scene):
    """One estimate per supplied footprint. No more, no fewer, same ids."""
    image = _render_u8(axis_aligned_scene)
    all_footprints = _footprints_from_scene(axis_aligned_scene, image.shape)
    subset = all_footprints[:2]
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline().run(image, subset, context)
    assert len(result) == 2
    assert [e.building_id for e in result] == [f.building_id for f in subset]


def test_no_footprints_means_no_estimates(axis_aligned_scene):
    image = _render_u8(axis_aligned_scene)
    context = DetectionContext(gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0)
    result = ShadowHeightPipeline().run(image, [], context)
    assert len(result) == 0
    assert result.successful == ()
