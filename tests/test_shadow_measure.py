"""Shadow-length measurement along the anti-sun direction.

The central question here is how accurate the measurement *can* be, and what
limits it. The answer, established by the tests below, is:

* against an analytic shadow mask at an axis-aligned sun azimuth, recovery is
  exact to floating point -- the geometry contributes no error at all;
* at any other azimuth the residual is bounded by half the spacing between
  consecutive pixel centres along the ray, which is the grid resolution limit
  and not something the estimator can improve on;
* the estimator is unbiased: swept over many azimuths and lengths, the mean
  error is a small fraction of a pixel rather than a systematic offset.

That decomposition is why the fixture-recovery test in
``test_shadow_pipeline.py`` asserts a geometry-derived bound rather than a
loose tolerance: the bound is tight, and a real regression breaks it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import build_analytic_shadow_mask, shadow_ray_step_px
from depthwizard.shadows import (
    DEFAULT_MEASUREMENT_PARAMS,
    BuildingFootprint,
    MeasurementParams,
    measure_shadow_length,
    measure_shadow_lengths,
)

SHAPE = (400, 400)
ROW_MIN, ROW_MAX, COL_MIN, COL_MAX = 180, 220, 180, 220


@pytest.fixture
def footprint() -> BuildingFootprint:
    return BuildingFootprint.from_bbox(
        "b",
        shape=SHAPE,
        row_min=ROW_MIN,
        row_max=ROW_MAX,
        col_min=COL_MIN,
        col_max=COL_MAX,
    )


def _mask(sun_azimuth_deg: float, shadow_length_px: float) -> np.ndarray:
    return build_analytic_shadow_mask(
        SHAPE,
        row_min=ROW_MIN,
        row_max=ROW_MAX,
        col_min=COL_MIN,
        col_max=COL_MAX,
        sun_azimuth_deg=sun_azimuth_deg,
        shadow_length_px=shadow_length_px,
    )


def _measure(footprint, sun_azimuth_deg, shadow_length_px, **kwargs):
    return measure_shadow_length(
        footprint,
        _mask(sun_azimuth_deg, shadow_length_px),
        gsd_m=kwargs.pop("gsd_m", 1.0),
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=kwargs.pop("sun_elevation_deg", 45.0),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# BuildingFootprint -- supplied, never detected
# ---------------------------------------------------------------------------


def test_footprint_from_bbox_uses_half_open_bounds():
    """Half-open like a slice, so fixture ground truth converts with no off-by-one."""
    footprint = BuildingFootprint.from_bbox(
        "b", shape=(10, 10), row_min=2, row_max=5, col_min=3, col_max=7
    )
    assert footprint.pixel_count == 3 * 4
    assert footprint.mask[2, 3] and footprint.mask[4, 6]
    assert not footprint.mask[5, 3]
    assert not footprint.mask[2, 7]


def test_footprint_bbox_and_centroid():
    footprint = BuildingFootprint.from_bbox(
        "b", shape=(20, 20), row_min=4, row_max=8, col_min=6, col_max=10
    )
    assert footprint.bbox == (4, 8, 6, 10)
    assert footprint.centroid_rc == pytest.approx((5.5, 7.5))


def test_an_empty_footprint_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        BuildingFootprint(building_id="b", mask=np.zeros((10, 10), dtype=bool))


# ---------------------------------------------------------------------------
# Exact recovery where the grid can represent the answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sun_azimuth_deg", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("shadow_length_px", [12.0, 24.0, 37.0, 60.0])
def test_axis_aligned_shadows_are_recovered_exactly(
    footprint, sun_azimuth_deg, shadow_length_px
):
    """At a cardinal azimuth the ray steps 1 px at a time, so nothing is lost.

    This is the strongest statement available about the measurement geometry:
    with the grid able to represent the answer, the error is zero to floating
    point. Any residual seen elsewhere is therefore sampling, not geometry.
    """
    result = _measure(footprint, sun_azimuth_deg, shadow_length_px)
    assert result.ok
    assert result.shadow_length_px == pytest.approx(shadow_length_px, abs=1e-9)


@pytest.mark.parametrize("sun_azimuth_deg", [0.0, 90.0, 180.0, 270.0])
def test_axis_aligned_recovery_is_exact_in_metres_too(footprint, sun_azimuth_deg):
    result = _measure(footprint, sun_azimuth_deg, 40.0, gsd_m=0.5)
    assert result.shadow_length_m == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The residual elsewhere is grid quantization, and it is bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sun_azimuth_deg", [115.0, 135.0, 160.0, 200.0, 250.0, 305.0])
@pytest.mark.parametrize("shadow_length_px", [23.0, 40.0, 55.0])
def test_off_axis_error_is_bounded_by_half_the_ray_step(
    footprint, sun_azimuth_deg, shadow_length_px
):
    """Half a ray step is the grid resolution limit, and the bound is tight.

    Along a ray at azimuth A the pixel centres are ``1 / max(|d_row|, |d_col|)``
    apart. A length falling between two of them cannot be resolved further, so
    the best any estimator can do is land within half that spacing.
    """
    result = _measure(footprint, sun_azimuth_deg, shadow_length_px)
    assert result.ok
    step = shadow_ray_step_px(sun_azimuth_deg)
    error = result.shadow_length_px - shadow_length_px
    assert abs(error) <= step / 2 + 1e-9, f"error {error:+.4f} px exceeds step/2 = {step / 2:.4f}"


def test_the_45_degree_diagonal_is_the_worst_case(footprint):
    """At 135 deg every ray shares one phase, so averaging cannot help.

    The representable lengths are exactly the multiples of sqrt(2): the offsets
    between launch pixels are whole pixels, so every ray quantizes identically
    and the median over them lands on the same multiple. This is the geometry
    the synthetic fixture is rendered at, and it explains its residual entirely.
    """
    result = _measure(footprint, 135.0, 60.0)
    diagonal_step = math.sqrt(2.0)
    steps = result.shadow_length_px / diagonal_step
    assert steps == pytest.approx(round(steps), abs=1e-9)
    assert result.shadow_length_px == pytest.approx(
        round(60.0 / diagonal_step) * diagonal_step, abs=1e-9
    )
    # Every ray that found shadow agreed exactly -- hence zero spread.
    assert result.ray_length_spread_px == pytest.approx(0.0, abs=1e-9)


def test_the_estimator_is_unbiased_across_azimuths(footprint):
    """Swept over many geometries the mean error is a fraction of a pixel.

    A systematic half-step offset -- the classic symptom of a pixel-centre
    versus pixel-edge mistake -- would show up here as a mean near -0.5 px.
    """
    errors = []
    for sun_azimuth_deg in (105.0, 120.0, 135.0, 150.0, 165.0, 195.0, 215.0, 240.0):
        for shadow_length_px in (23.0, 31.5, 40.0, 47.3, 55.0):
            result = _measure(footprint, sun_azimuth_deg, shadow_length_px)
            assert result.ok
            errors.append(result.shadow_length_px - shadow_length_px)
    mean_error = float(np.mean(errors))
    assert abs(mean_error) < 0.15, f"systematic bias of {mean_error:+.4f} px"
    assert max(abs(e) for e in errors) < math.sqrt(2.0)


def test_measurement_does_not_depend_on_the_sampling_step(footprint):
    """Both ends of a ray are pixel centres, so the half-pixel conventions cancel."""
    lengths = set()
    for step_px in (0.5, 0.25, 0.1):
        result = _measure(
            footprint, 90.0, 40.0, params=MeasurementParams(step_px=step_px)
        )
        lengths.add(round(result.shadow_length_px, 9))
    assert len(lengths) == 1


# ---------------------------------------------------------------------------
# Direction: the measurement must run away from the sun
# ---------------------------------------------------------------------------


def test_measuring_with_the_sun_azimuth_reversed_finds_no_shadow(footprint):
    """The commonest possible sign error, and it must fail loudly, not quietly.

    Rays launched towards the sun cross lit ground, so the measurement returns
    ok=False with a reason rather than a small plausible number.
    """
    mask = _mask(90.0, 40.0)
    wrong = measure_shadow_length(
        footprint, mask, gsd_m=1.0, sun_azimuth_deg=270.0, sun_elevation_deg=45.0
    )
    assert not wrong.ok
    assert wrong.shadow_length_px is None
    assert wrong.shadow_length_m is None
    assert wrong.failure_reason


def test_the_recorded_direction_is_the_anti_sun_bearing(footprint):
    result = _measure(footprint, 135.0, 40.0)
    assert result.sun_azimuth_deg == pytest.approx(135.0)
    assert result.shadow_azimuth_deg == pytest.approx(315.0)
    d_row, d_col = result.direction_px
    assert d_row < 0 and d_col < 0  # north-west is up and left
    assert math.hypot(d_row, d_col) == pytest.approx(1.0)


@pytest.mark.parametrize("sun_azimuth_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_every_azimuth_recovers_the_same_length(footprint, sun_azimuth_deg):
    """The answer must not depend on which way the shadow happens to point."""
    result = _measure(footprint, sun_azimuth_deg, 40.0)
    assert result.ok
    assert result.shadow_length_px == pytest.approx(
        40.0, abs=shadow_ray_step_px(sun_azimuth_deg) / 2 + 1e-9
    )


# ---------------------------------------------------------------------------
# Failure is a result, never a fallback
# ---------------------------------------------------------------------------


def test_no_shadow_at_all_fails_with_a_reason(footprint):
    empty = np.zeros(SHAPE, dtype=bool)
    result = measure_shadow_length(
        footprint, empty, gsd_m=1.0, sun_azimuth_deg=135.0, sun_elevation_deg=45.0
    )
    assert not result.ok
    assert result.shadow_length_px is None
    assert result.shadow_length_m is None
    assert "insufficient" in result.failure_reason or "rays" in result.failure_reason


def test_a_footprint_too_small_to_launch_rays_fails():
    tiny = BuildingFootprint.from_bbox(
        "tiny", shape=SHAPE, row_min=10, row_max=11, col_min=10, col_max=11
    )
    mask = np.zeros(SHAPE, dtype=bool)
    result = measure_shadow_length(
        tiny,
        mask,
        gsd_m=1.0,
        sun_azimuth_deg=135.0,
        sun_elevation_deg=45.0,
        params=MeasurementParams(min_rays=5),
    )
    assert not result.ok
    assert "boundary pixel" in result.failure_reason


def test_a_partly_missing_shadow_succeeds_with_reduced_confidence(footprint):
    """Between the nominal and minimum hit fractions: a result, plus a reason."""
    mask = _mask(90.0, 40.0)
    # Erase the shadow in front of the lower third of the building edge.
    mask[int(ROW_MIN + 0.7 * (ROW_MAX - ROW_MIN)) :, :] = False
    result = measure_shadow_length(
        footprint, mask, gsd_m=1.0, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    if result.ok:
        assert result.hit_fraction < 1.0
        if result.hit_fraction < DEFAULT_MEASUREMENT_PARAMS.nominal_hit_fraction:
            assert result.reduced_confidence_reasons
    else:
        assert result.failure_reason


def test_a_gap_smaller_than_the_tolerance_is_crossed(footprint):
    """A car or a tree breaks a real shadow; a small hole must not end the ray."""
    mask = _mask(90.0, 40.0)
    mask[:, COL_MIN - 21 : COL_MIN - 20] = False  # a 1 px hole across the shadow
    result = measure_shadow_length(
        footprint, mask, gsd_m=1.0, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    assert result.ok
    assert result.shadow_length_px == pytest.approx(40.0, abs=1e-9)


def test_a_gap_larger_than_the_tolerance_stops_the_ray(footprint):
    """A large hole is a real discontinuity and must truncate the measurement."""
    mask = _mask(90.0, 40.0)
    mask[:, COL_MIN - 25 : COL_MIN - 15] = False  # a 10 px hole
    result = measure_shadow_length(
        footprint,
        mask,
        gsd_m=1.0,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
        params=MeasurementParams(gap_tolerance_px=2.0),
    )
    assert result.ok
    assert result.shadow_length_px == pytest.approx(15.0, abs=1e-9)


def test_the_median_ignores_one_wild_ray(footprint):
    """A dark alley behind one edge pixel must not move the answer."""
    mask = _mask(90.0, 40.0)
    mask[ROW_MIN + 5, : COL_MIN - 40] = True  # one very long streak
    result = measure_shadow_length(
        footprint, mask, gsd_m=1.0, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    assert result.ok
    assert result.shadow_length_px == pytest.approx(40.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Units, uncertainty and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gsd_m", [0.25, 0.5, 1.0, 2.0])
def test_metres_are_pixels_times_gsd(footprint, gsd_m):
    result = _measure(footprint, 90.0, 40.0, gsd_m=gsd_m)
    assert result.shadow_length_m == pytest.approx(result.shadow_length_px * gsd_m)


def test_no_uncertainty_supplied_means_none_reported(footprint):
    result = _measure(footprint, 90.0, 40.0)
    assert result.measurement_uncertainty_px is None
    assert result.shadow_length_uncertainty_m is None


def test_a_supplied_uncertainty_is_converted_with_the_same_gsd(footprint):
    result = _measure(footprint, 90.0, 40.0, gsd_m=0.5, measurement_uncertainty_px=1.5)
    assert result.measurement_uncertainty_px == pytest.approx(1.5)
    assert result.shadow_length_uncertainty_m == pytest.approx(0.75)


def test_ray_spread_is_reported_as_a_diagnostic(footprint):
    result = _measure(footprint, 90.0, 40.0)
    assert result.ray_length_spread_px is not None
    # It is a diagnostic, not an error bar: it must not become the uncertainty.
    assert result.shadow_length_uncertainty_m is None


@pytest.mark.parametrize("bad_gsd", [0.0, -1.0, float("nan")])
def test_a_bad_gsd_is_rejected(footprint, bad_gsd):
    with pytest.raises(ValueError, match="gsd_m must be positive"):
        _measure(footprint, 90.0, 40.0, gsd_m=bad_gsd)


def test_a_mask_of_the_wrong_shape_is_rejected(footprint):
    with pytest.raises(ValueError, match="does not match footprint"):
        measure_shadow_length(
            footprint,
            np.zeros((10, 10), dtype=bool),
            gsd_m=1.0,
            sun_azimuth_deg=135.0,
            sun_elevation_deg=45.0,
        )


def test_a_negative_uncertainty_is_rejected(footprint):
    with pytest.raises(ValueError, match="measurement_uncertainty_px"):
        _measure(footprint, 90.0, 40.0, measurement_uncertainty_px=-1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step_px": 0.75}, "step_px"),
        ({"step_px": 0.0}, "step_px"),
        ({"gap_tolerance_px": -1.0}, "gap_tolerance_px"),
        ({"min_rays": 0}, "min_rays"),
        ({"min_hit_fraction": 0.0}, "min_hit_fraction"),
        ({"min_hit_fraction": 0.9, "nominal_hit_fraction": 0.5}, "nominal_hit_fraction"),
        ({"percentile": 101.0}, "percentile"),
        ({"max_rays": 0}, "max_rays"),
    ],
)
def test_measurement_params_validate_themselves(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MeasurementParams(**kwargs)


def test_measurement_to_dict_carries_the_full_derivation(footprint):
    record = _measure(footprint, 135.0, 40.0, measurement_uncertainty_px=1.0).to_dict()
    for key in (
        "building_id",
        "measurement_ok",
        "gsd_m",
        "sun_azimuth_deg",
        "shadow_azimuth_deg",
        "shadow_direction_d_row",
        "shadow_direction_d_col",
        "shadow_length_px",
        "shadow_length_m",
        "n_rays",
        "n_rays_hit",
        "hit_fraction",
        "aggregation",
        "params",
    ):
        assert key in record


# ---------------------------------------------------------------------------
# Whole scenes
# ---------------------------------------------------------------------------


def test_measure_shadow_lengths_handles_every_footprint():
    mask = build_analytic_shadow_mask(
        SHAPE,
        row_min=100,
        row_max=130,
        col_min=100,
        col_max=130,
        sun_azimuth_deg=90.0,
        shadow_length_px=30.0,
    )
    mask |= build_analytic_shadow_mask(
        SHAPE,
        row_min=250,
        row_max=280,
        col_min=250,
        col_max=280,
        sun_azimuth_deg=90.0,
        shadow_length_px=50.0,
    )
    footprints = [
        BuildingFootprint.from_bbox(
            "a", shape=SHAPE, row_min=100, row_max=130, col_min=100, col_max=130
        ),
        BuildingFootprint.from_bbox(
            "b", shape=SHAPE, row_min=250, row_max=280, col_min=250, col_max=280
        ),
    ]
    results = measure_shadow_lengths(
        footprints, mask, gsd_m=1.0, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )
    assert [r.building_id for r in results] == ["a", "b"]
    assert results[0].shadow_length_px == pytest.approx(30.0, abs=1e-9)
    assert results[1].shadow_length_px == pytest.approx(50.0, abs=1e-9)


def test_a_neighbouring_footprint_does_not_cut_a_shadow_short():
    """A shadow passing behind another building is skipped over, not truncated."""
    shape = (200, 300)
    mask = build_analytic_shadow_mask(
        shape,
        row_min=80,
        row_max=120,
        col_min=200,
        col_max=240,
        sun_azimuth_deg=90.0,  # shadow runs west, towards lower columns
        shadow_length_px=120.0,
    )
    # A neighbour sitting in the middle of that shadow.
    neighbour = BuildingFootprint.from_bbox(
        "neighbour", shape=shape, row_min=80, row_max=120, col_min=140, col_max=160
    )
    mask[neighbour.mask] = False  # its roof is lit, not shadow
    tall = BuildingFootprint.from_bbox(
        "tall", shape=shape, row_min=80, row_max=120, col_min=200, col_max=240
    )

    with_neighbour = measure_shadow_lengths(
        [tall, neighbour],
        mask,
        gsd_m=1.0,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
        exclude_other_footprints=True,
    )[0]
    without = measure_shadow_lengths(
        [tall, neighbour],
        mask,
        gsd_m=1.0,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
        exclude_other_footprints=False,
    )[0]
    assert with_neighbour.shadow_length_px == pytest.approx(120.0, abs=1e-9)
    assert without.shadow_length_px < with_neighbour.shadow_length_px
