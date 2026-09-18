"""The height equation, the uncertainty equation, and the sun-elevation gate.

Three separate contracts are pinned here:

* ``h = L * tan(theta)`` -- checked against closed forms, not against itself.
* ``dh = tan(theta) * dL`` -- exact, because h is linear in L at fixed theta.
* the 25-45 degree band -- a *confidence* gate. It must never reject a scene
  and never change a height; it may only attach a flag and a stated reason.
"""

from __future__ import annotations

import math

import pytest

from depthwizard.physics import (
    USABLE_SUN_ELEVATION_MAX_DEG,
    USABLE_SUN_ELEVATION_MIN_DEG,
    Confidence,
    HeightEstimate,
    SunBand,
    height_from_shadow,
    height_uncertainty_m,
    sun_band_status,
)


# ---------------------------------------------------------------------------
# h = L * tan(theta)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length_m", "elevation_deg", "expected_h"),
    [
        (30.0, 45.0, 30.0),  # tan(45) = 1
        (12.0, 45.0, 12.0),
        (0.0, 45.0, 0.0),  # a zero-length shadow is a zero height, not a failure
        (100.0, 30.0, 100.0 * math.tan(math.radians(30.0))),
        (50.0, 60.0, 50.0 * math.tan(math.radians(60.0))),
        (20.0, 25.0, 20.0 * math.tan(math.radians(25.0))),
    ],
)
def test_height_equation(length_m, elevation_deg, expected_h):
    estimate = height_from_shadow(length_m, elevation_deg)
    assert estimate.height_m == pytest.approx(expected_h, rel=1e-12)


def test_height_scales_linearly_with_shadow_length():
    """Doubling the measured shadow doubles the height, at fixed elevation."""
    single = height_from_shadow(20.0, 35.0).height_m
    double = height_from_shadow(40.0, 35.0).height_m
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_estimate_carries_every_number_behind_the_answer():
    """The arithmetic must be re-doable by hand from the returned object."""
    estimate = height_from_shadow(24.0, 35.0, shadow_length_uncertainty_m=0.5)
    assert isinstance(estimate, HeightEstimate)
    assert estimate.shadow_length_m == pytest.approx(24.0)
    assert estimate.sun_elevation_deg == pytest.approx(35.0)
    assert estimate.tan_theta == pytest.approx(math.tan(math.radians(35.0)))
    # The stated equation is the one that was actually evaluated.
    assert estimate.height_m == pytest.approx(estimate.shadow_length_m * estimate.tan_theta)
    assert "h = L * tan(theta)" in estimate.equation


def test_to_dict_exposes_the_equation_and_the_band():
    record = height_from_shadow(24.0, 35.0, shadow_length_uncertainty_m=0.5).to_dict()
    for key in (
        "shadow_length_m",
        "sun_elevation_deg",
        "tan_theta",
        "height_m",
        "height_uncertainty_m",
        "confidence_flag",
        "equation",
        "sun_band",
        "sun_band_in_band",
    ):
        assert key in record


@pytest.mark.parametrize("bad_length", [-0.1, -30.0])
def test_height_rejects_a_negative_shadow_length(bad_length):
    with pytest.raises(ValueError, match="shadow_length_m must be >= 0"):
        height_from_shadow(bad_length, 45.0)


@pytest.mark.parametrize("bad_elevation", [0.0, -5.0, 90.1, 180.0])
def test_height_rejects_an_out_of_range_elevation(bad_elevation):
    """(0, 90] only: at or below the horizon the shadow is undefined."""
    with pytest.raises(ValueError):
        height_from_shadow(30.0, bad_elevation)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_height_rejects_non_finite_input(bad):
    with pytest.raises(ValueError):
        height_from_shadow(bad, 45.0)
    with pytest.raises(ValueError):
        height_from_shadow(30.0, bad)


def test_elevation_of_exactly_90_degrees_is_allowed():
    """The sun overhead is legal; it just casts no shadow worth measuring."""
    estimate = height_from_shadow(0.0, 90.0)
    assert estimate.height_m == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# dh = tan(theta) * dL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("elevation_deg", "d_length_m"),
    [(45.0, 1.0), (30.0, 0.5), (25.0, 0.25), (60.0, 2.0), (15.0, 1.5), (80.0, 0.1)],
)
def test_uncertainty_equation(elevation_deg, d_length_m):
    expected = math.tan(math.radians(elevation_deg)) * d_length_m
    assert height_uncertainty_m(elevation_deg, d_length_m) == pytest.approx(expected, rel=1e-12)


def test_uncertainty_at_45_degrees_passes_straight_through():
    """tan(45) == 1, so a metre of shadow error is a metre of height error."""
    assert height_uncertainty_m(45.0, 0.7) == pytest.approx(0.7)


def test_uncertainty_grows_with_sun_elevation():
    """The reason for the 45 deg ceiling: tan amplifies error as the sun rises."""
    d_length_m = 0.5
    previous = 0.0
    for elevation_deg in (10.0, 25.0, 45.0, 60.0, 75.0, 85.0):
        current = height_uncertainty_m(elevation_deg, d_length_m)
        assert current > previous
        previous = current


def test_uncertainty_is_the_exact_derivative_not_a_linearisation():
    """h is linear in L at fixed theta, so dh/dL is exactly tan(theta)."""
    elevation_deg = 37.0
    base = height_from_shadow(20.0, elevation_deg).height_m
    perturbed = height_from_shadow(20.0 + 1e-6, elevation_deg).height_m
    assert (perturbed - base) / 1e-6 == pytest.approx(
        height_uncertainty_m(elevation_deg, 1.0), rel=1e-6
    )


def test_zero_uncertainty_in_gives_zero_uncertainty_out():
    assert height_uncertainty_m(45.0, 0.0) == pytest.approx(0.0)


def test_uncertainty_rejects_a_negative_dl():
    with pytest.raises(ValueError, match="must be >= 0"):
        height_uncertainty_m(45.0, -0.5)


def test_no_uncertainty_supplied_means_none_reported():
    """Nothing invents a default error bar."""
    estimate = height_from_shadow(30.0, 45.0)
    assert estimate.shadow_length_uncertainty_m is None
    assert estimate.height_uncertainty_m is None


def test_uncertainty_supplied_is_propagated_onto_the_estimate():
    estimate = height_from_shadow(30.0, 45.0, shadow_length_uncertainty_m=0.354)
    assert estimate.shadow_length_uncertainty_m == pytest.approx(0.354)
    assert estimate.height_uncertainty_m == pytest.approx(
        math.tan(math.radians(45.0)) * 0.354
    )
    # At 45 degrees the two are equal, which is the cheapest possible check.
    assert estimate.height_uncertainty_m == pytest.approx(0.354)


def test_uncertainty_on_the_estimate_matches_the_standalone_function():
    for elevation_deg in (20.0, 35.0, 50.0, 70.0):
        estimate = height_from_shadow(25.0, elevation_deg, shadow_length_uncertainty_m=0.4)
        assert estimate.height_uncertainty_m == pytest.approx(
            height_uncertainty_m(elevation_deg, 0.4)
        )


# ---------------------------------------------------------------------------
# The 25-45 degree sun band
# ---------------------------------------------------------------------------


def test_band_edges_are_the_documented_values():
    assert USABLE_SUN_ELEVATION_MIN_DEG == 25.0
    assert USABLE_SUN_ELEVATION_MAX_DEG == 45.0


@pytest.mark.parametrize("elevation_deg", [25.0, 25.001, 30.0, 35.0, 44.999, 45.0])
def test_inside_the_band_is_within_and_has_no_reason(elevation_deg):
    """Both edges are inclusive."""
    status = sun_band_status(elevation_deg)
    assert status.band is SunBand.WITHIN
    assert status.in_band is True
    assert status.reason is None


@pytest.mark.parametrize("elevation_deg", [0.5, 5.0, 15.0, 24.999])
def test_below_the_band_is_flagged_with_a_stated_reason(elevation_deg):
    status = sun_band_status(elevation_deg)
    assert status.band is SunBand.BELOW
    assert status.in_band is False
    assert status.reason is not None
    assert "below" in status.reason


@pytest.mark.parametrize("elevation_deg", [45.001, 50.0, 70.0, 90.0])
def test_above_the_band_is_flagged_with_a_stated_reason(elevation_deg):
    status = sun_band_status(elevation_deg)
    assert status.band is SunBand.ABOVE
    assert status.in_band is False
    assert status.reason is not None
    assert "above" in status.reason


def test_band_status_records_the_edges_it_used():
    status = sun_band_status(35.0, min_deg=20.0, max_deg=50.0)
    assert status.min_deg == 20.0
    assert status.max_deg == 50.0
    assert status.band is SunBand.WITHIN


def test_custom_band_edges_move_the_gate():
    """A scene in the default band can be out of a narrower custom one."""
    assert sun_band_status(40.0).band is SunBand.WITHIN
    assert sun_band_status(40.0, min_deg=25.0, max_deg=35.0).band is SunBand.ABOVE


def test_band_rejects_inverted_edges():
    with pytest.raises(ValueError, match="min_deg must be < max_deg"):
        sun_band_status(35.0, min_deg=45.0, max_deg=25.0)


def test_band_status_to_dict_is_self_describing():
    record = sun_band_status(12.0).to_dict()
    assert record["sun_elevation_deg"] == pytest.approx(12.0)
    assert record["sun_band"] == "below_band"
    assert record["sun_band_in_band"] is False
    assert record["sun_band_min_deg"] == 25.0
    assert record["sun_band_max_deg"] == 45.0
    assert record["sun_band_reason"]


# --- the gate downgrades confidence; it never rejects and never corrects ----


@pytest.mark.parametrize("elevation_deg", [25.0, 30.0, 35.0, 45.0])
def test_in_band_estimate_is_nominal(elevation_deg):
    estimate = height_from_shadow(20.0, elevation_deg)
    assert estimate.confidence is Confidence.NOMINAL
    assert estimate.reduced_confidence_reasons == ()


@pytest.mark.parametrize("elevation_deg", [10.0, 20.0, 24.9, 45.1, 60.0, 85.0])
def test_out_of_band_estimate_is_reduced_but_still_produced(elevation_deg):
    """Out of band is not a rejection: a height still comes back, flagged."""
    estimate = height_from_shadow(20.0, elevation_deg)
    assert estimate.confidence is Confidence.REDUCED
    assert len(estimate.reduced_confidence_reasons) >= 1
    assert estimate.height_m is not None
    assert estimate.height_m > 0.0


@pytest.mark.parametrize("elevation_deg", [10.0, 35.0, 60.0])
def test_the_band_gate_never_alters_the_height(elevation_deg):
    """The flag is metadata. The physics is identical either side of the edge."""
    estimate = height_from_shadow(20.0, elevation_deg)
    assert estimate.height_m == pytest.approx(20.0 * math.tan(math.radians(elevation_deg)))


def test_nothing_is_downgraded_anonymously():
    """Every reduced-confidence estimate lists at least one reason."""
    for elevation_deg in (5.0, 15.0, 50.0, 80.0):
        estimate = height_from_shadow(20.0, elevation_deg)
        assert estimate.confidence is Confidence.REDUCED
        assert all(reason.strip() for reason in estimate.reduced_confidence_reasons)


def test_upstream_reasons_are_carried_through_and_downgrade_an_in_band_estimate():
    """A measurement-stage complaint reduces confidence even at 35 degrees."""
    estimate = height_from_shadow(
        20.0, 35.0, extra_reduced_confidence_reasons=["only 6/10 rays found shadow"]
    )
    assert estimate.confidence is Confidence.REDUCED
    assert "only 6/10 rays found shadow" in estimate.reduced_confidence_reasons


def test_upstream_and_band_reasons_accumulate():
    estimate = height_from_shadow(
        20.0, 70.0, extra_reduced_confidence_reasons=["partially occluded"]
    )
    assert estimate.confidence is Confidence.REDUCED
    assert len(estimate.reduced_confidence_reasons) == 2
    assert "partially occluded" in estimate.reduced_confidence_reasons


def test_custom_band_edges_reach_the_estimate():
    estimate = height_from_shadow(20.0, 40.0, sun_band_min_deg=25.0, sun_band_max_deg=35.0)
    assert estimate.confidence is Confidence.REDUCED
    assert estimate.sun_band.band is SunBand.ABOVE
