"""Tests for the sun/shadow geometry in depthwizard.physics.sun."""

from __future__ import annotations

import math

import pytest

from depthwizard.physics.sun import (
    height_from_shadow_length_m,
    shadow_direction_unit,
    shadow_length_m,
    shadow_pixel_offset,
)


def test_shadow_length_at_45_degrees_equals_height() -> None:
    """tan(45) == 1, so shadow length equals height exactly."""
    assert shadow_length_m(30.0, 45.0) == pytest.approx(30.0)


def test_overhead_sun_casts_no_shadow() -> None:
    assert shadow_length_m(30.0, 90.0) == pytest.approx(0.0)


def test_low_sun_casts_long_shadow() -> None:
    """At 30 deg elevation a 10 m object casts 10/tan(30) = 17.32 m."""
    assert shadow_length_m(10.0, 30.0) == pytest.approx(10.0 / math.tan(math.radians(30.0)))
    assert shadow_length_m(10.0, 30.0) > shadow_length_m(10.0, 60.0)


@pytest.mark.parametrize("height", [1.0, 6.0, 12.5, 30.0, 120.0])
@pytest.mark.parametrize("elevation", [15.0, 30.0, 45.0, 62.5, 89.0])
def test_height_and_shadow_length_round_trip(height: float, elevation: float) -> None:
    length = shadow_length_m(height, elevation)
    assert height_from_shadow_length_m(length, elevation) == pytest.approx(height)


@pytest.mark.parametrize("elevation", [0.0, -1.0, 90.1])
def test_invalid_elevation_raises(elevation: float) -> None:
    with pytest.raises(ValueError, match="sun_elevation_deg"):
        shadow_length_m(10.0, elevation)


def test_negative_height_raises() -> None:
    with pytest.raises(ValueError, match="height_m"):
        shadow_length_m(-1.0, 45.0)


@pytest.mark.parametrize(
    "sun_azimuth_deg, expected_east, expected_north",
    [
        (0.0, 0.0, -1.0),  # sun in the North  -> shadow to the South
        (90.0, -1.0, 0.0),  # sun in the East   -> shadow to the West
        (180.0, 0.0, 1.0),  # sun in the South  -> shadow to the North
        (270.0, 1.0, 0.0),  # sun in the West   -> shadow to the East
    ],
)
def test_shadow_points_away_from_the_sun(
    sun_azimuth_deg: float, expected_east: float, expected_north: float
) -> None:
    east, north = shadow_direction_unit(sun_azimuth_deg)
    assert east == pytest.approx(expected_east, abs=1e-9)
    assert north == pytest.approx(expected_north, abs=1e-9)


def test_shadow_direction_is_a_unit_vector() -> None:
    for azimuth in range(0, 360, 17):
        east, north = shadow_direction_unit(float(azimuth))
        assert math.hypot(east, north) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "sun_azimuth_deg, expected_d_row, expected_d_col",
    [
        # 20 m tall, 45 deg elevation -> 20 m shadow -> 40 px at 0.5 m GSD.
        (0.0, 40.0, 0.0),  # shadow South -> row index increases
        (90.0, 0.0, -40.0),  # shadow West  -> col index decreases
        (180.0, -40.0, 0.0),  # shadow North -> row index decreases
        (270.0, 0.0, 40.0),  # shadow East  -> col index increases
    ],
)
def test_shadow_pixel_offset_cardinal_directions(
    sun_azimuth_deg: float, expected_d_row: float, expected_d_col: float
) -> None:
    d_row, d_col = shadow_pixel_offset(
        height_m=20.0, sun_elevation_deg=45.0, sun_azimuth_deg=sun_azimuth_deg, gsd_m=0.5
    )
    assert d_row == pytest.approx(expected_d_row, abs=1e-9)
    assert d_col == pytest.approx(expected_d_col, abs=1e-9)


def test_shadow_pixel_offset_magnitude_matches_shadow_length() -> None:
    height, elevation, gsd = 33.0, 37.0, 0.3
    d_row, d_col = shadow_pixel_offset(height, elevation, 212.0, gsd)
    expected_px = shadow_length_m(height, elevation) / gsd
    assert math.hypot(d_row, d_col) == pytest.approx(expected_px)


def test_zero_gsd_raises() -> None:
    with pytest.raises(ValueError, match="gsd_m"):
        shadow_pixel_offset(10.0, 45.0, 135.0, 0.0)
