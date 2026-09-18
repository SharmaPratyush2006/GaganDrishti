"""Shadow direction, the anti-sun azimuth, and the pixel-to-metre conversion.

Getting the shadow direction backwards is the single cheapest way to measure a
height of zero on real imagery -- rays launched into the sunlit side find no
shadow at all -- so the bearing is pinned from several independent directions
here: against a closed form, against the compass, and against the raster row
and column convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthwizard.physics import (
    height_from_shadow_length_m,
    pixels_to_metres,
    shadow_azimuth_deg,
    shadow_direction_pixels,
    shadow_direction_unit,
    shadow_length_m,
    shadow_pixel_offset,
)


# ---------------------------------------------------------------------------
# The anti-sun bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sun_deg", "expected_shadow_deg"),
    [
        (0.0, 180.0),  # sun due North -> shadow due South
        (90.0, 270.0),  # sun due East  -> shadow due West
        (135.0, 315.0),  # sun SE        -> shadow NW
        (180.0, 0.0),  # sun due South -> shadow due North
        (270.0, 90.0),  # sun due West  -> shadow due East
        (315.0, 135.0),  # sun NW        -> shadow SE
    ],
)
def test_shadow_azimuth_is_the_opposite_compass_bearing(sun_deg, expected_shadow_deg):
    assert shadow_azimuth_deg(sun_deg) == pytest.approx(expected_shadow_deg)


@pytest.mark.parametrize("sun_deg", [-360.0, -45.0, 0.0, 359.9, 360.0, 720.0, 1080.5])
def test_shadow_azimuth_wraps_into_0_360(sun_deg):
    """Any real bearing is accepted; the result is always a compass bearing."""
    result = shadow_azimuth_deg(sun_deg)
    assert 0.0 <= result < 360.0


def test_shadow_azimuth_is_its_own_inverse():
    """Applying it twice returns the original bearing: +180 twice is +360."""
    for sun_deg in np.arange(0.0, 360.0, 11.0):
        assert shadow_azimuth_deg(shadow_azimuth_deg(sun_deg)) == pytest.approx(sun_deg)


def test_shadow_azimuth_matches_a_full_turn_offset():
    """Equivalent bearings 360 deg apart give the same shadow direction."""
    for sun_deg in np.arange(0.0, 360.0, 17.0):
        assert shadow_azimuth_deg(sun_deg) == pytest.approx(shadow_azimuth_deg(sun_deg + 360.0))


# ---------------------------------------------------------------------------
# Ground-space unit vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sun_deg", "east", "north"),
    [
        (0.0, 0.0, -1.0),  # sun North -> shadow runs South
        (90.0, -1.0, 0.0),  # sun East  -> shadow runs West
        (180.0, 0.0, 1.0),  # sun South -> shadow runs North
        (270.0, 1.0, 0.0),  # sun West  -> shadow runs East
    ],
)
def test_shadow_direction_unit_cardinal_directions(sun_deg, east, north):
    got_east, got_north = shadow_direction_unit(sun_deg)
    assert got_east == pytest.approx(east, abs=1e-12)
    assert got_north == pytest.approx(north, abs=1e-12)


def test_shadow_direction_unit_is_a_unit_vector():
    for sun_deg in np.arange(0.0, 360.0, 13.0):
        east, north = shadow_direction_unit(sun_deg)
        assert math.hypot(east, north) == pytest.approx(1.0, abs=1e-12)


def test_shadow_direction_unit_points_away_from_the_sun():
    """The shadow vector is the exact negative of the vector towards the sun."""
    for sun_deg in np.arange(0.0, 360.0, 13.0):
        sun_east = math.sin(math.radians(sun_deg))
        sun_north = math.cos(math.radians(sun_deg))
        shadow_east, shadow_north = shadow_direction_unit(sun_deg)
        assert shadow_east == pytest.approx(-sun_east, abs=1e-12)
        assert shadow_north == pytest.approx(-sun_north, abs=1e-12)
        # Anti-parallel: the dot product is exactly -1.
        assert shadow_east * sun_east + shadow_north * sun_north == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Raster index space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sun_deg", "d_row", "d_col"),
    [
        # Row index grows southward, so a northward shadow is a NEGATIVE d_row.
        (0.0, 1.0, 0.0),  # sun North -> shadow South -> row increases
        (90.0, 0.0, -1.0),  # sun East  -> shadow West  -> col decreases
        (180.0, -1.0, 0.0),  # sun South -> shadow North -> row decreases
        (270.0, 0.0, 1.0),  # sun West  -> shadow East  -> col increases
        (135.0, -0.7071067811865476, -0.7071067811865476),  # sun SE -> shadow NW
    ],
)
def test_shadow_direction_pixels_row_grows_southward(sun_deg, d_row, d_col):
    got_row, got_col = shadow_direction_pixels(sun_deg)
    assert got_row == pytest.approx(d_row, abs=1e-12)
    assert got_col == pytest.approx(d_col, abs=1e-12)


def test_shadow_direction_pixels_is_a_unit_vector():
    for sun_deg in np.arange(0.0, 360.0, 13.0):
        d_row, d_col = shadow_direction_pixels(sun_deg)
        assert math.hypot(d_row, d_col) == pytest.approx(1.0, abs=1e-12)


def test_shadow_direction_pixels_flips_north_relative_to_ground_space():
    """East maps straight onto d_col; North maps onto MINUS d_row."""
    for sun_deg in np.arange(0.0, 360.0, 13.0):
        east, north = shadow_direction_unit(sun_deg)
        d_row, d_col = shadow_direction_pixels(sun_deg)
        assert d_col == pytest.approx(east, abs=1e-12)
        assert d_row == pytest.approx(-north, abs=1e-12)


def test_shadow_direction_pixels_agrees_with_shadow_pixel_offset():
    """The unit step and the full tip offset must describe the same ray."""
    height_m, elevation_deg, gsd_m = 30.0, 45.0, 0.5
    for sun_deg in np.arange(0.0, 360.0, 19.0):
        d_row, d_col = shadow_direction_pixels(sun_deg)
        offset_row, offset_col = shadow_pixel_offset(height_m, elevation_deg, sun_deg, gsd_m)
        length_px = shadow_length_m(height_m, elevation_deg) / gsd_m
        assert offset_row == pytest.approx(d_row * length_px, abs=1e-9)
        assert offset_col == pytest.approx(d_col * length_px, abs=1e-9)


# ---------------------------------------------------------------------------
# Pixels to metres
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length_px", "gsd_m", "expected_m"),
    [
        (0.0, 0.5, 0.0),
        (1.0, 0.5, 0.5),
        (60.0, 0.5, 30.0),
        (24.0, 0.5, 12.0),
        (100.0, 0.31, 31.0),
        (1.0, 2.0, 2.0),
    ],
)
def test_pixels_to_metres(length_px, gsd_m, expected_m):
    assert pixels_to_metres(length_px, gsd_m) == pytest.approx(expected_m)


def test_pixels_to_metres_is_linear_so_an_uncertainty_scales_like_its_measurement():
    """dL and L share one GSD, which is why dL_m / L_m == dL_px / L_px."""
    gsd_m = 0.37
    length_px, uncertainty_px = 42.0, 1.5
    length_m = pixels_to_metres(length_px, gsd_m)
    uncertainty_m = pixels_to_metres(uncertainty_px, gsd_m)
    assert uncertainty_m / length_m == pytest.approx(uncertainty_px / length_px)


def test_pixels_to_metres_rejects_a_negative_length():
    with pytest.raises(ValueError, match="length_px must be >= 0"):
        pixels_to_metres(-1.0, 0.5)


@pytest.mark.parametrize("gsd_m", [0.0, -0.5])
def test_pixels_to_metres_rejects_a_non_positive_gsd(gsd_m):
    with pytest.raises(ValueError, match="gsd_m must be positive"):
        pixels_to_metres(10.0, gsd_m)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_pixels_to_metres_rejects_non_finite_input(bad):
    """NaN and inf silently poison every downstream metre, so they stop here."""
    with pytest.raises(ValueError):
        pixels_to_metres(bad, 0.5)
    with pytest.raises(ValueError):
        pixels_to_metres(10.0, bad)


# ---------------------------------------------------------------------------
# Forward and inverse must be consistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("height_m", [1.0, 6.0, 12.0, 30.0, 45.0, 120.0])
@pytest.mark.parametrize("elevation_deg", [15.0, 25.0, 35.0, 45.0, 60.0, 80.0])
def test_shadow_length_and_height_are_exact_inverses(height_m, elevation_deg):
    length = shadow_length_m(height_m, elevation_deg)
    assert height_from_shadow_length_m(length, elevation_deg) == pytest.approx(height_m, rel=1e-12)


def test_at_45_degrees_shadow_length_equals_height():
    """tan(45) == 1, which is why the fixture uses 45 deg: L and h coincide."""
    for height_m in (6.0, 12.0, 30.0, 45.0):
        assert shadow_length_m(height_m, 45.0) == pytest.approx(height_m)
