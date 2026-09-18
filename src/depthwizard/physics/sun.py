"""Sun and shadow geometry.

This is the core physical relationship DepthWizard is built on. For a vertical
object of height ``h`` standing on flat ground, lit by a sun at elevation
``theta`` above the horizon, the cast shadow has length::

    L = h / tan(theta)

and therefore::

    h = L * tan(theta)

Measure ``L`` from imagery and you recover ``h`` without stereo, lidar or any
second view -- that is the whole idea.

Conventions used throughout DepthWizard
---------------------------------------
``sun_elevation_deg``
    Angle of the sun above the horizon, in degrees, in (0, 90].
    90 deg is directly overhead and casts no shadow.

``sun_azimuth_deg``
    Compass bearing of the sun as seen from the ground, in degrees clockwise
    from North: 0 = N, 90 = E, 180 = S, 270 = W.

Rasters
    Assumed north-up: column index increases towards East, row index increases
    towards South. This matches a standard GeoTIFF with a negative y pixel size.

The shadow points *away* from the sun, i.e. along bearing
``sun_azimuth_deg + 180``.
"""

from __future__ import annotations

import math

__all__ = [
    "shadow_length_m",
    "height_from_shadow_length_m",
    "shadow_azimuth_deg",
    "shadow_direction_unit",
    "shadow_direction_pixels",
    "shadow_pixel_offset",
]


def _validate_elevation(sun_elevation_deg: float) -> float:
    elevation = float(sun_elevation_deg)
    if not 0.0 < elevation <= 90.0:
        raise ValueError(
            f"sun_elevation_deg must be in (0, 90], got {elevation}. "
            "A sun at or below the horizon casts no measurable shadow."
        )
    return elevation


def shadow_length_m(height_m: float, sun_elevation_deg: float) -> float:
    """Length of the shadow cast by a vertical object of ``height_m``.

    ``L = h / tan(theta)``. At ``sun_elevation_deg == 90`` the result is 0.

    Args:
        height_m: Object height in metres (must be >= 0).
        sun_elevation_deg: Sun elevation above the horizon, in (0, 90].

    Returns:
        Shadow length in metres, measured on flat ground.
    """
    elevation = _validate_elevation(sun_elevation_deg)
    height = float(height_m)
    if height < 0.0:
        raise ValueError(f"height_m must be >= 0, got {height}")
    if elevation == 90.0:
        return 0.0
    return height / math.tan(math.radians(elevation))


def height_from_shadow_length_m(shadow_length: float, sun_elevation_deg: float) -> float:
    """Recover object height from a measured shadow length.

    The exact inverse of :func:`shadow_length_m`: ``h = L * tan(theta)``.

    This is the estimator the later phases will feed with *measured* shadow
    lengths. Phase 0 only uses it to prove the forward and inverse models agree.
    """
    elevation = _validate_elevation(sun_elevation_deg)
    length = float(shadow_length)
    if length < 0.0:
        raise ValueError(f"shadow_length must be >= 0, got {length}")
    return length * math.tan(math.radians(elevation))


def shadow_azimuth_deg(sun_azimuth_deg: float) -> float:
    """The anti-sun bearing: the compass direction a shadow runs towards.

    ``shadow_azimuth = (sun_azimuth + 180) % 360``

    Why the anti-sun direction? ``sun_azimuth_deg`` is the bearing *from which*
    the sunlight arrives -- the direction you look to see the sun. A vertical
    object blocks that light, so the unlit ground lies on the opposite side of
    the object, directly away from the sun. A sun in the south-east (135 deg)
    therefore throws shadows to the north-west (315 deg).

    Getting this backwards is the single easiest way to measure a height of
    zero on real imagery, which is why it is one named function used everywhere
    rather than a ``+ 180`` scattered through the codebase.

    Args:
        sun_azimuth_deg: Compass bearing of the sun, degrees clockwise from
            North (0 = N, 90 = E, 180 = S, 270 = W). Any real value is accepted
            and wrapped.

    Returns:
        The shadow bearing in degrees clockwise from North, wrapped to [0, 360).

    Examples:
        >>> shadow_azimuth_deg(135.0)
        315.0
        >>> shadow_azimuth_deg(315.0)
        135.0
    """
    return (float(sun_azimuth_deg) + 180.0) % 360.0


def shadow_direction_unit(sun_azimuth_deg: float) -> tuple[float, float]:
    """Unit vector, in ground coordinates, pointing along the shadow.

    Args:
        sun_azimuth_deg: Compass bearing of the sun, degrees clockwise from North.

    Returns:
        ``(east, north)`` components of a unit vector pointing away from the
        sun, i.e. along bearing :func:`shadow_azimuth_deg`.

    Examples:
        A sun in the East (azimuth 90) casts shadows towards the West::

            >>> e, n = shadow_direction_unit(90.0)
            >>> round(e, 6), round(n, 6)
            (-1.0, -0.0)
    """
    shadow_bearing = math.radians(shadow_azimuth_deg(sun_azimuth_deg))
    east = math.sin(shadow_bearing)
    north = math.cos(shadow_bearing)
    return east, north


def shadow_direction_pixels(sun_azimuth_deg: float) -> tuple[float, float]:
    """Unit vector, in raster index space, pointing along the shadow.

    The conversion from map to image coordinates for a north-up raster is the
    one recorded in this module's header: column index increases towards East,
    row index increases towards **South**. East therefore maps straight onto
    ``d_col``, while North maps onto ``-d_row``.

    Args:
        sun_azimuth_deg: Compass bearing of the sun, degrees clockwise from North.

    Returns:
        ``(d_row, d_col)`` components of a unit vector, so that stepping
        ``t`` pixels along the shadow from ``(row, col)`` lands on
        ``(row + t * d_row, col + t * d_col)``.

    Examples:
        A sun in the south-east (135) throws shadows to the north-west, which
        is decreasing row *and* decreasing column::

            >>> d_row, d_col = shadow_direction_pixels(135.0)
            >>> round(d_row, 6), round(d_col, 6)
            (-0.707107, -0.707107)
    """
    east, north = shadow_direction_unit(sun_azimuth_deg)
    return -north, east


def shadow_pixel_offset(
    height_m: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    gsd_m: float,
) -> tuple[float, float]:
    """Pixel offset from an object's base to the tip of its shadow.

    Combines :func:`shadow_length_m` with :func:`shadow_direction_unit` and
    converts to pixel units for a north-up raster.

    Args:
        height_m: Object height in metres.
        sun_elevation_deg: Sun elevation above the horizon, in (0, 90].
        sun_azimuth_deg: Compass bearing of the sun, degrees clockwise from North.
        gsd_m: Ground sample distance (pixel size) in metres.

    Returns:
        ``(d_row, d_col)`` offset in pixels. ``d_col`` is positive towards East,
        ``d_row`` is positive towards South (increasing row index).
    """
    gsd = float(gsd_m)
    if gsd <= 0.0:
        raise ValueError(f"gsd_m must be positive, got {gsd}")

    length = shadow_length_m(height_m, sun_elevation_deg)
    east, north = shadow_direction_unit(sun_azimuth_deg)

    d_col = east * length / gsd
    # Row index grows southward, so a northward ground offset is a negative d_row.
    d_row = -north * length / gsd
    return d_row, d_col
