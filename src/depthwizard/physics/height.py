"""Height from shadow length, with uncertainty and a sun-elevation gate.

This module is the inverse physics of :mod:`depthwizard.physics.sun`, packaged
with everything an inspector needs to decide whether to believe the answer.

The equation
------------
For a vertical object of height ``h`` standing on flat ground, lit by a sun at
elevation ``theta`` above the horizon, the shadow it casts on that ground has
length ``L = h / tan(theta)``. Rearranged, that is the estimator::

    h = L * tan(theta)

``theta`` arrives in **degrees** (that is what every satellite product reports)
and is converted to radians before it reaches :func:`math.tan`.

Assumptions baked into this one line -- all of them are error sources, and all
of them are deliberately *not* corrected here:

* The ground the shadow falls on is **flat and horizontal**. A slope towards or
  away from the sun lengthens or shortens the shadow.
* The object is **vertical**, and its top edge is directly above its base.
* The shadow falls on **ground**, not onto another building or a water body.
* The view is **nadir**, so the building's base and roof project to the same
  place and the measured shadow is the full shadow.
* The shadow is not clipped by an occluder, and its far end is the true tip.

Terrain-slope, off-nadir and per-scene bias corrections belong to the
calibration stage, not here. Keeping this function free of them means the raw
physical estimate is always available to compare a calibrated one against.

Uncertainty
-----------
``h = L * tan(theta)`` is linear in ``L`` at fixed ``theta``, so a shadow-length
uncertainty propagates exactly, with no linearisation error::

    dh = tan(theta) * dL

Only the *length* term is propagated. The sun angles come from the product's
metadata and are treated as exact; if a later phase gains a credible angle
uncertainty, it adds a second term here rather than inflating ``dL``.

``dL`` is never invented. It is supplied by the caller -- in metres, or in
pixels, in which case it is converted with the same GSD as the measurement::

    dL_m = dL_px * GSD_m

The usable sun-elevation band
-----------------------------
Shadow-based height is best conditioned when the sun sits between
:data:`USABLE_SUN_ELEVATION_MIN_DEG` and :data:`USABLE_SUN_ELEVATION_MAX_DEG`
degrees above the horizon:

* **Below 25 deg** the shadows are very long. They run into neighbouring
  buildings, off the edge of the tile, and across terrain far enough away that
  the flat-ground assumption stops holding.
* **Above 45 deg** the shadows are short. ``tan(theta)`` grows, so each pixel of
  measurement error is amplified into more metres of height error -- at 45 deg
  one pixel of shadow costs one pixel of height, at 70 deg it costs 2.75.

Outside the band the measurement is **not** rejected: the physics is unchanged
and the number is still the best estimate available. What changes is the
confidence flag and the recorded reason. See :func:`sun_band_status`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from depthwizard.physics.sun import height_from_shadow_length_m

__all__ = [
    "USABLE_SUN_ELEVATION_MIN_DEG",
    "USABLE_SUN_ELEVATION_MAX_DEG",
    "SunBand",
    "Confidence",
    "SunBandStatus",
    "HeightEstimate",
    "sun_band_status",
    "pixels_to_metres",
    "height_uncertainty_m",
    "height_from_shadow",
]

#: Lower edge of the recommended solar-elevation band, in degrees.
USABLE_SUN_ELEVATION_MIN_DEG = 25.0

#: Upper edge of the recommended solar-elevation band, in degrees.
USABLE_SUN_ELEVATION_MAX_DEG = 45.0


class SunBand(str, Enum):
    """Where a scene's sun elevation sits relative to the recommended band."""

    WITHIN = "within_band"
    BELOW = "below_band"
    ABOVE = "above_band"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Confidence(str, Enum):
    """How much to trust one height estimate.

    ``NOMINAL``
        Every gate passed: a shadow was measured, and the sun sat inside the
        recommended band.
    ``REDUCED``
        A height was produced, but at least one gate flagged it. The reasons
        are always listed alongside; nothing is downgraded anonymously.
    ``FAILED``
        No height was produced, because the shadow evidence was insufficient.
        There is no number to trust -- a failed estimate carries ``None``, not
        a guess.
    """

    NOMINAL = "nominal"
    REDUCED = "reduced"
    FAILED = "failed"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _finite(value: Any, name: str) -> float:
    """Coerce to float and reject NaN/inf, which silently poison the physics."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def _validate_elevation_deg(sun_elevation_deg: Any) -> float:
    elevation = _finite(sun_elevation_deg, "sun_elevation_deg")
    if not 0.0 < elevation <= 90.0:
        raise ValueError(
            f"sun_elevation_deg must be in (0, 90], got {elevation}. "
            "A sun at or below the horizon casts no measurable shadow."
        )
    return elevation


# ---------------------------------------------------------------------------
# The sun-elevation confidence gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SunBandStatus:
    """Result of the sun-elevation gate: which side of the band, and why."""

    elevation_deg: float
    band: SunBand
    min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG
    max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG
    #: Human-readable explanation, set only when the elevation is outside the band.
    reason: str | None = None

    @property
    def in_band(self) -> bool:
        """True when the sun sat inside the recommended band."""
        return self.band is SunBand.WITHIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "sun_elevation_deg": self.elevation_deg,
            "sun_band": self.band.value,
            "sun_band_min_deg": self.min_deg,
            "sun_band_max_deg": self.max_deg,
            "sun_band_in_band": self.in_band,
            "sun_band_reason": self.reason,
        }


def sun_band_status(
    sun_elevation_deg: float,
    *,
    min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
    max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
) -> SunBandStatus:
    """Classify a sun elevation against the recommended usable band.

    The band is inclusive at both edges: exactly 25 deg and exactly 45 deg are
    both ``SunBand.WITHIN``.

    This gate **never rejects a scene**. It returns a status; the caller marks
    reduced confidence and records the reason. Processing continues either way,
    because ``h = L * tan(theta)`` is just as true at 15 deg as at 35 deg -- it
    is merely worse conditioned.

    Args:
        sun_elevation_deg: Sun elevation above the horizon, in (0, 90].
        min_deg: Lower edge of the band. Defaults to 25 deg.
        max_deg: Upper edge of the band. Defaults to 45 deg.

    Returns:
        A :class:`SunBandStatus`. ``reason`` is None inside the band and a
        specific sentence outside it.
    """
    elevation = _validate_elevation_deg(sun_elevation_deg)
    low = _finite(min_deg, "min_deg")
    high = _finite(max_deg, "max_deg")
    if not low < high:
        raise ValueError(f"min_deg must be < max_deg, got {low} and {high}")

    if elevation < low:
        return SunBandStatus(
            elevation_deg=elevation,
            band=SunBand.BELOW,
            min_deg=low,
            max_deg=high,
            reason=(
                f"sun elevation outside recommended {low:g}-{high:g} degree band "
                f"({elevation:g} deg is below {low:g} deg): shadows are long and more "
                "likely to be occluded, truncated by the tile edge, or to fall on "
                "terrain that is not flat"
            ),
        )
    if elevation > high:
        return SunBandStatus(
            elevation_deg=elevation,
            band=SunBand.ABOVE,
            min_deg=low,
            max_deg=high,
            reason=(
                f"sun elevation outside recommended {low:g}-{high:g} degree band "
                f"({elevation:g} deg is above {high:g} deg): shadows are short and "
                f"tan(theta)={math.tan(math.radians(elevation)):.3g} amplifies every "
                "pixel of measurement error into more metres of height error"
            ),
        )
    return SunBandStatus(
        elevation_deg=elevation, band=SunBand.WITHIN, min_deg=low, max_deg=high, reason=None
    )


# ---------------------------------------------------------------------------
# Unit conversion and uncertainty
# ---------------------------------------------------------------------------


def pixels_to_metres(length_px: float, gsd_m: float) -> float:
    """Convert a length in pixels to a length in metres: ``L_m = L_px * GSD_m``.

    This is the only place a pixel becomes a metre in the height path, so the
    conversion is one multiplication with one documented direction. The same
    function converts a measurement and its uncertainty, which is why an
    uncertainty in pixels scales exactly like the measurement it belongs to.

    Args:
        length_px: Length in pixels (>= 0).
        gsd_m: Ground sample distance in metres per pixel (> 0).

    Returns:
        The length in metres.
    """
    length = _finite(length_px, "length_px")
    gsd = _finite(gsd_m, "gsd_m")
    if length < 0.0:
        raise ValueError(f"length_px must be >= 0, got {length}")
    if gsd <= 0.0:
        raise ValueError(f"gsd_m must be positive, got {gsd}")
    return length * gsd


def height_uncertainty_m(sun_elevation_deg: float, shadow_length_uncertainty_m: float) -> float:
    """Propagate a shadow-length uncertainty into a height uncertainty.

    ``dh = tan(theta) * dL``. Exact, not a linearisation: ``h`` is linear in
    ``L`` once ``theta`` is fixed.

    Args:
        sun_elevation_deg: Sun elevation above the horizon, in (0, 90].
        shadow_length_uncertainty_m: ``dL`` in **metres** (>= 0). If yours is in
            pixels, run it through :func:`pixels_to_metres` first, using the
            same GSD as the measurement.

    Returns:
        ``dh`` in metres.
    """
    elevation = _validate_elevation_deg(sun_elevation_deg)
    d_length = _finite(shadow_length_uncertainty_m, "shadow_length_uncertainty_m")
    if d_length < 0.0:
        raise ValueError(f"shadow_length_uncertainty_m must be >= 0, got {d_length}")
    return math.tan(math.radians(elevation)) * d_length


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeightEstimate:
    """One height, plus every number that produced it.

    Nothing here is derived behind the caller's back: the shadow length, the
    sun elevation, ``tan(theta)``, the uncertainties and the band status are all
    carried alongside the answer so the arithmetic can be re-done by hand.
    """

    height_m: float
    shadow_length_m: float
    sun_elevation_deg: float
    tan_theta: float
    sun_band: SunBandStatus
    confidence: Confidence
    #: ``dL`` in metres, or None when the caller supplied no uncertainty.
    shadow_length_uncertainty_m: float | None = None
    #: ``dh = tan(theta) * dL`` in metres, or None when ``dL`` was not supplied.
    height_uncertainty_m: float | None = None
    #: Every reason this estimate is not ``Confidence.NOMINAL``.
    reduced_confidence_reasons: tuple[str, ...] = ()

    @property
    def equation(self) -> str:
        """The calculation as a readable string, for logs and reports."""
        return (
            f"h = L * tan(theta) = {self.shadow_length_m:.4f} m * "
            f"tan({self.sun_elevation_deg:g} deg) = "
            f"{self.shadow_length_m:.4f} * {self.tan_theta:.6f} = {self.height_m:.4f} m"
        )

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "shadow_length_m": self.shadow_length_m,
            "sun_elevation_deg": self.sun_elevation_deg,
            "tan_theta": self.tan_theta,
            "height_m": self.height_m,
            "shadow_length_uncertainty_m": self.shadow_length_uncertainty_m,
            "height_uncertainty_m": self.height_uncertainty_m,
            "confidence_flag": self.confidence.value,
            "reduced_confidence_reasons": list(self.reduced_confidence_reasons),
            "equation": self.equation,
        }
        record.update(self.sun_band.to_dict())
        return record


def height_from_shadow(
    shadow_length_m: float,
    sun_elevation_deg: float,
    *,
    shadow_length_uncertainty_m: float | None = None,
    extra_reduced_confidence_reasons: Iterable[str] = (),
    sun_band_min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
    sun_band_max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
) -> HeightEstimate:
    """Estimate a height from a measured shadow length: ``h = L * tan(theta)``.

    The sun-elevation gate is applied here, but only ever *downgrades the
    confidence flag*. A scene with the sun at 60 deg or 10 deg still gets a
    height; it gets ``Confidence.REDUCED`` and a recorded reason with it.

    Args:
        shadow_length_m: Measured shadow length ``L`` in metres (>= 0, finite).
        sun_elevation_deg: Sun elevation ``theta`` above the horizon, in (0, 90].
            Degrees; converted to radians internally.
        shadow_length_uncertainty_m: ``dL`` in metres, if known. Supply it and
            both uncertainties are reported; omit it and both stay None. No
            default uncertainty is invented.
        extra_reduced_confidence_reasons: Reasons contributed by an earlier
            stage -- typically the shadow measurement -- that should also mark
            this estimate as reduced confidence.
        sun_band_min_deg: Lower edge of the usable band. Defaults to 25 deg.
        sun_band_max_deg: Upper edge of the usable band. Defaults to 45 deg.

    Returns:
        A :class:`HeightEstimate` carrying the height and its full derivation.

    Raises:
        ValueError: on a negative or non-finite shadow length, a non-finite or
            out-of-range sun elevation, or a negative uncertainty.
    """
    elevation = _validate_elevation_deg(sun_elevation_deg)
    length_m = _finite(shadow_length_m, "shadow_length_m")
    if length_m < 0.0:
        raise ValueError(f"shadow_length_m must be >= 0, got {length_m}")

    # Degrees in, radians to the trig function -- converted once, inside sun.py.
    height_m = height_from_shadow_length_m(length_m, elevation)
    tan_theta = math.tan(math.radians(elevation))

    d_length: float | None = None
    d_height: float | None = None
    if shadow_length_uncertainty_m is not None:
        d_length = _finite(shadow_length_uncertainty_m, "shadow_length_uncertainty_m")
        if d_length < 0.0:
            raise ValueError(f"shadow_length_uncertainty_m must be >= 0, got {d_length}")
        d_height = height_uncertainty_m(elevation, d_length)

    band = sun_band_status(elevation, min_deg=sun_band_min_deg, max_deg=sun_band_max_deg)

    reasons = list(extra_reduced_confidence_reasons)
    if band.reason is not None:
        reasons.append(band.reason)
    confidence = Confidence.REDUCED if reasons else Confidence.NOMINAL

    return HeightEstimate(
        height_m=height_m,
        shadow_length_m=length_m,
        sun_elevation_deg=elevation,
        tan_theta=tan_theta,
        sun_band=band,
        confidence=confidence,
        shadow_length_uncertainty_m=d_length,
        height_uncertainty_m=d_height,
        reduced_confidence_reasons=tuple(reasons),
    )
