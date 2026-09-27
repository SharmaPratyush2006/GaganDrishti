"""Sun azimuth recovered from where shadows lie relative to buildings.

DFC2019 Track 1 tiles carry no sun metadata, so the direction shadows run in
must come from the image. This module estimates it from a shadow mask and the
building footprints, and converts it to a **sun** azimuth.

The method: footprint-adjacent directional shadow occupancy
------------------------------------------------------------
For every candidate sun azimuth ``phi`` (a full 0-360 degree scan):

1. the shadow direction is the anti-sun bearing,
   :func:`~depthwizard.physics.sun.shadow_azimuth_deg` ``(phi)``, as a unit
   pixel step from :func:`~depthwizard.physics.sun.shadow_direction_pixels`;
2. the building mask is translated 1..``band_px`` pixels along that step and the
   union taken, minus the buildings themselves. That is the band of ground
   directly "downsun" of every building;
3. the score is the fraction of that band flagged as shadow.

The sun azimuth is the candidate with the highest score. Rounding to whole
pixels makes neighbouring candidates produce the same band, so the maximum is
usually a plateau; the estimate is the circular mean of that plateau.

Why this, and not PCA / gradient orientation / a Radon transform: all three
return an *axis*, with a 180-degree ambiguity that the anti-sun conversion
cannot survive, and all three measure the elongation of the shadow blobs --
which in a city follows facades and streets, not the sun. Occupancy next to the
casters measures the quantity itself (which side of a building its shadow is
on), resolves the sign, and is anti-sun by construction, because the scan is
over sun azimuths and each candidate's shadow direction is derived with the one
project-wide conversion.

Resolution
----------
A displacement of ``band_px`` pixels can only resolve directions to about
``atan(0.5 / band_px)`` (3.6 deg at the default 8 px): a smaller change of
angle does not move the band's far end by half a pixel. That is the stated
quantisation bound, :func:`angular_resolution_deg`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from depthwizard.physics.sun import shadow_azimuth_deg, shadow_direction_pixels

__all__ = [
    "AZIMUTH_METHOD",
    "AzimuthEstimate",
    "angular_resolution_deg",
    "angular_difference_deg",
    "estimate_sun_azimuth",
]

AZIMUTH_METHOD = "footprint_adjacent_directional_shadow_occupancy"


def angular_resolution_deg(band_px: int) -> float:
    """Quantisation bound of the estimator for a band of ``band_px`` pixels."""
    return math.degrees(math.atan(0.5 / float(band_px)))


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    diff = (float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


@dataclass(frozen=True)
class AzimuthEstimate:
    ok: bool
    method: str
    sun_azimuth_deg: float | None
    shadow_azimuth_deg: float | None
    peak_score: float
    median_score: float
    peak_to_median: float | None
    plateau_width_deg: float
    band_px: int
    resolution_deg: float
    failure_reason: str | None = None
    candidates_deg: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    scores: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "shadow_azimuth_deg": self.shadow_azimuth_deg,
            "peak_score": self.peak_score,
            "median_score": self.median_score,
            "peak_to_median": self.peak_to_median,
            "plateau_width_deg": self.plateau_width_deg,
            "band_px": self.band_px,
            "resolution_deg": self.resolution_deg,
            "failure_reason": self.failure_reason,
        }


def _shift(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    """``out[r, c] = mask[r - d_row, c - d_col]``, zero-filled."""
    out = np.zeros_like(mask)
    n_rows, n_cols = mask.shape
    r0, r1 = max(0, d_row), min(n_rows, n_rows + d_row)
    c0, c1 = max(0, d_col), min(n_cols, n_cols + d_col)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = mask[r0 - d_row : r1 - d_row, c0 - d_col : c1 - d_col]
    return out


def estimate_sun_azimuth(
    shadow_mask: np.ndarray,
    building_mask: np.ndarray,
    *,
    band_px: int = 8,
    step_deg: float = 1.0,
    min_building_px: int = 200,
    min_peak_to_median: float = 1.25,
) -> AzimuthEstimate:
    """Estimate the sun azimuth (degrees clockwise from North).

    Args:
        shadow_mask: Boolean shadow mask (north-up raster).
        building_mask: Boolean mask of the shadow casters.
        band_px: How far past each building the shadow occupancy is scored.
        step_deg: Candidate spacing over the full circle.
        min_building_px: Fewer casters than this -> not estimated.
        min_peak_to_median: Required peak / median score. Below it, shadows are
            no more likely on one side of the buildings than on any other and no
            direction is reported.

    Returns:
        An :class:`AzimuthEstimate`; ``ok`` is False, with a reason, when no
        defensible direction exists. No default azimuth is ever substituted.
    """
    shadow = np.asarray(shadow_mask, dtype=bool)
    casters = np.asarray(building_mask, dtype=bool)
    if shadow.shape != casters.shape:
        raise ValueError(f"shadow mask {shadow.shape} and building mask {casters.shape} differ")
    candidates = np.arange(0.0, 360.0, float(step_deg))
    resolution = angular_resolution_deg(band_px)

    def failed(reason: str, scores: np.ndarray | None = None) -> AzimuthEstimate:
        peak = float(scores.max()) if scores is not None and scores.size else 0.0
        median = float(np.median(scores)) if scores is not None and scores.size else 0.0
        return AzimuthEstimate(
            ok=False,
            method=AZIMUTH_METHOD,
            sun_azimuth_deg=None,
            shadow_azimuth_deg=None,
            peak_score=peak,
            median_score=median,
            peak_to_median=(peak / median) if median > 0 else None,
            plateau_width_deg=0.0,
            band_px=int(band_px),
            resolution_deg=resolution,
            failure_reason=reason,
            candidates_deg=candidates,
            scores=scores,
        )

    n_casters = int(casters.sum())
    if n_casters < min_building_px:
        return failed(f"only {n_casters} building pixels, need {min_building_px}")
    if not shadow.any():
        return failed("shadow mask is empty")

    # Shifted copies depend only on the integer offset, so cache them.
    cache: dict[tuple[int, int], np.ndarray] = {}
    scores = np.zeros(candidates.size, dtype=np.float64)
    for index, sun_az in enumerate(candidates):
        d_row, d_col = shadow_direction_pixels(float(sun_az))
        band = np.zeros_like(casters)
        for t in range(1, int(band_px) + 1):
            offset = (int(round(d_row * t)), int(round(d_col * t)))
            if offset not in cache:
                cache[offset] = _shift(casters, *offset)
            band |= cache[offset]
        band &= ~casters
        scores[index] = float(shadow[band].mean()) if band.any() else 0.0

    peak = float(scores.max())
    median = float(np.median(scores))
    if peak <= 0.0:
        return failed("no shadow adjacent to any building in any direction", scores)
    ratio = peak / median if median > 0 else math.inf
    if ratio < min_peak_to_median:
        return failed(
            f"peak/median shadow occupancy {ratio:.3f} is below {min_peak_to_median}; "
            "no dominant shadow direction",
            scores,
        )

    plateau = candidates[scores >= peak * (1.0 - 1e-12)]
    angles = np.radians(plateau)
    sun_az = math.degrees(math.atan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360.0
    # Plateau width, measured around its circular mean so a plateau across 0 deg is handled.
    offsets = np.array([((a - sun_az + 180.0) % 360.0) - 180.0 for a in plateau])
    width = float(offsets.max() - offsets.min()) if offsets.size else 0.0

    return AzimuthEstimate(
        ok=True,
        method=AZIMUTH_METHOD,
        sun_azimuth_deg=float(sun_az),
        shadow_azimuth_deg=shadow_azimuth_deg(sun_az),
        peak_score=peak,
        median_score=median,
        peak_to_median=float(ratio),
        plateau_width_deg=width,
        band_px=int(band_px),
        resolution_deg=resolution,
        candidates_deg=candidates,
        scores=scores,
    )
