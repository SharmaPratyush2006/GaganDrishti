"""Phase 5: the cheap confidence proxy -- flags, not probabilities.

This is **not** a learned uncertainty and **not** a probability of correctness.
It is a per-pixel label saying which known failure conditions of shadow-based
height apply, built only from information earlier phases already produce:

1. **Shadow occlusion** -- the pixel lies in a detected cast shadow, so its own
   appearance is hidden and its height is inferred from darker, noisier
   evidence. Mask: the Phase 2 :class:`~depthwizard.shadows.detector.ClassicalShadowDetector`
   (unchanged). On DFC2019 that detector is known to also mark dark asphalt
   (Phase 4a finding), so this flag over-counts there.
2. **Water** -- a water surface has no stable height, casts no shadow and
   reflects specularly. Mask: a label the data carries (DFC2019 CLS class 9).
   Unsuitable, not merely reduced.
3. **Sun elevation outside the usable band** (default 25-45 deg, inclusive) --
   a scene-level condition, evaluated with
   :func:`depthwizard.physics.height.sun_band_status`, the same gate Phase 2 uses.

States and precedence
---------------------
Each valid pixel receives exactly one state; the **first** that applies wins::

    code  state          meaning
    0     INVALID        no valid prediction/reference at this pixel
    1     UNSUITABLE     water: a height here is not meaningful
    2     REDUCED        shadow-occluded, and/or the scene's sun elevation is
                         outside the band (all applicable reasons are counted)
    3     NOT_ASSESSED   no condition fired, but at least one check could not
                         be run (its input was unavailable) -- so the pixel
                         cannot honestly be called HIGH
    4     HIGH           every check ran and none fired

An unavailable input is never treated as "no pixels affected": the check is
reported as not available, and pixels it would have covered cannot reach HIGH.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from depthwizard.physics.height import (
    USABLE_SUN_ELEVATION_MAX_DEG,
    USABLE_SUN_ELEVATION_MIN_DEG,
    sun_band_status,
)

__all__ = [
    "NOT_AVAILABLE",
    "ConfidenceState",
    "CONFIDENCE_MEANINGS",
    "ConfidenceResult",
    "confidence_proxy",
]

#: Rendered for a check whose input does not exist.
NOT_AVAILABLE = "not available"


class ConfidenceState(IntEnum):
    INVALID = 0
    UNSUITABLE = 1
    REDUCED = 2
    NOT_ASSESSED = 3
    HIGH = 4


CONFIDENCE_MEANINGS: dict[str, str] = {
    "INVALID": "no valid prediction/reference at this pixel",
    "UNSUITABLE": "water surface; a height here is not meaningful",
    "REDUCED": "shadow-occluded and/or scene sun elevation outside the usable band",
    "NOT_ASSESSED": "no condition fired, but at least one check's input was unavailable",
    "HIGH": "every check ran and none fired",
}


@dataclass(frozen=True)
class ConfidenceResult:
    """The per-pixel state raster and a diagnostic account of it."""

    #: ``(H, W)`` uint8 of :class:`ConfidenceState` codes.
    state: np.ndarray
    diagnostics: dict[str, Any]

    def mask(self, state: ConfidenceState) -> np.ndarray:
        return self.state == int(state)


def _check_mask(mask: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} mask has shape {mask.shape}, expected {shape}")
    return mask


def _flag_record(mask: np.ndarray | None, valid: np.ndarray, source: str, unavailable: str | None) -> dict[str, Any]:
    n_valid = int(valid.sum())
    if mask is None:
        return {"available": False, "status": NOT_AVAILABLE, "reason": unavailable, "source": source,
                "flagged_valid_pixels": NOT_AVAILABLE, "flagged_fraction_of_valid": NOT_AVAILABLE}
    flagged = int((mask & valid).sum())
    return {"available": True, "status": "evaluated", "reason": None, "source": source,
            "flagged_valid_pixels": flagged,
            "flagged_fraction_of_valid": flagged / n_valid if n_valid else NOT_AVAILABLE}


def confidence_proxy(
    valid: np.ndarray,
    *,
    shadow_mask: np.ndarray | None,
    water_mask: np.ndarray | None,
    sun_elevation_deg: float | None,
    shadow_source: str = "",
    water_source: str = "",
    sun_source: str = "",
    shadow_unavailable_reason: str | None = None,
    water_unavailable_reason: str | None = None,
    sun_unavailable_reason: str | None = None,
    sun_band_min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
    sun_band_max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
) -> ConfidenceResult:
    """Label every pixel with a :class:`ConfidenceState`.

    Args:
        valid: The evaluation validity mask; invalid pixels become INVALID.
        shadow_mask: Detected cast shadow, or None when unavailable.
        water_mask: Water pixels from a data label, or None when unavailable.
        sun_elevation_deg: The scene's sun elevation, or None when unknown.
        *_source: Where each input came from (recorded, not interpreted).
        *_unavailable_reason: Why an input is None (recorded in diagnostics).
        sun_band_min_deg, sun_band_max_deg: The usable band (inclusive).

    Returns:
        A :class:`ConfidenceResult`.
    """
    valid = np.asarray(valid, dtype=bool)
    shape = valid.shape
    shadow = _check_mask(shadow_mask, shape, "shadow")
    water = _check_mask(water_mask, shape, "water")

    if sun_elevation_deg is None:
        band = None
        sun_out = None
        sun_record: dict[str, Any] = {
            "available": False, "status": NOT_AVAILABLE, "reason": sun_unavailable_reason,
            "source": sun_source, "sun_elevation_deg": NOT_AVAILABLE,
            "band_deg": [sun_band_min_deg, sun_band_max_deg], "sun_band": NOT_AVAILABLE,
            "flagged_valid_pixels": NOT_AVAILABLE,
        }
    else:
        band = sun_band_status(sun_elevation_deg, min_deg=sun_band_min_deg, max_deg=sun_band_max_deg)
        sun_out = not band.in_band
        sun_record = {
            "available": True, "status": "evaluated", "reason": band.reason, "source": sun_source,
            "sun_elevation_deg": band.elevation_deg, "band_deg": [band.min_deg, band.max_deg],
            "sun_band": band.band.value, "in_band": band.in_band,
            "flagged_valid_pixels": int(valid.sum()) if sun_out else 0,
            "flagged_fraction_of_valid": (1.0 if sun_out else 0.0) if valid.any() else NOT_AVAILABLE,
        }

    reduced = np.zeros(shape, dtype=bool)
    if shadow is not None:
        reduced |= shadow
    if sun_out:
        reduced[:] = True
    unsuitable = water if water is not None else np.zeros(shape, dtype=bool)
    all_checks_ran = shadow is not None and water is not None and band is not None

    state = np.full(shape, int(ConfidenceState.HIGH if all_checks_ran else ConfidenceState.NOT_ASSESSED), dtype=np.uint8)
    # Lowest precedence first, so each later assignment overrides it.
    state[reduced] = int(ConfidenceState.REDUCED)
    state[unsuitable] = int(ConfidenceState.UNSUITABLE)
    state[~valid] = int(ConfidenceState.INVALID)

    n_valid = int(valid.sum())
    counts = {s.name: int((state == int(s)).sum()) for s in ConfidenceState}
    shadow_and_sun = (
        int((shadow & valid & ~unsuitable).sum()) if (shadow is not None and sun_out) else 0
    )
    diagnostics = {
        "kind": "confidence proxy (rule-based flags; NOT a calibrated probability of correctness)",
        "states": {s.name: {"code": int(s), "meaning": CONFIDENCE_MEANINGS[s.name]} for s in ConfidenceState},
        "precedence": ["INVALID", "UNSUITABLE", "REDUCED", "NOT_ASSESSED", "HIGH"],
        "valid_pixels": n_valid,
        "state_pixels": counts,
        "state_fraction_of_valid": {
            name: (count / n_valid if n_valid else NOT_AVAILABLE)
            for name, count in counts.items() if name != "INVALID"
        },
        "all_checks_available": all_checks_ran,
        "checks": {
            "shadow_occlusion": _flag_record(shadow, valid, shadow_source, shadow_unavailable_reason),
            "water": _flag_record(water, valid, water_source, water_unavailable_reason),
            "sun_band": sun_record,
        },
        "overlaps": {"shadow_and_sun_out_of_band_valid_pixels": shadow_and_sun},
    }
    return ConfidenceResult(state=state, diagnostics=diagnostics)
