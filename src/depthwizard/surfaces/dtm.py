"""Phase 6: DTM from a DSM and a ground mask, and nDSM = DSM - DTM.

Terms (kept distinct everywhere in Phase 6)
-------------------------------------------
DSM   Digital Surface Model: absolute elevation of the top surface (roofs, ground).
DTM   Digital Terrain Model: absolute elevation of the bare ground, including
      *under* buildings, where it cannot be observed and is interpolated.
nDSM  normalised DSM = DSM - DTM: height of each object **above the ground**.
      A DSM value on a roof is an absolute elevation, never a building height.

DTM construction
----------------
1. **Ground pixels keep their DSM value** -- they are the terrain observations.
   Building tops are never used: only pixels the ground extractor kept.
2. **TIN refinement** (simple, NOT full progressive TIN densification): a
   Delaunay TIN (triangulated irregular network) is built over the ground pixels;
   every non-ground pixel lying within ``readmit_within_m`` of it is re-admitted
   as ground; repeat until nothing changes or ``max_iterations`` is reached. This
   recovers terrain the morphological filter flagged wrongly (e.g. at sloped
   borders). It only ever re-admits; it does not remove ground pixels. Axelsson's
   PTD (seed minima, angle and distance criteria, iterative insertion) and cloth
   simulation are **not** implemented.
3. **Non-ground pixels inside the TIN's convex hull** get the TIN's linear
   interpolation: a convex combination of three ground pixels. It reproduces a
   planar terrain exactly and can never overshoot the ground values used.
4. **Non-ground pixels outside the hull** (objects touching the raster border,
   so no ground surrounds them) get the value of the **nearest ground pixel**.
   This is extrapolation; on terrain of slope ``s`` its error is at most
   ``s * distance``, and every such pixel is flagged ``extrapolated``.
5. **Invalid DSM pixels stay invalid** (NaN in memory, nodata on disk). The DTM
   is not guessed where there is no DSM observation.

nDSM and negative values
------------------------
``nDSM = DSM - DTM`` wherever both are valid; invalid elsewhere. On ground
pixels it is exactly 0 by construction. **Negative values are preserved, never
clamped**, and counted: they can only arise where a non-ground pixel lies below
the interpolated ground (a misclassification, a depression, or DSM noise), and
hiding them would hide that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

__all__ = ["DtmError", "DtmResult", "NdsmResult", "interpolate_dtm", "compute_ndsm"]


class DtmError(ValueError):
    """The DTM cannot be built from the given input."""


@dataclass(frozen=True)
class DtmResult:
    #: ``(H, W)`` float64 terrain elevation; NaN where the DSM was invalid.
    dtm: np.ndarray
    #: Final ground mask (after TIN refinement).
    ground: np.ndarray
    #: Non-ground pixels filled by linear TIN interpolation.
    interpolated: np.ndarray
    #: Non-ground pixels outside the TIN hull, filled from the nearest ground pixel.
    extrapolated: np.ndarray
    #: Distance (pixels) to the nearest ground pixel; 0 on ground, NaN where invalid.
    distance_to_ground_px: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NdsmResult:
    #: ``(H, W)`` float64 ``DSM - DTM``; NaN where either is invalid.
    ndsm: np.ndarray
    valid: np.ndarray
    negative_px: int
    min_m: float | None
    max_m: float | None

    def summary(self) -> dict[str, Any]:
        return {"definition": "nDSM = DSM - DTM", "valid_px": int(self.valid.sum()),
                "negative_px": self.negative_px, "negative_policy": "preserved, not clamped",
                "min_m": self.min_m, "max_m": self.max_m}


def _tin(dsm: np.ndarray, ground: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Linear Delaunay-TIN values at ``targets``; NaN outside the hull.

    Returns ``(values, inside, failure)``. Fewer than 3 ground pixels, or all of
    them collinear, cannot form a TIN: ``failure`` then says so and nothing is inside.
    """
    values = np.full(dsm.shape, np.nan)
    if not targets.any():
        return values, np.zeros(dsm.shape, bool), None
    points = np.argwhere(ground)
    if len(points) < 3:
        return values, np.zeros(dsm.shape, bool), f"only {len(points)} ground pixel(s); no TIN"
    try:
        interpolator = LinearNDInterpolator(points.astype(np.float64), dsm[ground])
    except QhullError as exc:
        return values, np.zeros(dsm.shape, bool), f"ground pixels do not span a TIN: {exc.__class__.__name__}"
    query = np.argwhere(targets)
    values[targets] = interpolator(query.astype(np.float64))
    return values, targets & np.isfinite(values), None


def interpolate_dtm(
    dsm: np.ndarray,
    ground: np.ndarray,
    valid: np.ndarray,
    *,
    readmit_within_m: float | None = None,
    max_iterations: int = 0,
) -> DtmResult:
    """Build the DTM: ground kept, TIN under objects, nearest ground outside the hull.

    Args:
        dsm: ``(H, W)`` DSM in metres.
        ground: Initial ground mask from a :class:`~depthwizard.surfaces.ground.GroundExtractor`.
        valid: DSM validity mask.
        readmit_within_m: TIN refinement tolerance; None disables refinement.
        max_iterations: Maximum refinement passes.

    Raises:
        DtmError: on a shape mismatch, or when no valid pixel is ground.
    """
    dsm = np.asarray(dsm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    ground = np.asarray(ground, dtype=bool)
    if not (dsm.shape == valid.shape == ground.shape) or dsm.ndim != 2:
        raise DtmError(f"shape mismatch: dsm {dsm.shape}, ground {ground.shape}, valid {valid.shape}")
    valid = valid & np.isfinite(dsm)
    ground = ground & valid
    if not ground.any():
        raise DtmError("no valid pixel is ground; the DTM is unconstrained")

    readmitted_per_pass: list[int] = []
    iterations = max_iterations if readmit_within_m is not None else 0
    failure = None
    while True:
        tin, inside, failure = _tin(dsm, ground, valid & ~ground)
        if len(readmitted_per_pass) >= iterations:
            break
        readmit = inside & (np.abs(dsm - tin) <= float(readmit_within_m))
        readmitted_per_pass.append(int(readmit.sum()))
        if not readmit.any():
            break
        ground = ground | readmit

    non_ground = valid & ~ground
    distance, (rows, cols) = ndimage.distance_transform_edt(~ground, return_indices=True)
    extrapolated = non_ground & ~inside
    dtm = np.full(dsm.shape, np.nan)
    dtm[ground] = dsm[ground]
    dtm[inside] = tin[inside]
    dtm[extrapolated] = dsm[rows[extrapolated], cols[extrapolated]]
    distance = np.where(valid, distance, np.nan)
    return DtmResult(
        dtm=dtm,
        ground=ground,
        interpolated=inside,
        extrapolated=extrapolated,
        distance_to_ground_px=distance,
        diagnostics={
            "ground_rule": "ground pixels keep their DSM value",
            "interpolation": "linear Delaunay TIN over ground pixels (inside the convex hull)",
            "extrapolation": "nearest ground pixel (outside the convex hull)",
            "tin_refinement": {
                "enabled": readmit_within_m is not None and max_iterations > 0,
                "readmit_within_m": readmit_within_m,
                "max_iterations": max_iterations,
                "readmitted_per_pass": readmitted_per_pass,
                "note": "simple re-admission only; NOT Axelsson progressive TIN densification, NOT cloth simulation",
            },
            "tin_failure": failure,
            "ground_px": int(ground.sum()),
            "interpolated_px": int(inside.sum()),
            "extrapolated_px": int(extrapolated.sum()),
            "max_extrapolation_distance_px": float(distance[extrapolated].max()) if extrapolated.any() else 0.0,
            "invalid_px": int((~valid).sum()),
        },
    )


def compute_ndsm(dsm: np.ndarray, dtm: np.ndarray) -> NdsmResult:
    """``nDSM = DSM - DTM`` where both are finite; NaN elsewhere. Negatives are kept."""
    dsm = np.asarray(dsm, dtype=np.float64)
    dtm = np.asarray(dtm, dtype=np.float64)
    if dsm.shape != dtm.shape:
        raise DtmError(f"DSM {dsm.shape} and DTM {dtm.shape} are not on the same grid")
    valid = np.isfinite(dsm) & np.isfinite(dtm)
    ndsm = np.full(dsm.shape, np.nan)
    ndsm[valid] = dsm[valid] - dtm[valid]
    values = ndsm[valid]
    return NdsmResult(
        ndsm=ndsm,
        valid=valid,
        negative_px=int((values < 0.0).sum()),
        min_m=float(values.min()) if values.size else None,
        max_m=float(values.max()) if values.size else None,
    )
