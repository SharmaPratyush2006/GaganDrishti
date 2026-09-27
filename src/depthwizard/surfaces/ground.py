"""Phase 6: ground extraction from a DSM -- which pixels are bare ground?

Input is a **DSM only** (absolute surface elevation, metres). Nothing here reads
the Phase 3/4a AGL, a DEM, or a semantic label: the ground mask is *inferred*
from surface shape alone, which is what makes Phase 6 independent of the rest
of the pipeline.

The interface
-------------
:class:`GroundExtractor` is the swappable part: ``extract(dsm, valid=, gsd_m=)``
returns a :class:`GroundResult` whose ``ground`` mask marks pixels judged to be
bare ground. Everything downstream (:mod:`depthwizard.surfaces.dtm`) only needs
that mask, so a cloth-simulation filter or a learned classifier could replace
the one below without touching a call site.

The implemented method: a progressive morphological filter
----------------------------------------------------------
:class:`MorphologicalGroundExtractor` follows Zhang et al. (2003, IEEE TGRS
41(4)). A grey **opening** (erosion, then dilation) with a flat square window
removes every raised object the window cannot fit inside, and leaves terrain
that varies more slowly than the window. Windows grow progressively::

    surface_0 = DSM
    surface_k = opening(surface_{k-1}, w_k)          w_k = 3, 5, 9, 17, 33, ... px, then w_max
    pixel is NON-GROUND if  surface_{k-1} - surface_k > dh_k   at any k
    dh_k = min_object_height_m + max_terrain_slope * (w_k * GSD)

Every parameter is in **metres** and converted with the raster's own GSD:

``max_building_extent_m``
    The largest horizontal extent (axis-aligned) of any building. The final
    window is the smallest odd pixel count **strictly larger** than it, so no
    building can contain the window and every building is opened away -- the
    master prompt's "structuring element larger than the biggest building
    footprint". A building larger than this survives the opening and is
    (wrongly) kept as ground. There is no default: it must be stated.
``max_terrain_slope``
    Rise over run the bare terrain may have. Across a window of ``w`` metres,
    terrain alone can make the opening differ from the surface by up to about
    ``slope * w``; the threshold grows with the window so sloped ground is not
    flagged.
``min_object_height_m``
    A height step smaller than this (above what the slope allows) is treated as
    ground. Objects lower than it (cars, low walls) are ground by design.

Scene borders use ``mode='reflect'``. Reflection turns a planar slope into a
ridge or valley at the border; the slope term in ``dh_k`` covers that. A
building *touching* the border is reflected into one twice as deep, so it is
only removed if the window exceeds twice its extent perpendicular to that
border -- a documented limitation.

Invalid (nodata / non-finite) pixels are filled with their nearest valid value
for filtering only; they are never reported as ground.

This is a classical height-and-shape filter. It is **not** a building
detector: it flags anything raised (trees, bridges, vehicles above the height
threshold) as non-ground, and it cannot see objects larger than
``max_building_extent_m``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

__all__ = [
    "GroundExtractionError",
    "GroundResult",
    "GroundExtractor",
    "MorphologicalGroundExtractor",
    "fill_invalid_nearest",
]


class GroundExtractionError(ValueError):
    """Ground cannot be extracted with the given input or parameters."""


@dataclass(frozen=True)
class GroundResult:
    #: ``(H, W)`` bool: pixel inferred to be bare ground (never True where invalid).
    ground: np.ndarray
    #: ``(H, W)`` bool: pixel had a finite, non-nodata DSM value.
    valid: np.ndarray
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def non_ground(self) -> np.ndarray:
        return self.valid & ~self.ground


class GroundExtractor(ABC):
    """Swappable ground classifier: DSM in, ground mask out."""

    name: str = "abstract"

    @abstractmethod
    def extract(self, dsm: np.ndarray, *, valid: np.ndarray, gsd_m: float) -> GroundResult:
        """Classify every valid pixel of ``dsm`` as ground or non-ground."""


def fill_invalid_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of ``values`` with every invalid pixel set to its nearest valid value.

    Raises:
        GroundExtractionError: if no pixel is valid.
    """
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise GroundExtractionError("the DSM has no valid pixel")
    if valid.all():
        return np.asarray(values, dtype=np.float64).copy()
    _, (rows, cols) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return np.asarray(values, dtype=np.float64)[rows, cols]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise GroundExtractionError(f"{name} must be finite and > 0, got {value}")
    return value


class MorphologicalGroundExtractor(GroundExtractor):
    """Progressive morphological filter (Zhang et al. 2003). See the module docstring."""

    name = "progressive_morphological_filter"

    def __init__(self, *, max_building_extent_m: float, max_terrain_slope: float, min_object_height_m: float) -> None:
        if max_building_extent_m is None:
            raise GroundExtractionError(
                "max_building_extent_m is required: the final opening window must exceed the largest "
                "building, and there is no safe default"
            )
        self.max_building_extent_m = _positive(max_building_extent_m, "max_building_extent_m")
        self.max_terrain_slope = _positive(max_terrain_slope, "max_terrain_slope")
        self.min_object_height_m = _positive(min_object_height_m, "min_object_height_m")

    def window_schedule(self, gsd_m: float) -> list[int]:
        """Odd window sizes in pixels: 3, 5, 9, 17, ... then the final window.

        The final window is the smallest odd pixel count strictly larger than
        ``max_building_extent_m / gsd_m``.
        """
        gsd_m = _positive(gsd_m, "gsd_m")
        extent_px = self.max_building_extent_m / gsd_m
        final = int(math.floor(extent_px)) + 1
        final += (final + 1) % 2  # make it odd, still strictly larger
        windows, k = [], 1
        while 2**k + 1 < final:
            windows.append(2**k + 1)
            k += 1
        windows.append(max(final, 3))
        return windows

    def threshold_m(self, window_px: int, gsd_m: float) -> float:
        """``dh_k = min_object_height_m + max_terrain_slope * window_metres``."""
        return self.min_object_height_m + self.max_terrain_slope * window_px * gsd_m

    def extract(self, dsm: np.ndarray, *, valid: np.ndarray | None = None, gsd_m: float) -> GroundResult:
        dsm = np.asarray(dsm, dtype=np.float64)
        if dsm.ndim != 2:
            raise GroundExtractionError(f"DSM must be 2-D, got shape {dsm.shape}")
        valid = np.isfinite(dsm) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(dsm))
        if valid.shape != dsm.shape:
            raise GroundExtractionError(f"valid mask {valid.shape} does not match DSM {dsm.shape}")
        surface = fill_invalid_nearest(dsm, valid)
        non_ground = np.zeros(dsm.shape, dtype=bool)
        stages = []
        for window in self.window_schedule(gsd_m):
            opened = ndimage.grey_opening(surface, size=(window, window), mode="reflect")
            threshold = self.threshold_m(window, gsd_m)
            flagged = (surface - opened) > threshold
            newly = flagged & valid & ~non_ground
            non_ground |= flagged
            stages.append({"window_px": window, "window_m": window * gsd_m, "threshold_m": threshold,
                           "newly_flagged_px": int(newly.sum())})
            surface = opened
        ground = valid & ~non_ground
        return GroundResult(
            ground=ground,
            valid=valid,
            method=self.name,
            diagnostics={
                "method": self.name,
                "reference": "Zhang et al. 2003, IEEE TGRS 41(4): progressive morphological filter",
                "gsd_m": float(gsd_m),
                "max_building_extent_m": self.max_building_extent_m,
                "max_terrain_slope": self.max_terrain_slope,
                "min_object_height_m": self.min_object_height_m,
                "border_mode": "reflect",
                "stages": stages,
                "valid_px": int(valid.sum()),
                "ground_px": int(ground.sum()),
                "non_ground_px": int((valid & ~ground).sum()),
            },
        )
