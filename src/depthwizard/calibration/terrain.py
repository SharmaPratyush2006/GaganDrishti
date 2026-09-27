"""Terrain (DEM) onto the image grid, for the georeferenced DSM (Phase 4b).

The DEM's only role is additive::

    DSM(x) = T(x) + AGL(x)

It is never a scale anchor for the relative field (Phase 3 predicts
above-ground height, which contains no terrain), and nothing is subtracted
from it.

The image grid is authoritative: CRS, transform, width and height of the
result are exactly the grid's. The DEM is reprojected onto it with
:func:`rasterio.warp.reproject` and **bilinear** resampling (terrain is a
continuous field). Pixels the DEM does not cover, and pixels whose resampling
kernel touches a DEM nodata sample (where GDAL would otherwise renormalise
over the valid neighbours, i.e. extrapolate), come back as NaN in ``values``
and False in ``valid`` -- never as 0, which is a legitimate elevation.

Every precondition fails loudly with :class:`TerrainError`: a missing CRS on
either side, a missing (identity) transform, empty dimensions, a CRS pair that
cannot be transformed, no spatial overlap, or no valid terrain pixel after
reprojection.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError, NotGeoreferencedWarning
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

from depthwizard.ingest.geotiff import pixel_to_map

__all__ = ["TerrainError", "GridSpec", "TerrainOnGrid", "load_terrain_on_grid"]


class TerrainError(RuntimeError):
    """The DEM cannot be placed on the image grid; the message says why."""


@dataclass(frozen=True)
class GridSpec:
    """The authoritative output grid (normally the image's)."""

    crs: Any
    transform: Affine | None
    width: int
    height: int

    @classmethod
    def from_raster(cls, path: str | Path) -> "GridSpec":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                return cls(crs=src.crs, transform=src.transform, width=int(src.width), height=int(src.height))

    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` from all four corners (rotation-safe)."""
        corners = [pixel_to_map(self.transform, c, r) for c, r in
                   ((0, 0), (self.width, 0), (0, self.height), (self.width, self.height))]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": CRS.from_user_input(self.crs).to_string() if self.crs else None,
            "transform": [float(v) for v in tuple(self.transform)[:6]] if self.transform else None,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class TerrainOnGrid:
    #: ``(height, width)`` float64 terrain in metres; NaN where invalid.
    values: np.ndarray
    valid: np.ndarray
    grid: GridSpec
    provenance: dict[str, Any]


def _check_georeferenced(what: str, crs: Any, transform: Affine | None, width: int, height: int) -> CRS:
    if not crs:
        raise TerrainError(f"{what} has no CRS; refusing to guess where it lies on the Earth")
    try:
        parsed = CRS.from_user_input(crs)
    except CRSError as exc:
        raise TerrainError(f"{what} CRS {crs!r} is not a valid CRS: {exc}") from exc
    if transform is None or tuple(transform)[:6] == tuple(Affine.identity())[:6]:
        raise TerrainError(
            f"{what} has no spatial transform (identity or missing); pixel positions are unknown"
        )
    if abs(transform.a * transform.e - transform.b * transform.d) == 0.0:
        raise TerrainError(f"{what} transform is degenerate (zero pixel area): {transform}")
    if int(width) <= 0 or int(height) <= 0:
        raise TerrainError(f"{what} has invalid dimensions {width}x{height}")
    return parsed


def load_terrain_on_grid(
    dem_path: str | Path,
    grid: GridSpec,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> TerrainOnGrid:
    """Reproject a DEM onto ``grid``. See the module docstring for the rules.

    Raises:
        TerrainError: on any failed precondition (never silently repaired).
    """
    dem_path = Path(dem_path)
    if not dem_path.is_file():
        raise TerrainError(f"DEM file not found: {dem_path}")
    dst_crs = _check_georeferenced("image grid", grid.crs, grid.transform, grid.width, grid.height)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(dem_path) as src:
            src_crs = _check_georeferenced(f"DEM {dem_path.name}", src.crs, src.transform, src.width, src.height)
            dem = src.read(1).astype(np.float64)
            src_transform, src_nodata = src.transform, src.nodata
            src_width, src_height = int(src.width), int(src.height)

    # Overlap, in the image CRS.
    dem_grid = GridSpec(src_crs, src_transform, src_width, src_height)
    try:
        dem_bounds = transform_bounds(src_crs, dst_crs, *dem_grid.bounds(), densify_pts=21)
    except Exception as exc:  # rasterio raises several error types for a failed transformation
        raise TerrainError(f"cannot transform DEM CRS {src_crs.to_string()} to image CRS {dst_crs.to_string()}: {exc}") from exc
    if not all(math.isfinite(v) for v in dem_bounds):
        raise TerrainError(f"DEM bounds are not representable in image CRS {dst_crs.to_string()}: {dem_bounds}")
    left, bottom, right, top = grid.bounds()
    d_left, d_bottom, d_right, d_top = dem_bounds
    if d_right <= left or d_left >= right or d_top <= bottom or d_bottom >= top:
        raise TerrainError(
            f"DEM does not overlap the image grid: DEM bounds in {dst_crs.to_string()} "
            f"{tuple(round(v, 3) for v in dem_bounds)} vs image {(left, bottom, right, top)}"
        )

    # Mask the source nodata explicitly (NaN), so it can never be averaged in.
    if src_nodata is not None and not math.isnan(src_nodata):
        dem = np.where(np.isclose(dem, src_nodata), np.nan, dem)
    dem[~np.isfinite(dem)] = np.nan

    values = np.full((grid.height, grid.width), np.nan, dtype=np.float64)
    reproject(
        source=dem,
        destination=values,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    # Near a nodata hole GDAL renormalises the kernel over the remaining valid
    # samples, which extrapolates. Reproject the source validity with the same
    # kernel: a target pixel is valid only if its whole kernel was valid
    # (coverage 1). Pixels outside the DEM stay at coverage 0.
    source_valid = np.isfinite(dem).astype(np.float64)
    coverage = np.zeros((grid.height, grid.width), dtype=np.float64)
    reproject(
        source=source_valid,
        destination=coverage,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    valid = np.isfinite(values) & (coverage >= 1.0 - 1e-9)
    values = np.where(valid, values, np.nan)
    if not valid.any():
        raise TerrainError("no valid terrain pixel on the image grid after reprojection")

    provenance = {
        "dem_path": str(dem_path),
        "source_crs": src_crs.to_string(),
        "target_crs": dst_crs.to_string(),
        "source_transform": [float(v) for v in tuple(src_transform)[:6]],
        "target_transform": [float(v) for v in tuple(grid.transform)[:6]],
        "source_dimensions": [src_height, src_width],
        "target_dimensions": [grid.height, grid.width],
        "resampling": resampling.name,
        "reprojected": src_crs != dst_crs,
        "source_nodata": src_nodata,
        "valid_pixels": int(valid.sum()),
        "nodata_pixels": int((~valid).sum()),
    }
    return TerrainOnGrid(values=values, valid=valid, grid=grid, provenance=provenance)
