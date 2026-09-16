"""GeoTIFF read/write helpers.

A thin wrapper over rasterio that keeps two things consistent across the whole
project:

1. Every raster DepthWizard writes is a north-up, single-band GeoTIFF with an
   explicit CRS and transform.
2. Illumination geometry travels *with* the pixels, as GeoTIFF metadata tags,
   so a ``.tif`` is self-describing -- you never have to remember which sun
   angle produced which file.

Tag keys are defined once in :class:`MetadataTags`. GeoTIFF stores all tags as
strings, so :func:`read_sun_metadata` handles parsing them back to floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from depthwizard.config import RasterConfig, SunConfig

__all__ = [
    "MetadataTags",
    "RasterReadResult",
    "pixel_to_map",
    "translate_transform",
    "transform_from_raster_config",
    "write_single_band",
    "read_raster",
    "read_sun_metadata",
]


class MetadataTags:
    """Canonical GeoTIFF tag keys written by DepthWizard."""

    SUN_ELEVATION = "SUN_ELEVATION_DEG"
    SUN_AZIMUTH = "SUN_AZIMUTH_DEG"
    GSD = "GSD_M"
    SCENE_NAME = "SCENE_NAME"
    SYNTHETIC = "SYNTHETIC"
    PRODUCER = "PRODUCER"
    VERSION = "DEPTHWIZARD_VERSION"
    RUN_ID = "RUN_ID"
    AZIMUTH_CONVENTION = "AZIMUTH_CONVENTION"

    #: Recorded verbatim in every file so the convention can never be guessed wrong.
    AZIMUTH_CONVENTION_VALUE = "degrees clockwise from North (0=N, 90=E, 180=S, 270=W)"


@dataclass(frozen=True)
class RasterReadResult:
    """Everything Phase 0 needs to verify a raster that was written back out."""

    array: np.ndarray
    crs: CRS
    transform: Affine
    tags: dict[str, str]
    nodata: float | None
    width: int
    height: int

    @property
    def gsd_m(self) -> float:
        """Pixel size in metres, read from the affine transform.

        Raises ``ValueError`` if the pixel is not square, since the shadow
        physics assumes a single ground sample distance.
        """
        x_size = abs(self.transform.a)
        y_size = abs(self.transform.e)
        if not np.isclose(x_size, y_size):
            raise ValueError(f"non-square pixels: x={x_size}, y={y_size}")
        return float(x_size)

    @property
    def is_north_up(self) -> bool:
        """True when rows run north -> south and columns run west -> east."""
        return self.transform.a > 0 and self.transform.e < 0 and self.transform.b == 0 and self.transform.d == 0


def pixel_to_map(transform: Affine, col: float, row: float) -> tuple[float, float]:
    """Map a (possibly fractional) pixel index to map coordinates.

    Applies the affine directly rather than using ``transform * (col, row)``,
    which is deprecated in recent versions of the ``affine`` package.

    Args:
        transform: The raster's affine transform.
        col: Column index; 0 is the left edge of the leftmost pixel.
        row: Row index; 0 is the top edge of the topmost pixel.

    Returns:
        ``(easting, northing)`` in the raster's CRS.
    """
    easting = transform.c + col * transform.a + row * transform.b
    northing = transform.f + col * transform.d + row * transform.e
    return float(easting), float(northing)


def translate_transform(transform: Affine, col_off: float, row_off: float) -> Affine:
    """Shift an affine transform's origin to a pixel offset within the raster.

    Used to give each tile its own georeferencing: a tile starting at
    ``(row_off, col_off)`` has the same pixel size and rotation as its parent,
    with the origin moved to that pixel's corner. Written out explicitly rather
    than as ``transform * Affine.translation(...)``, which is deprecated in
    recent versions of the ``affine`` package.
    """
    easting, northing = pixel_to_map(transform, col_off, row_off)
    return Affine(
        transform.a, transform.b, easting,
        transform.d, transform.e, northing,
    )


def transform_from_raster_config(raster: RasterConfig) -> Affine:
    """Build a north-up affine transform from a :class:`RasterConfig`.

    Equivalent to ``rasterio.transform.from_origin``, written out explicitly:
    columns step East by one GSD, rows step South by one GSD, and ``(c, f)`` is
    the north-west corner of the north-west pixel.
    """
    return Affine(
        raster.gsd_m, 0.0, raster.origin_easting_m,
        0.0, -raster.gsd_m, raster.origin_northing_m,
    )


def write_single_band(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: str | CRS,
    transform: Affine,
    tags: Mapping[str, Any] | None = None,
    nodata: float | None = None,
    compress: str = "deflate",
) -> Path:
    """Write ``array`` as a single-band GeoTIFF and return the path.

    Args:
        path: Destination ``.tif`` path. Parent directories are created.
        array: 2-D array. Its dtype becomes the band dtype.
        crs: Coordinate reference system, e.g. ``"EPSG:32643"``.
        transform: Affine transform mapping pixel -> map coordinates.
        tags: Extra metadata tags. Values are stringified by GDAL.
        nodata: Optional nodata value.
        compress: GeoTIFF compression scheme.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype.name,
        "crs": CRS.from_user_input(crs) if isinstance(crs, str) else crs,
        "transform": transform,
        "compress": compress,
        "tiled": False,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    return path


def read_raster(path: str | Path, band: int = 1) -> RasterReadResult:
    """Read a single band plus its georeferencing and metadata tags."""
    path = Path(path)
    with rasterio.open(path) as src:
        return RasterReadResult(
            array=src.read(band),
            crs=src.crs,
            transform=src.transform,
            tags=dict(src.tags()),
            nodata=src.nodata,
            width=src.width,
            height=src.height,
        )


def read_sun_metadata(path: str | Path) -> SunConfig:
    """Reconstruct the illumination geometry stored in a GeoTIFF's tags.

    Raises:
        KeyError: if the file carries no DepthWizard sun tags.
    """
    with rasterio.open(Path(path)) as src:
        tags = src.tags()
    missing = [k for k in (MetadataTags.SUN_ELEVATION, MetadataTags.SUN_AZIMUTH) if k not in tags]
    if missing:
        raise KeyError(f"{path}: missing sun metadata tag(s) {missing}")
    return SunConfig(
        elevation_deg=float(tags[MetadataTags.SUN_ELEVATION]),
        azimuth_deg=float(tags[MetadataTags.SUN_AZIMUTH]),
    )
