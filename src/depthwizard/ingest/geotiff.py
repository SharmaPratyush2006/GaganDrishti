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
    "CogError",
    "COG_OPTIONS",
    "write_cog",
    "validate_cog",
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


class CogError(RuntimeError):
    """A Cloud Optimized GeoTIFF could not be written, or failed validation."""


#: COG creation options. DEFLATE with the floating-point predictor is lossless;
#: 256 px internal tiles; overviews built automatically (GDAL's COG driver adds
#: levels until the smallest fits in one tile), averaged, nodata-aware.
COG_OPTIONS: dict[str, str] = {
    "COMPRESS": "DEFLATE",
    "PREDICTOR": "YES",
    "BLOCKSIZE": "256",
    "OVERVIEWS": "AUTO",
    "RESAMPLING": "AVERAGE",
}


def write_cog(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: str | CRS,
    transform: Affine,
    nodata: float,
    tags: Mapping[str, Any] | None = None,
    options: Mapping[str, str] | None = None,
) -> Path:
    """Write ``array`` as a single-band Cloud Optimized GeoTIFF.

    The array is first written to an in-memory GeoTIFF, then copied with GDAL's
    ``COG`` driver (a CreateCopy-only driver), which lays out tiles, overviews
    and headers as the COG specification requires. NaN pixels are written as
    ``nodata``; ``nodata`` must be finite and must not collide with a valid value.

    Raises:
        CogError: on invalid input or any GDAL write failure.
    """
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy as rio_copy

    array = np.asarray(array)
    if array.ndim != 2 or 0 in array.shape:
        raise CogError(f"expected a non-empty 2-D array, got shape {array.shape}")
    if not np.isfinite(nodata):
        raise CogError(f"nodata must be a finite number, got {nodata}")
    data = array.copy()
    if np.issubdtype(data.dtype, np.floating):
        invalid = ~np.isfinite(data)
        if np.any(data[~invalid] == nodata):
            raise CogError(f"nodata {nodata} collides with a valid pixel value")
        data[invalid] = nodata
    if not crs:
        raise CogError("a COG must carry a CRS")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
        "dtype": data.dtype.name, "crs": CRS.from_user_input(crs), "transform": transform, "nodata": nodata,
    }
    try:
        with MemoryFile() as memory:
            with memory.open(**profile) as dst:
                dst.write(data, 1)
                if tags:
                    dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
            with memory.open() as src:
                rio_copy(src, path, driver="COG", **dict(options or COG_OPTIONS))
    except Exception as exc:
        raise CogError(f"failed to write COG {path}: {exc}") from exc
    return path


def validate_cog(
    path: str | Path,
    *,
    crs: str | CRS,
    transform: Affine,
    width: int,
    height: int,
    nodata: float,
    dtype: str,
    expected: np.ndarray | None = None,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Reopen a COG and check it against what was meant to be written.

    Checks: COG layout, internal tiling, overviews (when the raster is larger
    than one tile), CRS, transform, dimensions, bounds, nodata, dtype,
    readability, and -- given ``expected`` (NaN = nodata) -- the pixel values
    within ``atol`` and the nodata mask exactly.

    Returns:
        The validation record.

    Raises:
        CogError: listing every failed check.
    """
    path = Path(path)
    failures: list[str] = []
    try:
        with rasterio.open(path) as src:
            layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
            block_shapes = src.block_shapes
            overviews = src.overviews(1)
            record: dict[str, Any] = {
                "path": str(path),
                "driver": src.driver,
                "layout": layout,
                "compression": src.tags(ns="IMAGE_STRUCTURE").get("COMPRESSION"),
                "tiled": bool(src.profile.get("tiled")),
                "block_shape": list(block_shapes[0]),
                "overview_factors": list(overviews),
                "crs": src.crs.to_string() if src.crs else None,
                "transform": [float(v) for v in tuple(src.transform)[:6]],
                "width": src.width,
                "height": src.height,
                "bounds": list(src.bounds),
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
            }
            data = src.read(1)
            ov_data = src.read(1, out_shape=(max(1, src.height // overviews[-1]), max(1, src.width // overviews[-1]))) if overviews else None
    except Exception as exc:
        raise CogError(f"COG {path} could not be reopened: {exc}") from exc

    if layout != "COG":
        failures.append(f"layout is {layout!r}, not 'COG'")
    if not record["tiled"]:
        failures.append("not internally tiled")
    block = int(block_shapes[0][0])
    if max(width, height) > block and not overviews:
        failures.append(f"no overviews on a {width}x{height} raster with {block} px tiles")
    if not record["crs"] or CRS.from_user_input(record["crs"]) != CRS.from_user_input(crs):
        failures.append(f"CRS {record['crs']} != expected {crs}")
    if not np.allclose(record["transform"], tuple(transform)[:6], rtol=0.0, atol=1e-9):
        failures.append(f"transform {record['transform']} != expected {tuple(transform)[:6]}")
    if (record["width"], record["height"]) != (width, height):
        failures.append(f"dimensions {record['width']}x{record['height']} != expected {width}x{height}")
    corners = [pixel_to_map(transform, c, r) for c, r in ((0, 0), (width, 0), (0, height), (width, height))]
    exp_bounds = [min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners)]
    if not np.allclose(record["bounds"], exp_bounds, rtol=0.0, atol=1e-6):
        failures.append(f"bounds {record['bounds']} != expected {exp_bounds}")
    if record["nodata"] is None or record["nodata"] != nodata:
        failures.append(f"nodata {record['nodata']} != expected {nodata}")
    if record["dtype"] != np.dtype(dtype).name:
        failures.append(f"dtype {record['dtype']} != expected {np.dtype(dtype).name}")
    if ov_data is not None and ov_data.size == 0:
        failures.append("overview is not readable")

    if expected is not None:
        expected = np.asarray(expected, dtype=np.float64)
        written_invalid = data == nodata
        expected_invalid = ~np.isfinite(expected)
        if not np.array_equal(written_invalid, expected_invalid):
            failures.append(f"nodata mask differs in {int((written_invalid != expected_invalid).sum())} pixel(s)")
        both = ~written_invalid & ~expected_invalid
        max_diff = float(np.abs(data[both].astype(np.float64) - expected[both]).max()) if both.any() else 0.0
        record["max_abs_diff_vs_memory"] = max_diff
        if max_diff > atol:
            failures.append(f"pixel values differ from memory by up to {max_diff} (> atol {atol})")
    record["valid_pixels"] = int((data != nodata).sum())
    record["nodata_pixels"] = int((data == nodata).sum())
    record["checks_failed"] = failures
    record["ok"] = not failures
    if failures:
        raise CogError(f"COG {path} failed validation: " + "; ".join(failures))
    return record


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
