"""Input loaders.

Two loaders, one output shape.

:class:`GeoTIFFLoader`
    Reads a georeferenced raster: CRS, geotransform, GSD in metres/pixel, and
    sun elevation/azimuth discovered through :mod:`depthwizard.ingest.metadata`.
    A file that carries all of it is :attr:`Mode.ABSOLUTE`.

:class:`ImageLoader`
    Reads a plain PNG/JPG. It deliberately does **not** look for world files,
    EXIF GPS or any other georeferencing: a plain image has no scale and no
    illumination geometry, so it is always :attr:`Mode.RELATIVE`.

Both return a :class:`LoadedScene` whose ``array`` is ``(rows, cols, bands)``,
so everything downstream -- normalisation, tiling -- has one shape to handle
regardless of where the pixels came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import rasterio

from depthwizard.ingest.metadata import (
    MissingMetadataError,
    SceneMetadata,
    SUN_AZIMUTH_ALIASES,
    SUN_ELEVATION_ALIASES,
    SUN_ZENITH_ALIASES,
    UnsupportedInputError,
    collect_candidates,
    read_georeference,
    read_rpc,
    resolve_sun,
)
from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = [
    "LoadedScene",
    "GeoTIFFLoader",
    "ImageLoader",
    "GEOTIFF_SUFFIXES",
    "IMAGE_SUFFIXES",
]

log = get_logger(__name__)

GEOTIFF_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff", ".gtif", ".gtiff"})
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


@dataclass(frozen=True)
class LoadedScene:
    """Pixels plus the metadata that describes them."""

    array: np.ndarray  # (rows, cols, bands)
    metadata: SceneMetadata

    @property
    def mode(self) -> Mode:
        return self.metadata.mode

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def band_count(self) -> int:
        return int(self.array.shape[2])

    def band(self, index: int) -> np.ndarray:
        """Return one band as a 2-D array."""
        return self.array[:, :, index]


def _to_hwc(array: np.ndarray) -> np.ndarray:
    """Normalise any 2-D or 3-D array to ``(rows, cols, bands)``."""
    if array.ndim == 2:
        return array[:, :, np.newaxis]
    if array.ndim == 3:
        return array
    raise UnsupportedInputError(f"expected a 2-D or 3-D array, got shape {array.shape}")


# ---------------------------------------------------------------------------
# GeoTIFF
# ---------------------------------------------------------------------------


class GeoTIFFLoader:
    """Load a georeferenced raster and everything known about its geometry.

    Args:
        require_absolute: When True (the default), a file that cannot reach
            :attr:`Mode.ABSOLUTE` raises :class:`MissingMetadataError` rather
            than quietly degrading. The router sets this False so it can make
            the downgrade explicit and logged instead.
    """

    suffixes = GEOTIFF_SUFFIXES

    def __init__(self, *, require_absolute: bool = True) -> None:
        self.require_absolute = require_absolute

    @classmethod
    def handles(cls, path: str | Path) -> bool:
        return Path(path).suffix.lower() in cls.suffixes

    def read_metadata(self, path: str | Path) -> SceneMetadata:
        """Probe a file's metadata without reading any pixels."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"raster not found: {path}")

        with rasterio.open(path) as src:
            candidates = collect_candidates(src, path)
            georeference = read_georeference(src)
            sun = resolve_sun(candidates, path)
            rpc = read_rpc(candidates)
            driver = src.driver
            width, height, count = src.width, src.height, src.count
            dtype = src.dtypes[0]

        probed = candidates.source_labels
        reason: str | None = None
        if georeference is None:
            reason = (
                "raster has no CRS or only an identity transform, so pixel size "
                "cannot be expressed in metres"
            )
        elif sun is None:
            reason = (
                "raster is georeferenced but no sun elevation/azimuth was found "
                "in any probed source"
            )

        mode = Mode.ABSOLUTE if reason is None else Mode.RELATIVE

        metadata = SceneMetadata(
            path=path,
            mode=mode,
            driver=driver,
            width=width,
            height=height,
            band_count=count,
            dtype=dtype,
            georeference=georeference,
            sun=sun,
            rpc=rpc,
            probed_sources=probed,
            downgrade_reason=reason,
        )

        if self.require_absolute and mode is not Mode.ABSOLUTE:
            missing: list[str] = []
            if georeference is None:
                missing.append("crs/geotransform")
            if sun is None:
                missing += ["sun_elevation_deg", "sun_azimuth_deg"]
            raise MissingMetadataError(
                missing=missing,
                probed_sources=probed,
                tried_keys={
                    "sun_elevation_deg": SUN_ELEVATION_ALIASES + SUN_ZENITH_ALIASES,
                    "sun_azimuth_deg": SUN_AZIMUTH_ALIASES,
                },
                path=path,
                hint=(
                    "Supply the angles as GeoTIFF tags (SUN_ELEVATION_DEG / "
                    "SUN_AZIMUTH_DEG), place the provider's .IMD sidecar next to "
                    "the file, or load in relative mode with "
                    "GeoTIFFLoader(require_absolute=False)."
                ),
            )
        return metadata

    def load(self, path: str | Path, bands: Sequence[int] | None = None) -> LoadedScene:
        """Read pixels and metadata.

        Args:
            path: Raster to read.
            bands: 1-based band indices. Defaults to every band.
        """
        metadata = self.read_metadata(path)
        with rasterio.open(path) as src:
            indexes = list(bands) if bands is not None else list(range(1, src.count + 1))
            array = src.read(indexes)  # (bands, rows, cols)
        array = np.transpose(array, (1, 2, 0))  # -> (rows, cols, bands)

        log.info(
            "loaded geotiff",
            extra={
                "path": str(path),
                "mode": metadata.mode.value,
                "size": f"{metadata.width}x{metadata.height}",
                "bands": array.shape[2],
                "gsd_m": metadata.georeference.gsd_x_m if metadata.georeference else None,
            },
        )
        return LoadedScene(array=array, metadata=metadata)


# ---------------------------------------------------------------------------
# Plain images
# ---------------------------------------------------------------------------


class ImageLoader:
    """Load a PNG/JPG with no geospatial assumptions at all.

    These inputs always route to :attr:`Mode.RELATIVE`. World files, EXIF GPS
    and filename conventions are deliberately ignored: guessing a scale from
    any of them is exactly the kind of invented metadata this phase forbids.
    """

    suffixes = IMAGE_SUFFIXES

    @classmethod
    def handles(cls, path: str | Path) -> bool:
        return Path(path).suffix.lower() in cls.suffixes

    def read_metadata(self, path: str | Path) -> SceneMetadata:
        """Read the header only, by decoding and discarding the pixels."""
        return self.load(path).metadata

    def load(self, path: str | Path) -> LoadedScene:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")

        # np.fromfile + imdecode rather than cv2.imread: imread cannot open
        # non-ASCII paths on Windows and returns None with no error.
        raw = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise UnsupportedInputError(
                f"could not decode {path} as an image; supported suffixes are "
                f"{sorted(IMAGE_SUFFIXES)}"
            )

        array = _to_hwc(decoded)
        note = None
        if array.shape[2] == 4:
            array = array[:, :, :3]
            note = "dropped alpha channel"
        if array.shape[2] == 3:
            # OpenCV decodes to BGR; store RGB so band order is predictable.
            array = array[:, :, ::-1]
        array = np.ascontiguousarray(array)

        metadata = SceneMetadata(
            path=path,
            mode=Mode.RELATIVE,
            driver=path.suffix.lstrip(".").upper(),
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            band_count=int(array.shape[2]),
            dtype=str(array.dtype),
            georeference=None,
            sun=None,
            rpc=None,
            probed_sources=("image file (no geospatial metadata by design)",),
            downgrade_reason=(
                "plain image input: no CRS, no geotransform and no sun geometry, "
                "so only relative heights are recoverable"
            ),
        )

        log.info(
            "loaded image",
            extra={
                "path": str(path),
                "mode": metadata.mode.value,
                "size": f"{metadata.width}x{metadata.height}",
                "bands": metadata.band_count,
                "note": note,
            },
        )
        return LoadedScene(array=array, metadata=metadata)
