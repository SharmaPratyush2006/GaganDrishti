"""Metadata discovery.

Satellite providers store the same handful of numbers under wildly different
names, in different places. The sun elevation might be ``SUN_ELEVATION`` in a
GDAL tag (Landsat), ``meanSunEl`` in a Maxar ``.IMD`` sidecar, or implied by
``MEAN_SUN_ZENITH_ANGLE`` (Sentinel-2, where elevation = 90 - zenith). This
module probes all of those places and resolves them to one canonical form.

Two rules govern everything here:

1. **Nothing is invented.** Every resolved value carries a :class:`Provenance`
   recording which source, which key, the raw string, and any conversion
   applied. If you cannot say where a number came from, it does not exist.
2. **Missing required metadata fails loudly.** :class:`MissingMetadataError`
   lists every source that was probed and every alias that was tried, so the
   error tells you what to fix rather than just that something is wrong.

Sources probed, in order:

* GDAL metadata domains (the default domain, plus ``IMD``, ``RPC``,
  ``IMAGERY``, ``EXIF``, ``TRE`` and any others the driver reports)
* RPC coefficients, via ``rasterio``'s parsed RPCs and the ``RPC`` tag domain
* ``.IMD`` sidecar files sitting next to the image
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from depthwizard.config import SunConfig
from depthwizard.ingest.geotiff import pixel_to_map
from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = [
    "MetadataError",
    "MissingMetadataError",
    "UnsupportedInputError",
    "Provenance",
    "SunMetadata",
    "GeoReference",
    "RpcInfo",
    "SceneMetadata",
    "MetadataCandidates",
    "collect_candidates",
    "parse_imd",
    "find_sidecars",
    "resolve_sun",
    "compute_gsd_m",
    "read_georeference",
    "read_rpc",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MetadataError(Exception):
    """Base class for metadata problems."""


class MissingMetadataError(MetadataError):
    """Required metadata could not be found in any probed source.

    Carries the full search record so the message is actionable.
    """

    def __init__(
        self,
        missing: Sequence[str],
        probed_sources: Sequence[str],
        tried_keys: Mapping[str, Sequence[str]],
        path: Path | None = None,
        hint: str | None = None,
    ) -> None:
        self.missing = tuple(missing)
        self.probed_sources = tuple(probed_sources)
        self.tried_keys = {k: tuple(v) for k, v in tried_keys.items()}
        self.path = path
        lines = [
            f"missing required metadata: {', '.join(self.missing)}"
            + (f" for {path}" if path else "")
        ]
        lines.append(
            "  probed sources: "
            + (", ".join(self.probed_sources) if self.probed_sources else "(none found)")
        )
        for field_name in self.missing:
            aliases = self.tried_keys.get(field_name, ())
            if aliases:
                lines.append(f"  keys tried for {field_name}: {', '.join(aliases)}")
        if hint:
            lines.append(f"  hint: {hint}")
        super().__init__("\n".join(lines))


class UnsupportedInputError(MetadataError):
    """The file is not an input DepthWizard knows how to read."""


# ---------------------------------------------------------------------------
# Value + provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a single metadata value came from, and what was done to it."""

    source: str
    key: str
    raw_value: str
    conversion: str | None = None

    def describe(self) -> str:
        text = f"{self.source}[{self.key}]={self.raw_value}"
        if self.conversion:
            text += f" ({self.conversion})"
        return text


@dataclass(frozen=True)
class SunMetadata:
    """Resolved illumination geometry, with provenance for each angle.

    Angles follow the project-wide convention fixed in Phase 0: elevation is
    degrees above the horizon in (0, 90]; azimuth is the compass bearing of the
    sun, degrees clockwise from North.
    """

    elevation_deg: float
    azimuth_deg: float
    elevation_provenance: Provenance
    azimuth_provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "elevation_deg", float(self.elevation_deg))
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg) % 360.0)
        if not 0.0 < self.elevation_deg <= 90.0:
            raise MetadataError(
                f"sun elevation {self.elevation_deg} deg is outside (0, 90]; "
                f"resolved from {self.elevation_provenance.describe()}. "
                "A sun at or below the horizon casts no measurable shadow."
            )

    def as_sun_config(self) -> SunConfig:
        """Convert to the plain config dataclass used by the physics layer."""
        return SunConfig(elevation_deg=self.elevation_deg, azimuth_deg=self.azimuth_deg)


@dataclass(frozen=True)
class GeoReference:
    """Georeferencing of a raster, with ground sample distance in metres."""

    crs: CRS
    transform: Affine
    width: int
    height: int
    gsd_x_m: float
    gsd_y_m: float
    gsd_note: str

    @property
    def gsd_m(self) -> float:
        """Single ground sample distance in metres/pixel.

        Raises ``MetadataError`` if pixels are not square, since the shadow
        geometry assumes one GSD.
        """
        if not math.isclose(self.gsd_x_m, self.gsd_y_m, rel_tol=1e-6):
            raise MetadataError(
                f"non-square pixels: x={self.gsd_x_m} m, y={self.gsd_y_m} m. "
                "Resample to square pixels before estimating heights."
            )
        return self.gsd_x_m

    @property
    def is_north_up(self) -> bool:
        t = self.transform
        return t.a > 0 and t.e < 0 and t.b == 0 and t.d == 0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in the raster's CRS."""
        west, north = pixel_to_map(self.transform, 0, 0)
        east, south = pixel_to_map(self.transform, self.width, self.height)
        return (min(west, east), min(north, south), max(west, east), max(north, south))

    @property
    def centre_lonlat(self) -> tuple[float, float] | None:
        """Scene centre as ``(lon, lat)``, or None if it cannot be computed."""
        x, y = pixel_to_map(self.transform, self.width / 2.0, self.height / 2.0)
        if self.crs.is_geographic:
            return (x, y)
        try:
            from rasterio.warp import transform as warp_transform

            lons, lats = warp_transform(self.crs, CRS.from_epsg(4326), [x], [y])
            return (float(lons[0]), float(lats[0]))
        except Exception:  # pragma: no cover - depends on PROJ data availability
            return None


@dataclass(frozen=True)
class RpcInfo:
    """Rational polynomial coefficients, when the product ships them.

    RPCs describe the sensor's viewing geometry. They never contain sun angles,
    but they do pin the scene's approximate geographic centre even when the
    file has no CRS, and later phases need them for off-nadir corrections.
    Phase 1 only records them.
    """

    source: str
    line_off: float | None = None
    samp_off: float | None = None
    lat_off_deg: float | None = None
    long_off_deg: float | None = None
    height_off_m: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def centre_lonlat(self) -> tuple[float, float] | None:
        if self.long_off_deg is None or self.lat_off_deg is None:
            return None
        return (self.long_off_deg, self.lat_off_deg)


@dataclass(frozen=True)
class SceneMetadata:
    """Everything ingest learned about one input file."""

    path: Path
    mode: Mode
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    georeference: GeoReference | None = None
    sun: SunMetadata | None = None
    rpc: RpcInfo | None = None
    probed_sources: tuple[str, ...] = ()
    #: Set when a georeferenced input still could not reach ABSOLUTE mode.
    downgrade_reason: str | None = None

    @property
    def crs(self) -> CRS | None:
        return self.georeference.crs if self.georeference else None

    @property
    def gsd_m(self) -> float | None:
        return self.georeference.gsd_m if self.georeference else None

    @property
    def sun_elevation_deg(self) -> float | None:
        return self.sun.elevation_deg if self.sun else None

    @property
    def sun_azimuth_deg(self) -> float | None:
        return self.sun.azimuth_deg if self.sun else None

    def require_absolute(self) -> "SceneMetadata":
        """Return self, or raise if this scene cannot support metric heights."""
        if self.mode is Mode.ABSOLUTE:
            return self
        missing = []
        if self.georeference is None:
            missing.append("crs/geotransform")
        if self.sun is None:
            missing.extend(["sun_elevation_deg", "sun_azimuth_deg"])
        raise MissingMetadataError(
            missing=missing or ["absolute-mode metadata"],
            probed_sources=self.probed_sources,
            tried_keys={
                "sun_elevation_deg": SUN_ELEVATION_ALIASES + SUN_ZENITH_ALIASES,
                "sun_azimuth_deg": SUN_AZIMUTH_ALIASES,
            },
            path=self.path,
            hint=self.downgrade_reason
            or "ABSOLUTE mode needs a CRS, a geotransform and both sun angles.",
        )

    def summary(self) -> str:
        """Short human-readable report, used by scripts/inspect_scene.py."""
        lines = [
            f"path            : {self.path}",
            f"mode            : {self.mode.value}",
            f"driver          : {self.driver}",
            f"size            : {self.width} x {self.height} px, {self.band_count} band(s), {self.dtype}",
        ]
        if self.georeference:
            geo = self.georeference
            lines += [
                f"crs             : {geo.crs.to_string()}",
                f"geotransform    : {tuple(round(v, 6) for v in tuple(geo.transform)[:6])}",
                f"gsd             : {geo.gsd_x_m:.6g} m/px (x), {geo.gsd_y_m:.6g} m/px (y)",
                f"gsd note        : {geo.gsd_note}",
                f"north-up        : {geo.is_north_up}",
                f"bounds          : {tuple(round(v, 3) for v in geo.bounds)}",
            ]
        else:
            lines.append("crs             : (none - not georeferenced)")
        if self.sun:
            lines += [
                f"sun elevation   : {self.sun.elevation_deg:.6g} deg  <- {self.sun.elevation_provenance.describe()}",
                f"sun azimuth     : {self.sun.azimuth_deg:.6g} deg  <- {self.sun.azimuth_provenance.describe()}",
            ]
        else:
            lines.append("sun             : (none found)")
        if self.rpc:
            lines.append(f"rpc             : present via {self.rpc.source}, centre={self.rpc.centre_lonlat}")
        lines.append(f"probed sources  : {', '.join(self.probed_sources) or '(none)'}")
        if self.downgrade_reason:
            lines.append(f"downgraded      : {self.downgrade_reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key aliases
#
# Normalised comparison (lowercase, alphanumerics only) means one entry covers
# SUN_ELEVATION, sun-elevation, SunElevation and sunElevation alike.
# ---------------------------------------------------------------------------


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


SUN_ELEVATION_ALIASES: tuple[str, ...] = (
    "SUN_ELEVATION_DEG",  # DepthWizard native (Phase 0 writes this)
    "SUN_ELEVATION",  # Landsat MTL, generic GDAL
    "SUN_ELEV",
    "SUNEL",  # Maxar / WorldView TRE
    "meanSunEl",  # Maxar .IMD sidecar
    "MEANSUNEL",
    "SOLAR_ELEVATION",
    "SolarElevation",
    "sun_elevation",  # Planet
    "SUN_ELEVATION_ANGLE",
)

#: Providers that report zenith instead; elevation = 90 - zenith.
SUN_ZENITH_ALIASES: tuple[str, ...] = (
    "MEAN_SUN_ZENITH_ANGLE",  # Sentinel-2
    "SUN_ZENITH",
    "SUN_ZENITH_ANGLE",
    "SOLAR_ZENITH",
    "SOLAR_ZENITH_ANGLE",
    "SolarZenith",
)

SUN_AZIMUTH_ALIASES: tuple[str, ...] = (
    "SUN_AZIMUTH_DEG",  # DepthWizard native
    "SUN_AZIMUTH",  # Landsat MTL, generic GDAL
    "SUNAZ",  # Maxar / WorldView TRE
    "meanSunAz",  # Maxar .IMD sidecar
    "MEANSUNAZ",
    "SOLAR_AZIMUTH",
    "SolarAzimuth",
    "sun_azimuth",  # Planet
    "MEAN_SUN_AZIMUTH_ANGLE",  # Sentinel-2
    "SUN_AZIMUTH_ANGLE",
)

_ELEVATION_LOOKUP = {_normalise_key(k) for k in SUN_ELEVATION_ALIASES}
_ZENITH_LOOKUP = {_normalise_key(k) for k in SUN_ZENITH_ALIASES}
_AZIMUTH_LOOKUP = {_normalise_key(k) for k in SUN_AZIMUTH_ALIASES}

#: GDAL domains worth probing even when the driver does not advertise them.
EXTRA_GDAL_DOMAINS: tuple[str, ...] = ("IMD", "RPC", "IMAGERY", "EXIF", "TRE", "GEOLOCATION")

#: Sidecar extensions carrying provider metadata, checked next to the image.
IMD_SIDECAR_SUFFIXES: tuple[str, ...] = (".IMD", ".imd")


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadataCandidates:
    """Flat key/value maps gathered from every probed source, in priority order."""

    #: ``(source_label, {key: value})`` pairs.
    tables: tuple[tuple[str, Mapping[str, str]], ...] = ()

    @property
    def source_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.tables)

    def find(self, normalised_keys: set[str]) -> tuple[str, str, str] | None:
        """Return ``(source, key, raw_value)`` for the first matching key."""
        for label, table in self.tables:
            for key, value in table.items():
                if _normalise_key(key) in normalised_keys:
                    if value is None or str(value).strip() == "":
                        continue
                    return label, str(key), str(value)
        return None


def parse_imd(text: str) -> dict[str, str]:
    """Parse a Maxar/DigitalGlobe ``.IMD`` sidecar into a flat dict.

    The format is ``key = value;`` lines wrapped in ``BEGIN_GROUP`` /
    ``END_GROUP`` blocks. Group structure is flattened; the keys DepthWizard
    cares about (``meanSunEl``, ``meanSunAz``) are unique across groups.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("BEGIN_GROUP", "END_GROUP", "END;", "END")):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().rstrip(";").strip().strip('"')
        if key and value:
            values[key] = value
    return values


def find_sidecars(path: Path) -> list[Path]:
    """Locate ``.IMD`` sidecars belonging to ``path``.

    Checks both ``scene.IMD`` (replacing the extension) and ``scene.tif.IMD``
    (appending), in either case sensitivity.
    """
    candidates: list[Path] = []
    for suffix in IMD_SIDECAR_SUFFIXES:
        candidates.append(path.with_suffix(suffix))
        candidates.append(path.with_name(path.name + suffix))
    seen: set[Path] = set()
    found: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            found.append(candidate)
    return found


def collect_candidates(dataset: rasterio.DatasetReader, path: Path) -> MetadataCandidates:
    """Gather every key/value table worth searching, in priority order."""
    tables: list[tuple[str, Mapping[str, str]]] = []

    default_tags = dataset.tags()
    if default_tags:
        tables.append(("gdal:default", dict(default_tags)))

    domains = list(dict.fromkeys([*dataset.tag_namespaces(), *EXTRA_GDAL_DOMAINS]))
    for domain in domains:
        try:
            domain_tags = dataset.tags(ns=domain)
        except Exception:  # pragma: no cover - driver dependent
            continue
        if domain_tags:
            tables.append((f"gdal:{domain}", dict(domain_tags)))

    try:
        rpcs = dataset.rpcs
    except Exception:  # pragma: no cover - driver dependent
        rpcs = None
    if rpcs:
        rpc_dict = rpcs.to_dict() if hasattr(rpcs, "to_dict") else dict(rpcs)
        tables.append(("rpc", {k: str(v) for k, v in rpc_dict.items()}))

    for sidecar in find_sidecars(path):
        try:
            parsed = parse_imd(sidecar.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:  # pragma: no cover - unreadable sidecar
            log.warning("could not read sidecar", extra={"sidecar": str(sidecar), "error": str(exc)})
            continue
        if parsed:
            tables.append((f"imd:{sidecar.name}", parsed))

    return MetadataCandidates(tables=tuple(tables))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _to_float(raw: str) -> float | None:
    """Parse a number out of a metadata string, tolerating units and commas."""
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw).replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - regex guarantees parseability
        return None


def resolve_sun(candidates: MetadataCandidates, path: Path | None = None) -> SunMetadata | None:
    """Resolve sun elevation and azimuth from probed metadata.

    Returns None when either angle is absent. Nothing is defaulted or guessed:
    an absent angle stays absent, and the caller decides whether that is fatal.
    """
    elevation: tuple[float, Provenance] | None = None

    direct = candidates.find(_ELEVATION_LOOKUP)
    if direct is not None:
        source, key, raw = direct
        value = _to_float(raw)
        if value is not None:
            elevation = (value, Provenance(source=source, key=key, raw_value=raw))

    if elevation is None:
        zenith = candidates.find(_ZENITH_LOOKUP)
        if zenith is not None:
            source, key, raw = zenith
            value = _to_float(raw)
            if value is not None:
                elevation = (
                    90.0 - value,
                    Provenance(
                        source=source,
                        key=key,
                        raw_value=raw,
                        conversion="elevation_deg = 90 - zenith_deg",
                    ),
                )

    azimuth: tuple[float, Provenance] | None = None
    found_azimuth = candidates.find(_AZIMUTH_LOOKUP)
    if found_azimuth is not None:
        source, key, raw = found_azimuth
        value = _to_float(raw)
        if value is not None:
            azimuth = (value, Provenance(source=source, key=key, raw_value=raw))

    if elevation is None or azimuth is None:
        return None

    return SunMetadata(
        elevation_deg=elevation[0],
        azimuth_deg=azimuth[0],
        elevation_provenance=elevation[1],
        azimuth_provenance=azimuth[1],
    )


def read_rpc(candidates: MetadataCandidates) -> RpcInfo | None:
    """Extract RPC offsets, if the product carries RPCs."""
    for label, table in candidates.tables:
        if label != "rpc" and not label.endswith(":RPC"):
            continue
        lookup = {_normalise_key(k): v for k, v in table.items()}
        if not lookup:
            continue
        return RpcInfo(
            source=label,
            line_off=_to_float(lookup.get("lineoff", "")) if "lineoff" in lookup else None,
            samp_off=_to_float(lookup.get("sampoff", "")) if "sampoff" in lookup else None,
            lat_off_deg=_to_float(lookup.get("latoff", "")) if "latoff" in lookup else None,
            long_off_deg=_to_float(lookup.get("longoff", "")) if "longoff" in lookup else None,
            height_off_m=_to_float(lookup.get("heightoff", "")) if "heightoff" in lookup else None,
            raw=dict(table),
        )
    return None


# ---------------------------------------------------------------------------
# Ground sample distance
# ---------------------------------------------------------------------------


def _metres_per_degree(latitude_deg: float) -> tuple[float, float]:
    """WGS84 metres per degree of longitude and latitude at a given latitude.

    Standard truncated series; accurate to well under a metre, which is far
    finer than any GSD this project deals with.
    """
    phi = math.radians(latitude_deg)
    per_deg_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    per_deg_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return per_deg_lon, per_deg_lat


def compute_gsd_m(
    crs: CRS, transform: Affine, width: int, height: int
) -> tuple[float, float, str]:
    """Ground sample distance in metres/pixel, for projected or geographic CRSs.

    Returns ``(gsd_x_m, gsd_y_m, note)``. The note records how the value was
    obtained, since a geographic CRS requires a latitude-dependent conversion
    that is an approximation.

    Pixel sizes use ``hypot`` over the transform's linear terms, so rotated
    (non-north-up) transforms are handled correctly.
    """
    if crs is None:
        raise MetadataError("cannot compute GSD without a CRS")

    # Column step and row step in CRS units, valid under rotation.
    col_step = math.hypot(transform.a, transform.d)
    row_step = math.hypot(transform.b, transform.e)

    if crs.is_geographic:
        _, centre_lat = pixel_to_map(transform, width / 2.0, height / 2.0)
        per_deg_lon, per_deg_lat = _metres_per_degree(centre_lat)
        gsd_x = col_step * per_deg_lon
        gsd_y = row_step * per_deg_lat
        note = (
            f"converted from geographic CRS degrees at centre latitude "
            f"{centre_lat:.4f} deg (approximate)"
        )
        return gsd_x, gsd_y, note

    factor = 1.0
    unit_name = "unknown"
    try:
        units = crs.linear_units_factor
        if units:
            unit_name, factor = units[0], float(units[1])
    except Exception:  # pragma: no cover - CRS without linear units
        factor = 1.0

    if math.isclose(factor, 1.0):
        note = f"projected CRS, linear unit '{unit_name}' (already metres)"
    else:
        note = f"projected CRS, converted from '{unit_name}' (x{factor} m/unit)"
    return col_step * factor, row_step * factor, note


def read_georeference(dataset: rasterio.DatasetReader) -> GeoReference | None:
    """Build a :class:`GeoReference` if the dataset is actually georeferenced.

    A CRS alone is not enough: GDAL hands back an identity transform for
    ungeoreferenced rasters, which would silently produce a GSD of 1.0.
    """
    if dataset.crs is None:
        return None
    transform = dataset.transform
    if transform is None or transform.is_identity:
        return None

    gsd_x, gsd_y, note = compute_gsd_m(
        dataset.crs, transform, dataset.width, dataset.height
    )
    return GeoReference(
        crs=dataset.crs,
        transform=transform,
        width=dataset.width,
        height=dataset.height,
        gsd_x_m=gsd_x,
        gsd_y_m=gsd_y,
        gsd_note=note,
    )
