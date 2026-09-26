"""YAML-backed dataclass configuration for DepthWizard.

Design goals:

* One dataclass per config block, so that every setting has a name, a type and
  a docstring in exactly one place.
* Strict loading. Unknown keys and missing required keys raise
  :class:`ConfigError` instead of being silently dropped -- a typo in a YAML
  file should fail loudly, not produce a subtly wrong run.
* Validation at construction time, so an invalid config can never reach the
  rendering or physics code.

Typical use::

    from depthwizard.config import AppConfig
    cfg = AppConfig.from_yaml("configs/default.yaml")
    print(cfg.scene.sun.elevation_deg)
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "ConfigError",
    "ProjectConfig",
    "PathsConfig",
    "LoggingConfig",
    "RasterConfig",
    "SunConfig",
    "GroundConfig",
    "BuildingSpec",
    "SceneConfig",
    "RadiometryConfig",
    "TilingConfig",
    "IngestConfig",
    "AppConfig",
    "load_config",
    "check_keys",
    "build_dataclass",
]

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed, incomplete or invalid."""


# ---------------------------------------------------------------------------
# Strict mapping -> dataclass helpers
# ---------------------------------------------------------------------------


def _check_keys(cls: type, data: Mapping[str, Any], where: str) -> None:
    """Reject unknown keys and report missing required ones for ``cls``."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed keys are {sorted(known)}"
        )

    required = {
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = required - set(data)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {sorted(missing)}")


def _build(cls: type, data: Mapping[str, Any], where: str):
    """Construct dataclass ``cls`` from ``data`` after strict key checking."""
    if not is_dataclass(cls):  # pragma: no cover - programmer error
        raise TypeError(f"{cls!r} is not a dataclass")
    _check_keys(cls, data, where)
    try:
        return cls(**data)
    except TypeError as exc:  # pragma: no cover - guarded by _check_keys
        raise ConfigError(f"{where}: {exc}") from exc


# ---------------------------------------------------------------------------
# Config blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectConfig:
    """Identity of the project and of this particular run."""

    name: str = "depthwizard"
    run_id: str = "dev"


@dataclass(frozen=True)
class PathsConfig:
    """Where data lives. All paths are relative to the repository root."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    outputs: Path = Path("data/outputs")

    def __post_init__(self) -> None:
        # YAML gives us strings; normalise to Path without breaking frozen=True.
        for f in fields(self):
            object.__setattr__(self, f.name, Path(getattr(self, f.name)))


@dataclass(frozen=True)
class LoggingConfig:
    """Logging verbosity and output shape."""

    level: str = "INFO"
    format: str = "text"  # "text" | "json"

    _LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    _FORMATS = ("text", "json")

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", str(self.level).upper())
        object.__setattr__(self, "format", str(self.format).lower())
        if self.level not in self._LEVELS:
            raise ConfigError(f"logging.level must be one of {self._LEVELS}, got {self.level!r}")
        if self.format not in self._FORMATS:
            raise ConfigError(f"logging.format must be one of {self._FORMATS}, got {self.format!r}")


@dataclass(frozen=True)
class RasterConfig:
    """Geometry and georeferencing of the raster grid to synthesise."""

    width_px: int
    height_px: int
    gsd_m: float
    crs: str
    origin_easting_m: float
    origin_northing_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "height_px", int(self.height_px))
        object.__setattr__(self, "gsd_m", float(self.gsd_m))
        object.__setattr__(self, "origin_easting_m", float(self.origin_easting_m))
        object.__setattr__(self, "origin_northing_m", float(self.origin_northing_m))
        if self.width_px <= 0 or self.height_px <= 0:
            raise ConfigError("raster.width_px and raster.height_px must be positive")
        if self.gsd_m <= 0:
            raise ConfigError("raster.gsd_m must be positive")
        if not str(self.crs).strip():
            raise ConfigError("raster.crs must be a non-empty CRS string, e.g. 'EPSG:32643'")

    @property
    def width_m(self) -> float:
        """Ground width of the raster in metres."""
        return self.width_px * self.gsd_m

    @property
    def height_m(self) -> float:
        """Ground height of the raster in metres."""
        return self.height_px * self.gsd_m


@dataclass(frozen=True)
class SunConfig:
    """Illumination geometry.

    ``azimuth_deg`` is the compass bearing of the sun as seen from the ground,
    measured clockwise from North (0 = N, 90 = E, 180 = S, 270 = W).
    ``elevation_deg`` is the angle of the sun above the horizon.
    """

    elevation_deg: float
    azimuth_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "elevation_deg", float(self.elevation_deg))
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg) % 360.0)
        if not 0.0 < self.elevation_deg <= 90.0:
            raise ConfigError(
                f"sun.elevation_deg must be in (0, 90], got {self.elevation_deg}. "
                "A sun at or below the horizon casts no usable shadow."
            )


@dataclass(frozen=True)
class GroundConfig:
    """Brightness values used to paint the synthetic scene.

    Values are unitless reflectance-like floats in [0, 1]; they are scaled to
    the raster's output dtype when the GeoTIFF is written.
    """

    reflectance: float = 0.35
    roof_reflectance: float = 0.65
    shadow_reflectance: float = 0.08
    noise_sigma: float = 0.0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("reflectance", "roof_reflectance", "shadow_reflectance", "noise_sigma"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "noise_seed", int(self.noise_seed))
        for name in ("reflectance", "roof_reflectance", "shadow_reflectance"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"ground.{name} must be in [0, 1], got {value}")
        if self.noise_sigma < 0.0:
            raise ConfigError("ground.noise_sigma must be >= 0")
        if not self.shadow_reflectance < self.reflectance:
            raise ConfigError(
                "ground.shadow_reflectance must be darker than ground.reflectance, "
                "otherwise shadows are not detectable in the fixture"
            )


@dataclass(frozen=True)
class BuildingSpec:
    """One axis-aligned box building with a known height.

    ``x_m`` / ``y_m`` locate the footprint's top-left corner relative to the
    raster's north-west corner: ``x_m`` metres East, ``y_m`` metres South.
    ``height_m`` is the ground truth this project exists to recover.
    """

    name: str
    x_m: float
    y_m: float
    width_m: float
    depth_m: float
    height_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "width_m", "depth_m", "height_m"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.width_m <= 0 or self.depth_m <= 0:
            raise ConfigError(f"building {self.name!r}: width_m and depth_m must be positive")
        if self.height_m <= 0:
            raise ConfigError(f"building {self.name!r}: height_m must be positive")
        if self.x_m < 0 or self.y_m < 0:
            raise ConfigError(f"building {self.name!r}: x_m and y_m must be >= 0")


@dataclass(frozen=True)
class SceneConfig:
    """A complete synthetic scene: a grid, a sun, a ground and some buildings."""

    name: str
    raster: RasterConfig
    sun: SunConfig
    ground: GroundConfig = field(default_factory=GroundConfig)
    buildings: Sequence[BuildingSpec] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "buildings", tuple(self.buildings))
        names = [b.name for b in self.buildings]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigError(f"scene.buildings: duplicate building name(s) {duplicates}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "scene") -> "SceneConfig":
        data = dict(data)
        _check_keys(cls, data, where)
        data["raster"] = _build(RasterConfig, data["raster"], f"{where}.raster")
        data["sun"] = _build(SunConfig, data["sun"], f"{where}.sun")
        if "ground" in data:
            data["ground"] = _build(GroundConfig, data["ground"], f"{where}.ground")
        buildings = data.get("buildings") or []
        if not isinstance(buildings, Sequence) or isinstance(buildings, (str, bytes)):
            raise ConfigError(f"{where}.buildings: expected a list")
        data["buildings"] = tuple(
            _build(BuildingSpec, b, f"{where}.buildings[{i}]") for i, b in enumerate(buildings)
        )
        return cls(**data)


@dataclass(frozen=True)
class RadiometryConfig:
    """Percentile stretch applied when normalising imagery to 8-bit."""

    lower_percentile: float = 2.0
    upper_percentile: float = 98.0
    per_band: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_percentile", float(self.lower_percentile))
        object.__setattr__(self, "upper_percentile", float(self.upper_percentile))
        object.__setattr__(self, "per_band", bool(self.per_band))
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ConfigError(
                "radiometry needs 0 <= lower_percentile < upper_percentile <= 100, got "
                f"{self.lower_percentile} and {self.upper_percentile}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Tile geometry. Overlap must be smaller than the tile, or tiles never advance."""

    tile_size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_size", int(self.tile_size))
        object.__setattr__(self, "overlap", int(self.overlap))
        if self.tile_size <= 0:
            raise ConfigError(f"tiling.tile_size must be positive, got {self.tile_size}")
        if self.overlap < 0:
            raise ConfigError(f"tiling.overlap must be >= 0, got {self.overlap}")
        if self.overlap >= self.tile_size:
            raise ConfigError(
                f"tiling.overlap ({self.overlap}) must be smaller than "
                f"tiling.tile_size ({self.tile_size})"
            )

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap


@dataclass(frozen=True)
class IngestConfig:
    """Phase 1 ingest settings."""

    radiometry: RadiometryConfig = field(default_factory=RadiometryConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "ingest") -> "IngestConfig":
        data = dict(data)
        _check_keys(cls, data, where)
        if "radiometry" in data:
            data["radiometry"] = _build(RadiometryConfig, data["radiometry"], f"{where}.radiometry")
        if "tiling" in data:
            data["tiling"] = _build(TilingConfig, data["tiling"], f"{where}.tiling")
        return cls(**data)


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration object -- the root of the YAML document."""

    scene: SceneConfig
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        data = dict(data)
        _check_keys(cls, data, "config root")
        for key, block in (("project", ProjectConfig), ("paths", PathsConfig), ("logging", LoggingConfig)):
            if key in data:
                data[key] = _build(block, data[key], key)
        if "ingest" in data:
            data["ingest"] = IngestConfig.from_dict(data["ingest"])
        data["scene"] = SceneConfig.from_dict(data["scene"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            raise ConfigError(f"config file is empty: {path}")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config root of {path} must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate a DepthWizard config file."""
    return AppConfig.from_yaml(path)


# Public aliases for the strict loader helpers above. Phase 3 keeps its own
# config root in `depthwizard.relative.config` (it configures training, not a
# scene) but must load it with exactly the same strictness, so it reuses these
# rather than growing a second, looser parser.
check_keys = _check_keys
build_dataclass = _build
