"""Phase 5 configuration: the validation harness (``configs/phase5.yaml``).

Loaded with the same strict machinery as every other DepthWizard config
(:func:`depthwizard.config.check_keys` / :func:`depthwizard.config.build_dataclass`):
an unknown or misspelt key raises :class:`~depthwizard.config.ConfigError`.

There is deliberately **no random split option** anywhere in this file. The
real-data split is declared as named regions (``dfc.split``), and
:mod:`depthwizard.validation.spatial_split` checks that declaration against
the split the checkpoint was actually trained with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from depthwizard.config import ConfigError, build_dataclass, check_keys
from depthwizard.physics.height import USABLE_SUN_ELEVATION_MAX_DEG, USABLE_SUN_ELEVATION_MIN_DEG
from depthwizard.validation.metrics import DEFAULT_DELTA_THRESHOLDS

__all__ = [
    "ValidationError",
    "ValidationInputError",
    "SunBandConfig",
    "ErrorMapConfig",
    "SyntheticValidationConfig",
    "RegionSplitConfig",
    "DfcValidationConfig",
    "Phase5Config",
    "load_phase5_config",
    "DEFAULT_PHASE5_CONFIG_PATH",
    "SUPPORTED_REGION_KEYS",
    "SYNTHETIC_PRODUCTS",
]

DEFAULT_PHASE5_CONFIG_PATH = Path("configs/phase5.yaml")
#: The only region identifier DFC2019 actually carries: the city code in the
#: filename (``JAX_004_007`` -> ``JAX``), read by the Phase 3 scene regex.
SUPPORTED_REGION_KEYS = ("city",)
#: Phase 4b synthetic products the harness can score.
SYNTHETIC_PRODUCTS = ("agl", "dsm")
#: Phase 4a fit labels (``depthwizard.calibration.fusion.IDENTIFIABILITY_CLASSES``
#: minus the UNIDENTIFIABLE ones, which return no a, b to evaluate).
FIT_CLASSES = ("identifiable_positive", "statistically_weak", "non_positive")


class ValidationError(RuntimeError):
    """Phase 5 cannot produce a trustworthy result; the message says why."""


class ValidationInputError(ValidationError):
    """A required input is missing or inconsistent."""


@dataclass(frozen=True)
class SunBandConfig:
    """The usable sun-elevation band. Defaults are the project's 25-45 deg."""

    min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG
    max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG

    def __post_init__(self) -> None:
        if not (0.0 < self.min_deg < self.max_deg <= 90.0):
            raise ConfigError(f"sun_band needs 0 < min_deg < max_deg <= 90, got {self.min_deg}, {self.max_deg}")


@dataclass(frozen=True)
class ErrorMapConfig:
    #: Written for invalid error-map pixels. Never 0, which is a real error.
    nodata: float = -9999.0
    #: The figure's colour scale spans +/- this percentile of |error|. Errors
    #: beyond it are drawn at the end colour (never hidden), counted on the
    #: figure, and the full range is printed; the raster itself is unclipped.
    display_percentile: float = 99.0
    #: DFC2019 only: figures drawn for at most this many tiles (rasters are
    #: written for every evaluated tile).
    max_figures: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.nodata) or self.nodata == 0.0:
            raise ConfigError("error_map.nodata must be finite and non-zero")
        if not 50.0 <= self.display_percentile <= 100.0:
            raise ConfigError("error_map.display_percentile must be in [50, 100]")
        if self.max_figures < 0:
            raise ConfigError("error_map.max_figures must be >= 0")


@dataclass(frozen=True)
class SyntheticValidationConfig:
    """Scores the Phase 4b SYNTHETIC products against the fixture's truth."""

    #: Written by ``python -m depthwizard.surfaces.run synthetic``.
    phase4b_report: Path = Path("data/outputs/phase4b/phase4b_report.json")
    #: Phase 0 scene the fixture was rendered from. Checked against the
    #: fixture's truth sidecar; needed to rebuild the analytic terrain truth.
    scene_config: Path = Path("configs/default.yaml")
    products: tuple[str, ...] = SYNTHETIC_PRODUCTS

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase4b_report", Path(self.phase4b_report))
        object.__setattr__(self, "scene_config", Path(self.scene_config))
        object.__setattr__(self, "products", tuple(self.products))
        unknown = [p for p in self.products if p not in SYNTHETIC_PRODUCTS]
        if unknown or not self.products:
            raise ConfigError(f"synthetic.products must be a non-empty subset of {SYNTHETIC_PRODUCTS}, got {self.products}")


@dataclass(frozen=True)
class RegionSplitConfig:
    """The declared geographic split. Both lists are required and disjoint."""

    region_key: str = "city"
    train_regions: tuple[str, ...] = ()
    heldout_regions: tuple[str, ...] = ()
    #: Why this split, in words, recorded in the report.
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_regions", tuple(str(r) for r in self.train_regions))
        object.__setattr__(self, "heldout_regions", tuple(str(r) for r in self.heldout_regions))
        if self.region_key not in SUPPORTED_REGION_KEYS:
            raise ConfigError(f"dfc.split.region_key must be one of {SUPPORTED_REGION_KEYS}, got {self.region_key!r}")
        if not self.train_regions or not self.heldout_regions:
            raise ConfigError("dfc.split needs non-empty train_regions and heldout_regions")
        overlap = set(self.train_regions) & set(self.heldout_regions)
        if overlap:
            raise ConfigError(f"dfc.split: regions {sorted(overlap)} are both training and held out")


@dataclass(frozen=True)
class DfcValidationConfig:
    """Scores Phase 4a calibrated AGL on DFC2019 against DFC2019 lidar AGL."""

    #: A Phase 4a output directory: phase4a_report.json + per_tile.jsonl. The
    #: checkpoint and Phase 4 settings are taken from that report, so the
    #: evaluated field is exactly the one a, b were fitted to.
    #: Required: there is no default split, because a split is a claim.
    split: RegionSplitConfig
    phase4a_dir: Path = Path("data/outputs/phase4a")
    cls_suffix: str = "_CLS.tif"
    #: DFC2019 CLS (ASPRS LAS) codes. Verified values: 2 ground, 5 high
    #: vegetation, 6 building, 9 water, 17 bridge deck, 65 unlabelled.
    terrain_classes: tuple[int, ...] = (2,)
    building_classes: tuple[int, ...] = (6,)
    water_classes: tuple[int, ...] = (9,)
    #: Which Phase 4a fit labels to evaluate.
    fit_classes: tuple[str, ...] = FIT_CLASSES

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase4a_dir", Path(self.phase4a_dir))
        for name in ("terrain_classes", "building_classes", "water_classes"):
            object.__setattr__(self, name, tuple(int(c) for c in getattr(self, name)))
        object.__setattr__(self, "fit_classes", tuple(self.fit_classes))
        unknown = [c for c in self.fit_classes if c not in FIT_CLASSES]
        if unknown or not self.fit_classes:
            raise ConfigError(f"dfc.fit_classes must be a non-empty subset of {FIT_CLASSES}, got {self.fit_classes}")
        if set(self.terrain_classes) & set(self.building_classes):
            raise ConfigError("dfc: a CLS class cannot be both terrain and building")


@dataclass(frozen=True)
class Phase5Config:
    run_id: str = "phase5"
    output_dir: Path = Path("data/outputs/phase5")
    delta_thresholds: tuple[float, ...] = DEFAULT_DELTA_THRESHOLDS
    sun_band: SunBandConfig = field(default_factory=SunBandConfig)
    error_map: ErrorMapConfig = field(default_factory=ErrorMapConfig)
    synthetic: SyntheticValidationConfig = field(default_factory=SyntheticValidationConfig)
    #: None -> no real-data split declared; the ``dfc`` target then refuses.
    dfc: DfcValidationConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        thresholds = tuple(float(t) for t in self.delta_thresholds)
        if not thresholds or any(not (math.isfinite(t) and t > 1.0) for t in thresholds):
            raise ConfigError(f"delta_thresholds must be non-empty and each > 1, got {self.delta_thresholds}")
        object.__setattr__(self, "delta_thresholds", thresholds)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Phase5Config":
        data = dict(data)
        check_keys(cls, data, "phase5 config root")
        for key, block in (("sun_band", SunBandConfig), ("error_map", ErrorMapConfig),
                           ("synthetic", SyntheticValidationConfig)):
            if key in data:
                data[key] = build_dataclass(block, data[key], key)
        if "dfc" in data:
            dfc = dict(data["dfc"])
            check_keys(DfcValidationConfig, dfc, "dfc")
            if "split" in dfc:
                dfc["split"] = build_dataclass(RegionSplitConfig, dfc["split"], "dfc.split")
            data["dfc"] = build_dataclass(DfcValidationConfig, dfc, "dfc")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase5Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"phase 5 config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config root of {path} must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            if hasattr(value, "__dataclass_fields__"):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            return value

        return convert(self)


def load_phase5_config(path: str | Path = DEFAULT_PHASE5_CONFIG_PATH) -> Phase5Config:
    return Phase5Config.from_yaml(path)
