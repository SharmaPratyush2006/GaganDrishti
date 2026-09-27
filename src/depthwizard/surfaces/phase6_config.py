"""Phase 6 configuration (``configs/phase6.yaml``): DTM and nDSM from a DSM.

Strict, like every DepthWizard config: an unknown or misspelt key raises
:class:`~depthwizard.config.ConfigError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from depthwizard.config import ConfigError, build_dataclass, check_keys

__all__ = ["GroundFilterConfig", "Phase6SyntheticConfig", "Phase6Config", "load_phase6_config"]


@dataclass(frozen=True)
class GroundFilterConfig:
    """Parameters of the progressive morphological filter and the TIN refinement (metres)."""

    #: Largest axis-aligned building extent. None: derived from the known footprints
    #: of the SYNTHETIC fixture (the only case where they are known); required otherwise.
    max_building_extent_m: float | None = None
    max_terrain_slope: float = 0.05
    min_object_height_m: float = 2.0
    #: TIN refinement passes (0 disables). Re-admission tolerance = min_object_height_m.
    tin_refinement_max_iterations: int = 5

    def __post_init__(self) -> None:
        if self.max_building_extent_m is not None and not (
                math.isfinite(self.max_building_extent_m) and self.max_building_extent_m > 0):
            raise ConfigError("ground.max_building_extent_m must be > 0 or null")
        if not (math.isfinite(self.max_terrain_slope) and self.max_terrain_slope > 0):
            raise ConfigError("ground.max_terrain_slope must be > 0")
        if not (math.isfinite(self.min_object_height_m) and self.min_object_height_m > 0):
            raise ConfigError("ground.min_object_height_m must be > 0")
        if self.tin_refinement_max_iterations < 0:
            raise ConfigError("ground.tin_refinement_max_iterations must be >= 0")


@dataclass(frozen=True)
class Phase6SyntheticConfig:
    #: Written by ``python -m depthwizard.surfaces.run synthetic``.
    phase4b_report: Path = Path("data/outputs/phase4b/phase4b_report.json")
    #: Phase 0 scene the fixture was rendered from (for the analytic terrain truth).
    scene_config: Path = Path("configs/default.yaml")
    #: Phase 4b DSM products to process; the FIRST is the acceptance input.
    dsm_products: tuple[str, ...] = ("synthetic_dsm_known_calibration.tif", "synthetic_dsm.tif")

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase4b_report", Path(self.phase4b_report))
        object.__setattr__(self, "scene_config", Path(self.scene_config))
        object.__setattr__(self, "dsm_products", tuple(self.dsm_products))
        if not self.dsm_products:
            raise ConfigError("synthetic.dsm_products must not be empty")


@dataclass(frozen=True)
class Phase6Config:
    run_id: str = "phase6"
    output_dir: Path = Path("data/outputs/phase6")
    nodata: float = -9999.0
    ground: GroundFilterConfig = field(default_factory=GroundFilterConfig)
    synthetic: Phase6SyntheticConfig = field(default_factory=Phase6SyntheticConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not math.isfinite(self.nodata) or self.nodata == 0.0:
            raise ConfigError("nodata must be finite and non-zero")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Phase6Config":
        data = dict(data)
        check_keys(cls, data, "phase6 config root")
        if "ground" in data:
            data["ground"] = build_dataclass(GroundFilterConfig, data["ground"], "ground")
        if "synthetic" in data:
            data["synthetic"] = build_dataclass(Phase6SyntheticConfig, data["synthetic"], "synthetic")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase6Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"phase 6 config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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


def load_phase6_config(path: str | Path = "configs/phase6.yaml") -> Phase6Config:
    return Phase6Config.from_yaml(path)
