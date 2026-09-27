"""Phase 4a configuration: pixel-space metric calibration on DFC2019.

Loaded with the same strict machinery as every other DepthWizard config
(:func:`depthwizard.config.check_keys` / :func:`depthwizard.config.build_dataclass`):
an unknown key or a missing required key raises :class:`ConfigError`.

Two facts are recorded here rather than assumed in code:

* ``dataset.building_class`` -- the DFC2019 CLS value for buildings. It is 6
  (ASPRS LAS "building"), verified against the official DFC2019 baseline code;
  see :data:`depthwizard.calibration.footprints.DFC2019_BUILDING_CLASS_SOURCE`.
* ``dataset.gsd_m`` -- **null by default**. The ground sample distance of the
  1024x1024 DFC2019 Track 1 tiles could not be verified from an authoritative
  source (see ``dataset.gsd_note`` in ``configs/phase4.yaml``). Phase 4a is
  formulated so every metric result is GSD-invariant; only the reported sun
  elevation in degrees needs a GSD, and without one it is reported as not
  determinable rather than computed from a guessed number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from depthwizard.config import ConfigError, build_dataclass, check_keys

__all__ = [
    "DfcDataConfig",
    "FootprintConfig",
    "AzimuthConfig",
    "MeasurementConfig",
    "ReferenceConfig",
    "FitConfig",
    "SyntheticControlConfig",
    "Phase4Config",
    "load_phase4_config",
    "DEFAULT_PHASE4_CONFIG_PATH",
]

DEFAULT_PHASE4_CONFIG_PATH = Path("configs/phase4.yaml")


def _positive(value: float, name: str) -> None:
    if not value > 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def _fraction(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class DfcDataConfig:
    """Which DFC2019 files to read, and the facts about them."""

    #: CLS raster suffix, next to ``*_AGL.tif`` in the truth directory.
    cls_suffix: str = "_CLS.tif"
    #: Verified DFC2019 building class (ASPRS LAS code 6).
    building_class: int = 6
    #: CLS classes on which a cast shadow is accepted as evidence. Dark tree
    #: canopy and water are otherwise indistinguishable from shadow to a
    #: brightness threshold.
    shadow_surface_classes: tuple[int, ...] = (2,)
    #: Metres per pixel, or None when unverified (the shipped default).
    gsd_m: float | None = None
    gsd_note: str = ""
    #: Phase 3 inference tile; also the calibration unit (one a,b per tile).
    crop_size: int = 512
    #: Which side of the Phase 3 split to evaluate. Only "val" is held out.
    split_side: str = "val"
    #: Cap on image pairs (deterministic: the first N in sorted order). None = all.
    max_pairs: int | None = None
    #: Views per geographic tile, evenly spaced in sorted order. None = all.
    #: Used for the training diagnostics; the held-out run uses every view.
    views_per_scene: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shadow_surface_classes", tuple(int(c) for c in self.shadow_surface_classes))
        if not self.shadow_surface_classes:
            raise ConfigError("dataset.shadow_surface_classes must not be empty")
        if self.gsd_m is not None:
            _positive(float(self.gsd_m), "dataset.gsd_m")
            if not self.gsd_note:
                raise ConfigError("dataset.gsd_m is set but dataset.gsd_note gives no source for it")
        _positive(self.crop_size, "dataset.crop_size")
        if self.split_side not in ("val", "train"):
            raise ConfigError(f"dataset.split_side must be 'val' or 'train', got {self.split_side!r}")
        if self.max_pairs is not None:
            _positive(self.max_pairs, "dataset.max_pairs")
        if self.views_per_scene is not None:
            _positive(self.views_per_scene, "dataset.views_per_scene")


@dataclass(frozen=True)
class FootprintConfig:
    #: Components smaller than this many pixels are rejected.
    min_area_px: int = 50
    #: 4 or 8. 4 never joins two roofs that only touch at a corner.
    connectivity: int = 4
    #: Reject components touching the tile border: their footprint, their mean
    #: relative height and their shadow are all truncated by the crop.
    reject_edge_touching: bool = True

    def __post_init__(self) -> None:
        _positive(self.min_area_px, "footprints.min_area_px")
        if self.connectivity not in (4, 8):
            raise ConfigError(f"footprints.connectivity must be 4 or 8, got {self.connectivity}")


@dataclass(frozen=True)
class AzimuthConfig:
    #: Distance beyond each footprint, in pixels, over which shadow occupancy is scored.
    band_px: int = 8
    #: Candidate sun-azimuth step, degrees.
    step_deg: float = 1.0
    #: Fewer building pixels than this and no azimuth is estimated.
    min_building_px: int = 200
    #: Required peak-over-median occupancy ratio. Below it the direction is
    #: not distinguishable from "shadow everywhere" and the tile is skipped.
    min_peak_to_median: float = 1.25

    def __post_init__(self) -> None:
        _positive(self.band_px, "azimuth.band_px")
        _positive(self.step_deg, "azimuth.step_deg")
        _positive(self.min_building_px, "azimuth.min_building_px")
        if self.min_peak_to_median < 1.0:
            raise ConfigError("azimuth.min_peak_to_median must be >= 1")


@dataclass(frozen=True)
class MeasurementConfig:
    """Passed straight through to Phase 2 :class:`MeasurementParams`."""

    step_px: float = 0.5
    gap_tolerance_px: float = 2.0
    max_rays: int = 64
    min_rays: int = 3
    min_hit_fraction: float = 0.5
    nominal_hit_fraction: float = 0.8
    percentile: float = 50.0
    #: A measurement whose hit rays mostly stopped at the tile edge measured a
    #: truncated shadow, and is dropped.
    max_edge_terminated_fraction: float = 0.5
    #: A measurement whose hit rays mostly crossed another building's footprint
    #: is a merged shadow and is dropped. None disables the check.
    max_merged_fraction: float | None = 0.5

    def __post_init__(self) -> None:
        _fraction(self.max_edge_terminated_fraction, "measurement.max_edge_terminated_fraction")
        if self.max_merged_fraction is not None:
            _fraction(self.max_merged_fraction, "measurement.max_merged_fraction")


@dataclass(frozen=True)
class ReferenceConfig:
    #: Reference buildings used for the tile's sun calibration (spec: 2-3).
    primary_count: int = 3
    #: Fewer valid candidates than this -> calibration-insufficient.
    min_count: int = 2
    #: Reference-count sensitivity study.
    sensitivity_counts: tuple[int, ...] = (2, 3, 5)
    min_area_px: int = 200
    #: No other building-class pixel within this many pixels of the footprint.
    #: 0 disables the isolation gate.
    isolation_px: int = 10
    #: How reference k_i = h_ref / L_px are aggregated: median | mean | inverse_variance.
    k_estimator: str = "median"
    #: footprint area / convex hull area.
    min_solidity: float = 0.8
    #: Median ground-truth AGL over the footprint must reach this.
    min_height_m: float = 3.0
    #: Share of footprint pixels with valid ground-truth AGL.
    min_valid_fraction: float = 0.9
    #: Share of the swept shadow zone labelled a shadow-surface class (flat, open ground).
    min_shadow_zone_ground_fraction: float = 0.8
    #: ray_length_spread_px / shadow_length_px must not exceed this.
    max_relative_spread: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitivity_counts", tuple(int(c) for c in self.sensitivity_counts))
        if not 2 <= self.min_count <= self.primary_count:
            raise ConfigError("reference: need 2 <= min_count <= primary_count")
        for name in ("min_solidity", "min_valid_fraction", "min_shadow_zone_ground_fraction"):
            _fraction(getattr(self, name), f"reference.{name}")
        _positive(self.max_relative_spread, "reference.max_relative_spread")
        if self.isolation_px < 0:
            raise ConfigError("reference.isolation_px must be >= 0 (0 disables isolation)")
        if self.k_estimator not in ("median", "mean", "inverse_variance"):
            raise ConfigError(f"reference.k_estimator must be median, mean or inverse_variance, got {self.k_estimator!r}")


@dataclass(frozen=True)
class FitConfig:
    #: Fewer usable constraints than this -> the tile's a,b fit fails loudly.
    min_constraints: int = 5
    #: Weight multiplier for REDUCED-confidence constraints (NOMINAL = 1).
    reduced_weight_factor: float = 0.5
    #: Refuse fits whose weighted design matrix is worse conditioned than this.
    max_condition: float = 1.0e6
    #: RANSAC before the WLS fit.
    ransac: bool = True
    #: Inlier if |residual| <= ransac_threshold_sigma * dh_i.
    ransac_threshold_sigma: float = 3.0
    ransac_iterations: int = 500
    ransac_seed: int = 20260927

    def __post_init__(self) -> None:
        if self.min_constraints < 3:
            raise ConfigError("fit.min_constraints must be >= 3 (two unknowns plus a residual)")
        if not 0.0 < self.reduced_weight_factor <= 1.0:
            raise ConfigError("fit.reduced_weight_factor must be in (0, 1]")
        _positive(self.max_condition, "fit.max_condition")
        _positive(self.ransac_threshold_sigma, "fit.ransac_threshold_sigma")
        _positive(self.ransac_iterations, "fit.ransac_iterations")


@dataclass(frozen=True)
class SyntheticControlConfig:
    #: Phase 0 scene config the control is rendered from.
    scene_config: Path = Path("configs/default.yaml")
    #: Constant c in the constructed field z = log(AGL + 1) + c, giving the
    #: known truth a = exp(-c), b = -1.
    log_offset: float = 0.5
    #: The four fixture buildings are all the constraints there are.
    min_constraints: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_config", Path(self.scene_config))
        if self.min_constraints < 3:
            raise ConfigError("synthetic.min_constraints must be >= 3")


@dataclass(frozen=True)
class Phase4Config:
    run_id: str = "phase4a"
    output_dir: Path = Path("data/outputs/phase4a")
    #: Phase 3 config supplying the dataset root, split and normalisation.
    relative_config: Path = Path("configs/phase3.yaml")
    checkpoint: Path = Path("data/outputs/phase3/checkpoints/best.pt")
    dataset: DfcDataConfig = field(default_factory=DfcDataConfig)
    footprints: FootprintConfig = field(default_factory=FootprintConfig)
    azimuth: AzimuthConfig = field(default_factory=AzimuthConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    synthetic: SyntheticControlConfig = field(default_factory=SyntheticControlConfig)

    def __post_init__(self) -> None:
        for name in ("output_dir", "relative_config", "checkpoint"):
            object.__setattr__(self, name, Path(getattr(self, name)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Phase4Config":
        data = dict(data)
        check_keys(cls, data, "phase4 config root")
        blocks = {
            "dataset": DfcDataConfig,
            "footprints": FootprintConfig,
            "azimuth": AzimuthConfig,
            "measurement": MeasurementConfig,
            "reference": ReferenceConfig,
            "fit": FitConfig,
            "synthetic": SyntheticControlConfig,
        }
        for key, block in blocks.items():
            if key in data:
                data[key] = build_dataclass(block, data[key], key)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase4Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"phase 4 config file not found: {path}")
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


def load_phase4_config(path: str | Path = DEFAULT_PHASE4_CONFIG_PATH) -> Phase4Config:
    return Phase4Config.from_yaml(path)
