"""Phase 5: the validation harness, one command per data source.

::

    # SYNTHETIC: Phase 4b products vs the Phase 0 fixture's specified truth
    python -m depthwizard.validation.run synthetic --config configs/phase5.yaml

    # REAL-WORLD: Phase 4a calibrated AGL on DFC2019 vs DFC2019 lidar AGL,
    # on a geographically held-out city
    python -m depthwizard.validation.run dfc --config configs/phase5.yaml \\
        --relative-config configs/phase3.local.yaml

Both targets run the same steps: load config -> identify the held-out region
-> load predictions -> load references -> validity masks -> terrain/building
masks -> metrics -> error raster -> error figure -> confidence proxy ->
Markdown + JSON report -> a short printed summary.

Neither target ever substitutes one data source for the other. A missing input
raises :class:`~depthwizard.validation.config.ValidationInputError` naming it;
a split that fails the geographic audit raises
:class:`~depthwizard.validation.spatial_split.SplitError` after writing a
report that says real-world validation is not yet measured, and why. The CLI
exits with status 2 in both cases.

The harness evaluates what earlier phases produced; it never trains, fits or
re-calibrates anything. On DFC2019 the Phase 3 field is re-inferred (Phase 4a
did not store it) with the checkpoint named in the Phase 4a report, and each
tile's ``a, b`` is read from that run's ``per_tile.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from depthwizard.config import ConfigError
from depthwizard.logging_setup import get_logger
from depthwizard.validation.config import (
    Phase5Config,
    ValidationError,
    ValidationInputError,
    load_phase5_config,
)
from depthwizard.validation.confidence import ConfidenceResult, ConfidenceState, confidence_proxy
from depthwizard.validation.error_map import error_array, plot_error_map, write_error_raster
from depthwizard.validation.evaluation import NOT_MEASURED
from depthwizard.validation.masks import (
    CATEGORIES,
    EXCLUSION_ORDER,
    CategoryMasks,
    ValidityMask,
    build_valid_mask,
    category_masks_from_labels,
)
from depthwizard.validation.metrics import MetricAccumulator, RegionMetrics
from depthwizard.validation.report import write_markdown_report
from depthwizard.validation.spatial_split import (
    CheckpointSelection,
    SplitError,
    audit_model_selection,
    audit_region_split,
)

__all__ = ["SYNTHETIC", "REAL_WORLD", "ProductAccumulator", "run_synthetic", "run_dfc", "main"]

log = get_logger(__name__)

SYNTHETIC = "SYNTHETIC"
REAL_WORLD = "REAL-WORLD"
SYNTHETIC_BANNER = (
    "heights were SPECIFIED when the fixture was generated, not measured. These numbers test the "
    "validation harness and the pipeline's geometry; they are NOT a real-world accuracy claim."
)
REAL_BANNER = "DFC2019 lidar AGL on a geographically held-out region."
FRAMING = "City-held-out model evaluation with LiDAR-anchored per-tile metric calibration."


def _measured_banner(train_regions: Sequence[str], heldout_regions: Sequence[str]) -> str:
    """The measured-run banner. Region names come from the audited split, never from constants."""
    train, heldout = ", ".join(train_regions), ", ".join(heldout_regions)
    return (
        f"{REAL_BANNER} {FRAMING} Training region: {train} (the Phase 3 model was trained on {train} only). "
        f"Held-out region: {heldout}, excluded from Phase 3 training and from checkpoint selection (both audited). "
        f"{heldout} LiDAR is used for per-tile metric calibration (2-3 reference buildings per tile), and those "
        "calibration reference buildings are excluded from scoring. This is NOT an unseen-city zero-shot result."
    )
REPORT_JSON = "phase5_report.json"
REPORT_MD = "validation_report.md"

#: Called as ``predict(pair, row_off, col_off, size)``; returns the tile dict
#: documented on :func:`_model_predictor`.
TilePredictor = Callable[[Any, int, int, int], Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class ProductAccumulator:
    """Streams (prediction, reference) tiles of one product into every metric."""

    def __init__(self, product: str, units: str, description: str, thresholds: Sequence[float]) -> None:
        self.product = product
        self.units = units
        self.description = description
        self.thresholds = tuple(thresholds)
        self.categories = {c: MetricAccumulator(self.thresholds) for c in CATEGORIES}
        self.category_seen = {c: False for c in CATEGORIES}
        self.category_unavailable: dict[str, str] = {}
        self.by_confidence = {
            s.name: MetricAccumulator(self.thresholds) for s in ConfidenceState if s is not ConfidenceState.INVALID
        }
        self.exclusions = {name: 0 for name in EXCLUSION_ORDER}
        self.total_pixels = 0
        self.valid_pixels = 0
        self.confidence_diagnostics: list[dict[str, Any]] = []

    def add(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
        validity: ValidityMask,
        categories: CategoryMasks,
        confidence: ConfidenceResult,
    ) -> None:
        valid = validity.valid
        for category in CATEGORIES:
            mask = categories.get(category)
            if mask is None:
                self.category_unavailable.setdefault(
                    category, categories.definitions.get(category) or "the data has no label for this category")
                continue
            self.category_seen[category] = True
            self.categories[category].add(prediction, reference, valid & mask)
        for name, acc in self.by_confidence.items():
            acc.add(prediction, reference, valid & (confidence.state == int(ConfidenceState[name])))
        for name, count in validity.exclusions.items():
            self.exclusions[name] += count
        self.total_pixels += validity.total_pixels
        self.valid_pixels += validity.valid_pixels
        self.confidence_diagnostics.append(confidence.diagnostics)

    def _region(self, category: str) -> RegionMetrics:
        if not self.category_seen[category]:
            return RegionMetrics.unavailable(
                category, self.units, self.category_unavailable.get(category, "no data"), self.thresholds)
        return self.categories[category].result(category=category, units=self.units)

    def result(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "units": self.units,
            "description": self.description,
            "validity": {
                "total_pixels": self.total_pixels,
                "valid_pixels": self.valid_pixels,
                "excluded_pixels": dict(self.exclusions),
                "exclusion_precedence": list(EXCLUSION_ORDER),
            },
            "metrics": {c: self._region(c).to_dict() for c in CATEGORIES},
            "metrics_by_confidence": {
                name: acc.result(category=f"confidence:{name}", units=self.units).to_dict()
                for name, acc in self.by_confidence.items()
            },
            "confidence": pool_confidence(self.confidence_diagnostics),
        }


def pool_confidence(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-tile confidence diagnostics. A check absent on every tile stays absent."""
    if not diagnostics:
        return {"status": NOT_MEASURED}
    if len(diagnostics) == 1:
        return dict(diagnostics[0])
    first = diagnostics[0]
    valid = sum(d["valid_pixels"] for d in diagnostics)
    counts = {name: sum(d["state_pixels"][name] for d in diagnostics) for name in first["state_pixels"]}
    checks: dict[str, Any] = {}
    for name in first["checks"]:
        records = [d["checks"][name] for d in diagnostics]
        available = [r for r in records if r["available"]]
        if not available:
            checks[name] = dict(records[0])
            continue
        flagged = sum(int(r["flagged_valid_pixels"]) for r in available)
        merged = dict(available[0])
        merged["flagged_valid_pixels"] = flagged
        merged["flagged_fraction_of_valid"] = flagged / valid if valid else NOT_MEASURED
        if len(available) < len(records):
            merged["status"] = f"available on {len(available)} of {len(records)} tiles"
        if name == "sun_band" and len({r.get("sun_elevation_deg") for r in available}) > 1:
            merged["sun_elevation_deg"] = "varies by tile"
            merged["sun_band"] = "varies by tile"
        checks[name] = merged
    return {
        **{k: first[k] for k in ("kind", "states", "precedence")},
        "valid_pixels": valid,
        "state_pixels": counts,
        "state_fraction_of_valid": {k: (v / valid if valid else NOT_MEASURED) for k, v in counts.items() if k != "INVALID"},
        "all_checks_available": all(d["all_checks_available"] for d in diagnostics),
        "checks": checks,
        "overlaps": {"shadow_and_sun_out_of_band_valid_pixels":
                     sum(d["overlaps"]["shadow_and_sun_out_of_band_valid_pixels"] for d in diagnostics)},
        "tiles": len(diagnostics),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, *, what: str, hint: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationInputError(f"{what} not found: {path}\n  {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, what: str, hint: str = "") -> Path:
    if not path.is_file():
        raise ValidationInputError(f"{what} not found: {path}" + (f"\n  {hint}" if hint else ""))
    return path


def _git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "not available (git not reachable)"
    commit = head.stdout.strip()
    return f"{commit} (working tree has uncommitted changes)" if status.stdout.strip() else commit


def _write_confidence_raster(path: Path, confidence: ConfidenceResult, *, crs: Any, transform: Any) -> str:
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    from depthwizard.ingest.geotiff import write_single_band

    tags = {f"STATE_{s.value}": s.name for s in ConfidenceState}
    tags["CONTENT"] = "Phase 5 confidence proxy (rule-based flags, NOT a probability)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        write_single_band(path, confidence.state, crs=crs or None, transform=transform, tags=tags)
    return str(path)


def _finish(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from depthwizard.calibration.diagnostics import write_json

    outputs = result.setdefault("reproducibility", {}).setdefault("outputs", {})
    outputs["json_report"] = str(out_dir / REPORT_JSON)
    outputs["markdown_report"] = str(out_dir / REPORT_MD)
    write_json(out_dir / REPORT_JSON, result)
    write_markdown_report(out_dir / REPORT_MD, result)
    log.info("wrote Phase 5 report", extra={"markdown": str(out_dir / REPORT_MD)})
    return result


# ---------------------------------------------------------------------------
# SYNTHETIC
# ---------------------------------------------------------------------------


def _same_grid(a: Any, b: Any, what: str) -> None:
    if a.array.shape != b.array.shape or a.crs != b.crs or not np.allclose(
            tuple(a.transform)[:6], tuple(b.transform)[:6], rtol=0.0, atol=1e-9):
        raise ValidationInputError(
            f"{what} is not on the reference grid: shape {a.array.shape} vs {b.array.shape}, "
            f"CRS {a.crs} vs {b.crs}, transform {tuple(a.transform)[:6]} vs {tuple(b.transform)[:6]}"
        )


def run_synthetic(cfg: Phase5Config, *, command: str = "", config_path: str | Path = "") -> dict[str, Any]:
    """Score the Phase 4b SYNTHETIC products against the fixture's specified truth."""
    from depthwizard.config import load_config
    from depthwizard.ingest.geotiff import read_raster, read_sun_metadata
    from depthwizard.ingest.synthetic import TerrainSpec, terrain_truth_on_grid
    from depthwizard.shadows.detector import ClassicalShadowDetector, DetectionContext

    s = cfg.synthetic
    out = cfg.output_dir / "synthetic"
    hint = "run `python -m depthwizard.surfaces.run synthetic --config configs/phase4.yaml` first"
    report4b = _read_json(s.phase4b_report, what="Phase 4b report", hint=hint)
    fixture_image = _require_file(Path(report4b["image_grid"]["source"]), "Phase 4b fixture image", hint)
    stem = fixture_image.stem
    truth_path = _require_file(fixture_image.with_name(f"{stem}_truth.json"), "fixture truth sidecar", hint)
    truth_height_path = _require_file(fixture_image.with_name(f"{stem}_truth_height.tif"), "fixture truth height", hint)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if truth.get("synthetic") is not True:
        raise ValidationInputError(f"{truth_path} is not marked synthetic; refusing to label it SYNTHETIC")
    scene = load_config(s.scene_config).scene
    if scene.name != truth.get("scene_name"):
        raise ValidationInputError(
            f"{s.scene_config} describes scene {scene.name!r}, but the fixture is {truth.get('scene_name')!r}")

    agl_truth = read_raster(truth_height_path)
    shape = agl_truth.array.shape
    building = np.zeros(shape, dtype=bool)
    for b in truth["buildings"]:
        fp = b["footprint_px"]
        building[fp["row_min"]:fp["row_max"], fp["col_min"]:fp["col_max"]] = True
    categories = CategoryMasks(
        shape=shape, terrain=~building, building=building,
        source=f"the fixture's truth sidecar ({truth_path.name}): specified building footprints",
        definitions={"overall": "every valid pixel", "terrain": "outside every specified footprint (bare ground, "
                     "including shadowed ground)", "building": "inside a specified building footprint"},
    )
    image = read_raster(fixture_image)
    _same_grid(image, agl_truth, "fixture image")
    shadow = ClassicalShadowDetector().detect(image.array, DetectionContext(gsd_m=float(truth["raster"]["gsd_m"])))
    sun_elevation = read_sun_metadata(fixture_image).elevation_deg

    products_4b = report4b["products"]
    references = {
        "agl": (
            "synthetic_agl.tif", agl_truth.array.astype(np.float64),
            "calibrated above-ground height, a*exp(z_rel)+b with the Phase 4a SYNTHETIC calibration",
            f"specified building heights ({truth_height_path.name}); 0 m on bare ground",
        ),
    }
    if "dsm" in s.products:
        terrain = terrain_truth_on_grid(scene, TerrainSpec(**report4b["dem"]["terrain_spec"]))
        references["dsm"] = (
            "synthetic_dsm.tif", terrain + agl_truth.array.astype(np.float64),
            "DSM = T + a*exp(z_rel)+b (Phase 4b, Phase 4a SYNTHETIC calibration)",
            "analytic terrain plane (TerrainSpec from the Phase 4b report) + specified building heights",
        )

    out.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, Any]] = []
    for product in s.products:
        filename, reference, description, reference_description = references[product]
        pred_path = _require_file(Path(products_4b[filename]), f"Phase 4b product {filename}", hint)
        pred = read_raster(pred_path)
        _same_grid(pred, agl_truth, f"Phase 4b product {filename}")
        units = pred.tags.get("UNITS")
        if not units:
            raise ValidationInputError(f"{pred_path} carries no UNITS tag; refusing to assume its units")

        validity = build_valid_mask(pred.array, reference, prediction_nodata=pred.nodata,
                                    reference_nodata=agl_truth.nodata)
        confidence = confidence_proxy(
            validity.valid, shadow_mask=shadow.mask, water_mask=None, sun_elevation_deg=sun_elevation,
            shadow_source=f"Phase 2 ClassicalShadowDetector ({shadow.method}) on {fixture_image.name}",
            water_unavailable_reason="the synthetic fixture carries no land-cover labels (no water class)",
            sun_source=f"SUN_ELEVATION_DEG tag of {fixture_image.name}",
            sun_band_min_deg=cfg.sun_band.min_deg, sun_band_max_deg=cfg.sun_band.max_deg,
        )
        acc = ProductAccumulator(product, units, description, cfg.delta_thresholds)
        acc.add(pred.array, reference, validity, categories, confidence)
        entry = acc.result()
        entry["prediction_path"] = str(pred_path)
        entry["reference"] = reference_description

        error = error_array(pred.array, reference, validity.valid)
        raster = write_error_raster(
            out / f"error_map_{product}.tif", error, crs=pred.crs, transform=pred.transform,
            nodata=cfg.error_map.nodata, tags={"UNITS": units, "SYNTHETIC": "true", "PRODUCT": product},
        )
        figure = plot_error_map(
            out / f"error_map_{product}.png", error, units=units, crs=pred.crs, transform=pred.transform,
            title=f"SYNTHETIC - {product.upper()} error (prediction - reference)\nnot a real-world accuracy result",
            display_percentile=cfg.error_map.display_percentile,
        )
        entry["error_map"] = {
            "description": f"{product.upper()} prediction minus reference, per pixel, on the prediction's own grid "
                           "(positive = over-prediction); invalid pixels are nodata",
            "raster": raster, "figure": figure,
        }
        entry["confidence_raster"] = _write_confidence_raster(
            out / f"confidence_{product}.tif", confidence, crs=pred.crs, transform=pred.transform)
        products.append(entry)

    result = {
        "phase": "5",
        "validation_kind": SYNTHETIC,
        "status": "measured",
        "label": SYNTHETIC_BANNER,
        "real_world_validation": f"{NOT_MEASURED} (this run is SYNTHETIC; see the `dfc` target)",
        "dataset": {
            "name": f"SYNTHETIC fixture {truth['scene_name']} (Phase 0 generator, via Phase 4b)",
            "description": f"{shape[1]} x {shape[0]} px, {truth['raster']['crs']}, GSD {truth['raster']['gsd_m']} m, "
                           f"sun {truth['sun']['elevation_deg']} / {truth['sun']['azimuth_deg']} deg, "
                           f"{len(truth['buildings'])} buildings",
            "training_region": "none: nothing is trained on the synthetic fixture (the relative field is "
                               "constructed from its truth; a, b come from its own shadows)",
            "heldout_region": f"{truth['scene_name']} (the whole fixture; the only synthetic scene)",
            "split": {
                "rationale": "not applicable: one synthetic scene and no learned component, so there is "
                             "nothing to hold out from",
                "verification": "not applicable: no checkpoint is involved",
                "type": "none",
                "random_split": False,
            },
            "evaluated_tiles": 1,
            "evaluated_images": 1,
            "reference": "fixture truth: SPECIFIED heights (and, for the DSM, the analytic terrain plane)",
            "prediction": f"Phase 4b SYNTHETIC products ({', '.join(s.products)}); relative field: "
                          f"{report4b.get('relative_field', {}).get('z_rel', NOT_MEASURED)}",
        },
        "categories": {"source": categories.source, "definitions": dict(categories.definitions)},
        "products": products,
        "calibration": {k: {"a": v["a"], "b": v["b"]} for k, v in report4b.get("calibrations", {}).items()},
        "reproducibility": {
            "command": command or "python -m depthwizard.validation.run synthetic",
            "config_path": str(config_path),
            "git_commit": _git_commit(),
            "checkpoint": "none (no network is involved in the SYNTHETIC path)",
            "inputs": {"phase4b_report": str(s.phase4b_report), "fixture_image": str(fixture_image),
                       "truth": str(truth_path), "truth_height": str(truth_height_path)},
            "config": cfg.to_dict(),
        },
        "limitations": [
            "SYNTHETIC: the fixture's heights were specified, not measured. These metrics validate the harness "
            "and the pipeline's geometry, not real-world accuracy.",
            "The relative field is the Phase 4a/4b constructed stand-in, not a Phase 3 network prediction.",
            "No training/held-out split applies: one synthetic scene, nothing trained on it.",
            "Water: not available on the fixture (no land-cover labels), so no pixel can be HIGH confidence; "
            "clean pixels are NOT_ASSESSED.",
            "δ metrics only cover pixels where prediction and reference are both > 0; bare-ground AGL is 0 m, "
            "so terrain δ for AGL is not computable by definition.",
            f"Real-world validation: {NOT_MEASURED}.",
        ],
    }
    return _finish(result, out)


# ---------------------------------------------------------------------------
# REAL-WORLD: DFC2019
# ---------------------------------------------------------------------------


def read_checkpoint(path: Path) -> tuple[Any, CheckpointSelection]:
    """The :class:`~depthwizard.relative.config.RelativeConfig` a checkpoint embeds,
    and which epoch it holds."""
    import torch

    from depthwizard.relative.config import RelativeConfig

    payload = torch.load(_require_file(path, "Phase 3 checkpoint"), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "config" not in payload:
        raise ValidationInputError(f"{path} embeds no config; its training split cannot be verified")
    return RelativeConfig.from_dict(payload["config"]), CheckpointSelection.from_payload(payload)


def _model_predictor(checkpoint: Path, device: str | None) -> TilePredictor:
    """Phase 3 inference for one tile.

    Returns a callable producing ``{z_rel, log_space, agl, agl_valid, rgb_u8,
    transform, crs}`` -- the same arrays Phase 4a calibrated on.
    """
    from depthwizard.relative.inference import infer_tile, load_model_from_checkpoint

    model, model_cfg = load_model_from_checkpoint(checkpoint)

    def predict(pair: Any, row: int, col: int, size: int) -> dict[str, Any]:
        prediction = infer_tile(model, model_cfg, pair.image_path, height_path=pair.height_path,
                                row_off=row, col_off=col, tile_size=size, device=device)
        if prediction.is_fallback or prediction.fallback_kind:
            raise ValidationInputError("Phase 5 evaluates the trained Phase 3 model, not a fallback")
        return {
            "z_rel": prediction.relative_height,
            "log_space": prediction.log_space,
            "agl": prediction.ground_truth,
            "agl_valid": prediction.valid,
            "rgb_u8": np.clip(np.rint(prediction.image_rgb * 255.0), 0, 255).astype(np.uint8),
            "transform": prediction.metadata.get("transform"),
            "crs": prediction.metadata.get("crs") or None,
        }

    return predict


def _refused(cfg: Phase5Config, out: Path, reason: str, *, dataset: dict[str, Any],
             reproducibility: dict[str, Any]) -> dict[str, Any]:
    result = {
        "phase": "5",
        "validation_kind": REAL_WORLD,
        "status": "refused",
        "status_reason": reason,
        "label": REAL_BANNER,
        "real_world_validation": NOT_MEASURED,
        "dataset": dataset,
        "products": [],
        "reproducibility": reproducibility,
        "limitations": [f"Real-world validation: {NOT_MEASURED}. {reason}"],
    }
    out.mkdir(parents=True, exist_ok=True)
    return _finish(result, out)


def _selected_tiles(per_tile: Path, heldout_scenes: set[str], heldout: set[str],
                    fit_classes: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Calibrated held-out tiles from a Phase 4a ``per_tile.jsonl``, plus skip counts."""
    selected: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with per_tile.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("city") not in heldout or record.get("scene_id") not in heldout_scenes:
                skip("not in the held-out region")
            elif record.get("status") != "calibrated":
                skip(f"Phase 4a status {record.get('status')}")
            elif record["fit"]["identifiability"] not in fit_classes:
                skip(f"fit label {record['fit']['identifiability']} not selected")
            else:
                selected.append({k: record[k] for k in ("tile_id", "city", "image", "scene_id", "row_off", "col_off")}
                                | {"a": record["fit"]["fit"]["a"], "b": record["fit"]["fit"]["b"],
                                   "identifiability": record["fit"]["identifiability"],
                                   "reference_ids": [r["building_id"] for r in record["sun_calibration"]["references"]]})
    return sorted(selected, key=lambda r: r["tile_id"]), dict(sorted(skipped.items()))


def run_dfc(
    cfg: Phase5Config,
    *,
    relative_config: str | Path | None = None,
    device: str | None = None,
    command: str = "",
    config_path: str | Path = "",
    tile_predictor: TilePredictor | None = None,
) -> dict[str, Any]:
    """Score Phase 4a calibrated AGL on a held-out DFC2019 region.

    Args:
        cfg: Phase 5 config; ``cfg.dfc`` must declare the split.
        relative_config: Phase 3 config supplying the local DFC2019 root
            (defaults to the one the Phase 4a run recorded).
        device: Inference device (default: the checkpoint config's).
        tile_predictor: Injected per-tile predictor (tests); default runs the
            Phase 3 checkpoint named by the Phase 4a report.

    Raises:
        ValidationInputError: a required input is missing or inconsistent.
        SplitError: the declared split fails the audit (a refusal report is
            written first).
    """
    from depthwizard.calibration.config import Phase4Config
    from depthwizard.calibration.footprints import extract_building_footprints
    from depthwizard.calibration.fusion import CalibrationError, apply_affine, linearise_relative
    from depthwizard.relative.config import load_relative_config
    from depthwizard.relative.data import DatasetError, discover_pairs, open_raster
    from depthwizard.shadows.detector import ClassicalShadowDetector, DetectionContext
    from rasterio.transform import Affine

    if cfg.dfc is None:
        raise ValidationInputError("the Phase 5 config declares no `dfc` block, so there is no real-data split")
    d = cfg.dfc
    out = cfg.output_dir / "dfc"
    hint = ("run the held-out Phase 4a calibration first: python -m depthwizard.calibration.run dfc "
            "--config configs/phase4.yaml --relative-config configs/phase3.local.yaml")
    report4a = _read_json(d.phase4a_dir / "phase4a_report.json", what="Phase 4a report", hint=hint)
    per_tile = _require_file(d.phase4a_dir / "per_tile.jsonl", "Phase 4a per_tile.jsonl", hint)
    if report4a.get("split", {}).get("side") != "val":
        raise ValidationInputError(
            f"{d.phase4a_dir} is a Phase 4a run on the TRAINING side ({report4a.get('split')}); it is not held out")
    p4 = Phase4Config.from_dict(report4a["config"])
    checkpoint = p4.checkpoint
    ckpt_cfg, selection = read_checkpoint(checkpoint)
    rel_cfg = load_relative_config(relative_config or p4.relative_config)
    dataset_cfg = replace(ckpt_cfg.dataset, root=rel_cfg.dataset.root, image_subdir=rel_cfg.dataset.image_subdir,
                          height_subdir=rel_cfg.dataset.height_subdir)
    try:
        pairs = discover_pairs(dataset_cfg)
    except DatasetError as exc:
        raise ValidationInputError(f"DFC2019 data unavailable: {exc}\n  pass --relative-config "
                                   "configs/phase3.local.yaml pointing at your local copy") from exc

    reproducibility = {
        "command": command or "python -m depthwizard.validation.run dfc",
        "config_path": str(config_path),
        "git_commit": _git_commit(),
        "checkpoint": f"{checkpoint} (trained with split.mode={ckpt_cfg.dataset.split.mode!r})",
        "inputs": {"phase4a_report": str(d.phase4a_dir / "phase4a_report.json"), "per_tile": str(per_tile),
                   "dataset_root": str(dataset_cfg.root)},
        "config": cfg.to_dict(),
    }
    dataset = {
        "name": "IEEE GRSS DFC2019 Track 1 (US3D): RGB tiles, lidar AGL and CLS labels",
        "training_region": ", ".join(d.split.train_regions) + " (declared)",
        "heldout_region": ", ".join(d.split.heldout_regions) + " (declared)",
        "split": {"rationale": d.split.rationale, "region_key": d.split.region_key,
                  "checkpoint_split_mode": ckpt_cfg.dataset.split.mode,
                  "verification": "FAILED - see the refusal reason"},
        "reference": "DFC2019 lidar-derived AGL (metres), CLS labels for terrain/building/water",
        "prediction": "Phase 4a calibrated AGL = a*exp(z_rel)+b per 512 px tile",
    }
    try:
        audit = audit_region_split(d.split, pairs, ckpt_cfg.dataset.split)
        selection_record = audit_model_selection(selection, audit)
    except SplitError as exc:
        log.error("real-world validation refused", extra={"reason": str(exc)})
        _refused(cfg, out, str(exc), dataset=dataset, reproducibility=reproducibility)
        raise

    tiles, skipped = _selected_tiles(per_tile, set(audit.heldout_scene_ids), set(audit.heldout_regions), d.fit_classes)
    if not tiles:
        raise ValidationInputError(f"{per_tile} has no calibrated tile in held-out region(s) "
                                   f"{list(audit.heldout_regions)} (skipped: {skipped})")
    predict = tile_predictor or _model_predictor(checkpoint, device)
    by_stem = {p.stem: p for p in pairs}
    size = p4.dataset.crop_size
    units = "metres"
    acc = ProductAccumulator("agl", units, "Phase 4a calibrated above-ground height on DFC2019", cfg.delta_thresholds)
    detector = ClassicalShadowDetector()
    error_dir = out / "error_maps"
    figures: list[dict[str, Any]] = []
    rasters: list[dict[str, Any]] = []
    cls_cache: dict[str, np.ndarray] = {}
    sun_reason = ("DFC2019 Track 1 carries no sun metadata, and the sun elevation is not determinable from the "
                  "imagery because the tile GSD is unverified (Phase 4a, configs/phase4.yaml gsd_note)")

    for index, tile in enumerate(tiles):
        pair = by_stem.get(tile["image"])
        if pair is None:
            raise ValidationInputError(f"Phase 4a tile {tile['tile_id']} names image {tile['image']!r}, "
                                       "which is not in the dataset")
        if pair.stem not in cls_cache:
            cls_cache.clear()  # one image at a time: tiles are sorted by image
            cls_path = pair.height_path.with_name(pair.height_path.name[: -len(dataset_cfg.height_suffix)] + d.cls_suffix)
            with open_raster(_require_file(cls_path, "DFC2019 CLS raster")) as src:
                cls_cache[pair.stem] = src.read(1)
        row, col = int(tile["row_off"]), int(tile["col_off"])
        cls = cls_cache[pair.stem][row: row + size, col: col + size]
        arrays = predict(pair, row, col, size)
        try:
            prediction = apply_affine(linearise_relative(arrays["z_rel"], log_space=arrays["log_space"]),
                                      tile["a"], tile["b"])
        except CalibrationError as exc:
            raise ValidationInputError(f"{tile['tile_id']}: {exc}") from exc
        reference = np.asarray(arrays["agl"], dtype=np.float64)
        if cls.shape != prediction.shape:
            raise ValidationInputError(f"{tile['tile_id']}: CLS {cls.shape} != prediction {prediction.shape}")

        footprints = extract_building_footprints(
            cls, tile_id=tile["tile_id"], building_class=p4.dataset.building_class,
            min_area_px=p4.footprints.min_area_px, connectivity=p4.footprints.connectivity,
            reject_edge_touching=p4.footprints.reject_edge_touching,
        ).footprints
        masks = {fp.building_id: fp.mask for fp in footprints}
        missing = [i for i in tile["reference_ids"] if i not in masks]
        if missing:
            raise ValidationInputError(f"{tile['tile_id']}: reference building(s) {missing} not re-extracted; "
                                       "the Phase 4 footprint settings differ from the Phase 4a run")
        exclude = np.zeros(prediction.shape, dtype=bool)
        for building_id in tile["reference_ids"]:
            exclude |= masks[building_id]

        validity = build_valid_mask(prediction, reference, reference_valid=arrays["agl_valid"], exclude=exclude)
        categories = category_masks_from_labels(cls, terrain_classes=d.terrain_classes,
                                                building_classes=d.building_classes, source="DFC2019 CLS labels")
        shadow = detector.detect(arrays["rgb_u8"], DetectionContext(band_order="rgb"))
        confidence = confidence_proxy(
            validity.valid, shadow_mask=shadow.mask, water_mask=np.isin(cls, d.water_classes),
            sun_elevation_deg=None,
            shadow_source=f"Phase 2 ClassicalShadowDetector ({shadow.method}) on the RGB tile",
            water_source=f"DFC2019 CLS in {list(d.water_classes)}", sun_unavailable_reason=sun_reason,
            sun_band_min_deg=cfg.sun_band.min_deg, sun_band_max_deg=cfg.sun_band.max_deg,
        )
        acc.add(prediction, reference, validity, categories, confidence)

        transform = Affine(*arrays["transform"]) if arrays.get("transform") else Affine.identity()
        error = error_array(prediction, reference, validity.valid)
        rasters.append(write_error_raster(
            error_dir / f"{tile['tile_id']}.tif", error, crs=arrays.get("crs"), transform=transform,
            nodata=cfg.error_map.nodata, tags={"UNITS": units, "SYNTHETIC": "false", "TILE": tile["tile_id"],
                                               "CALIB_A": tile["a"], "CALIB_B": tile["b"]},
        ))
        if len(figures) < cfg.error_map.max_figures:
            drawn = plot_error_map(
                error_dir / f"{tile['tile_id']}.png", error, units=units, crs=arrays.get("crs"), transform=transform,
                title=f"DFC2019 {tile['tile_id']} ({tile['identifiability']}) - AGL error (prediction - reference)",
                display_percentile=cfg.error_map.display_percentile,
            )
            if drawn:
                figures.append(drawn)
        if (index + 1) % 25 == 0 or index + 1 == len(tiles):
            log.info("phase 5 progress", extra={"tiles_done": index + 1, "tiles": len(tiles)})

    entry = acc.result()
    entry["reference"] = "DFC2019 lidar AGL; reference-building pixels (used to fit each tile's sun scale) excluded"
    entry["error_map"] = {
        "description": "per-tile AGL prediction minus DFC2019 AGL (positive = over-prediction), in each tile's "
                       "pixel grid; DFC2019 Track 1 has no CRS, so none is written",
        "raster": rasters[0], "raster_dir": str(error_dir), "rasters_written": len(rasters),
        "figure": figures[0] if figures else None, "figures": [f["path"] for f in figures],
        "figure_selection": "the first tiles in sorted tile-id order (never chosen by accuracy)",
    }
    identifiability: dict[str, int] = {}
    for tile in tiles:
        identifiability[tile["identifiability"]] = identifiability.get(tile["identifiability"], 0) + 1
    dataset.update(
        evaluated_tiles=len(tiles),
        evaluated_images=len({t["image"] for t in tiles}),
        split={**audit.to_dict(), "model_selection": selection_record,
               "verification": "PASSED: no held-out region in the checkpoint's training pairs, and the "
                               "checkpoint was not selected with a held-out validation loss"},
        training_region=", ".join(audit.train_regions),
        heldout_region=", ".join(audit.heldout_regions),
        phase4a_tiles_skipped=skipped,
        fit_labels_evaluated=identifiability,
    )
    result = {
        "phase": "5",
        "validation_kind": REAL_WORLD,
        "status": "measured",
        "label": _measured_banner(audit.train_regions, audit.heldout_regions),
        "real_world_validation": "measured on the held-out region below",
        "dataset": dataset,
        "categories": {"source": "DFC2019 CLS labels", "definitions": {
            "overall": "every valid pixel, whatever its CLS label",
            "terrain": f"CLS in {list(d.terrain_classes)}", "building": f"CLS in {list(d.building_classes)}"}},
        "products": [entry],
        "reproducibility": reproducibility,
        "limitations": [
            "Only tiles where Phase 4a returned a shadow calibration are evaluated; the rest of the held-out "
            f"region is not measured (skipped: {skipped}).",
            "Sun band: not available on DFC2019 (no sun metadata; GSD unverified), so no pixel can be HIGH.",
            "The shadow flag uses the Phase 2 classical detector, which also marks dark asphalt on DFC2019.",
            "DFC2019 Track 1 has no georeferencing: error maps are per-tile pixel grids, not a mosaic.",
            "Phase 4a fixes each tile's sun scale from the DFC2019 lidar AGL of 2-3 reference buildings INSIDE "
            "that held-out tile. Their pixels are excluded from every metric here, but that ground truth does "
            "enter the calibration, so these are not fully truth-free held-out predictions.",
            "Phase 4a thresholds are used exactly as frozen in the config the Phase 4a run records; this harness "
            "does not itself verify which data they were selected on. They must be selected on training-region "
            "data only, and that provenance (e.g. a training-region-only ablation and its frozen decisions) must "
            "be recorded alongside this report.",
        ],
    }
    return _finish(result, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(result: Mapping[str, Any]) -> str:
    lines = [f"{result['validation_kind']} VALIDATION - status: {result['status']}"]
    if result.get("status_reason"):
        lines.append(f"  refused: {result['status_reason']}")
    lines.append(f"  real-world validation: {result.get('real_world_validation')}")
    for product in result.get("products", []):
        lines.append(f"  [{product['product']}] units={product['units']} "
                     f"valid={product['validity']['valid_pixels']:,}/{product['validity']['total_pixels']:,} px")
        for category in CATEGORIES:
            m = product["metrics"][category]

            def show(metric: Mapping[str, Any]) -> str:
                value = metric["value"]
                return f"{value:.4f}" if isinstance(value, float) else "n/a"

            deltas = " ".join(f"{x['name']}={show(x)}" for x in m["deltas"])
            lines.append(f"    {category:<8} n={m['valid_pixels']:>9,}  MAE={show(m['mae'])}  RMSE={show(m['rmse'])}  "
                         f"r={show(m['pearson_r'])}  {deltas}")
    outputs = result.get("reproducibility", {}).get("outputs", {})
    lines.append(f"  report: {outputs.get('markdown_report', NOT_MEASURED)}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    from depthwizard.logging_setup import setup_logging

    argv = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description="DepthWizard Phase 5: validation harness")
    parser.add_argument("target", choices=("synthetic", "dfc"))
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--relative-config", default=None,
                        help="dfc: Phase 3 config with the local DFC2019 root (e.g. configs/phase3.local.yaml)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    setup_logging({"level": "INFO", "format": args.log_format})

    import sys

    command = "python -m depthwizard.validation.run " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        cfg = load_phase5_config(args.config)
        if args.output_dir is not None:
            cfg = replace(cfg, output_dir=Path(args.output_dir))
        if args.target == "synthetic":
            result = run_synthetic(cfg, command=command, config_path=args.config)
        else:
            result = run_dfc(cfg, relative_config=args.relative_config, device=args.device,
                             command=command, config_path=args.config)
    except (ValidationError, ConfigError) as exc:
        kind = "REAL-WORLD" if args.target == "dfc" else "SYNTHETIC"
        print(f"{kind} VALIDATION FAILED: {exc}")
        print(f"Real-world validation: {NOT_MEASURED}")
        return 2
    print(_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
