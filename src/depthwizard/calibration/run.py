"""Phase 4a: pixel-space metric calibration, run end to end.

Entry points::

    # Real DFC2019 Track 1, held-out Phase 3 validation tiles
    python -m depthwizard.calibration.run dfc \\
        --config configs/phase4.yaml --relative-config configs/phase3.local.yaml

    # The same pipeline on TRAINING tiles (diagnostics only; Phase 3 saw them)
    python -m depthwizard.calibration.run dfc --split-side train --views-per-scene 4 \\
        --output-dir data/outputs/phase4a_train_diagnostics ...

    # TRAINING-only threshold ablations (no model needed)
    python -m depthwizard.calibration.run ablation ...

    # SYNTHETIC control on the Phase 0 fixture (true sun elevation)
    python -m depthwizard.calibration.run synthetic --config configs/phase4.yaml

Per 512 px tile (the Phase 3 inference tile, and the calibration unit)
---------------------------------------------------------------------
1. Phase 3 relative field ``z_rel`` (log space) for the tile.
2. Building footprints: CLS == 6 -> connected components -> min-area filter.
3. Shadow mask: the Phase 2 classical detector, restricted to pixels whose CLS
   label is a shadow-receiving surface (ground by default).
4. Sun azimuth from footprint-adjacent shadow occupancy.
5. Phase 2 shadow lengths for every footprint, plus the Phase 4a validity
   rules (edge truncation, merged shadow).
6. Reference buildings, selected automatically; ``metres_per_shadow_px`` (=
   ``GSD * tan(theta)``) from their DFC2019 AGL.
7. Shadow heights for all other buildings; RANSAC then WLS for ``a, b``; the
   fit is labelled with an identifiability class (never gated on ``a > 0``).
8. ``AGL_pred = a * exp(z_rel) + b`` scored against DFC2019 AGL, with the
   reference buildings' pixels excluded.

Also per tile: the ORACLE control (``a, b`` fitted directly to ground-truth AGL
on the same buildings -- an upper-bound diagnostic, not a deployable method),
the reference-count sensitivity (2 / 3 / 5), and the two uncertainty forms.

Every tile record carries additive error sums, so the report can be grouped
by city, tile height group or identifiability class without re-running.

Leakage rules enforced here
---------------------------
* The held-out run processes only the Phase 3 **validation** split
  (tile-disjoint from training; ``dataset.split_side``).
* Reference buildings are excluded from every constraint set and every metric.
* Evaluation buildings' AGL is used only to score them (and by the oracle).
  The tile height group is computed from AGL too, and is only ever a report
  grouping, never an input to the method.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from depthwizard.calibration.azimuth import (
    angular_difference_deg,
    angular_resolution_deg,
    estimate_sun_azimuth,
)
from depthwizard.calibration.config import Phase4Config, load_phase4_config
from depthwizard.calibration.diagnostics import (
    ErrorAccumulator,
    binned_errors,
    count_reasons,
    dense_error_sums,
    distribution,
    height_distribution,
    plot_tile_diagnostic,
    to_jsonable,
    write_json,
)
from depthwizard.calibration.footprints import (
    DFC2019_BUILDING_CLASS_SOURCE,
    extract_building_footprints,
)
from depthwizard.calibration.fusion import (
    IDENTIFIABILITY_CLASSES,
    IDENTIFIABLE_T,
    AffineFit,
    CalibrationError,
    apply_affine,
    classify_failure,
    classify_fit,
    fit_affine_wls,
    footprint_means,
    linearise_relative,
)
from depthwizard.calibration.ransac import ransac_affine
from depthwizard.calibration.shadow_anchor import (
    PROXY_LABEL,
    UNCERTAINTY_EXPLICIT,
    UNCERTAINTY_RAY_SPREAD_PROXY,
    HeightConstraint,
    ReferenceCandidate,
    ShadowScale,
    assess_reference_candidates,
    explicit_grid_dl_px,
    measure_building_shadows,
    scale_from_known_geometry,
    scale_from_references,
    shadow_height_constraints,
)
from depthwizard.logging_setup import get_logger
from depthwizard.shadows.detector import ClassicalShadowDetector, DetectionContext
from depthwizard.shadows.measure import BuildingFootprint, MeasurementParams
from depthwizard.validation.evaluation import NOT_MEASURED

__all__ = [
    "FitOutcome",
    "fit_constraints",
    "fit_oracle",
    "height_group",
    "process_dfc_tile",
    "summarise_records",
    "run_dfc",
    "run_synthetic",
    "main",
]

log = get_logger(__name__)

ORACLE_LABEL = (
    "ORACLE CONTROL (diagnostic upper bound, NOT deployable): a,b fitted directly to "
    "DFC2019 ground-truth AGL of the same constraint buildings. The gap to the shadow "
    "calibration is the error the shadow anchoring adds; the oracle's own error is the "
    "error of the Phase 3 relative field under a per-tile affine map."
)

CITIES = ("JAX", "OMA")
#: Tile height groups (max median AGL over the tile's kept footprints). Diagnostic only.
HEIGHT_GROUPS = (("0-5 m", 0.0, 5.0), ("5-10 m", 5.0, 10.0), ("10-20 m", 10.0, 20.0), (">=20 m", 20.0, math.inf))
SUN_CALIBRATED = ("calibrated", "fit_failed")


def height_group(max_height_m: float | None) -> str:
    if max_height_m is None or not math.isfinite(max_height_m):
        return "no buildings"
    for label, low, high in HEIGHT_GROUPS:
        if low <= max_height_m < high:
            return label
    return HEIGHT_GROUPS[0][0]  # negative AGL maxima: treat as the lowest group


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FitOutcome:
    ok: bool
    fit: AffineFit | None = None
    failure_reason: str | None = None
    n_constraints: int = 0
    n_usable: int = 0
    n_inliers: int | None = None
    used_ids: tuple[str, ...] = ()
    ransac: dict[str, Any] | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def identifiability(self) -> str:
        return classify_fit(self.fit) if self.ok and self.fit else classify_failure(self.failure_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "identifiability": self.identifiability,
            "unidentifiable": not self.ok,
            "failure_reason": self.failure_reason,
            "n_constraints": self.n_constraints,
            "n_usable": self.n_usable,
            "n_used_in_fit": len(self.used_ids),
            "n_rejected": self.n_constraints - self.n_usable,
            "rejection_reasons": self.rejection_reasons,
            "ransac": self.ransac,
            "fit": self.fit.to_dict() if self.fit else None,
        }


def fit_constraints(
    constraints: Sequence[HeightConstraint],
    r_bar: dict[str, float],
    *,
    min_constraints: int,
    max_condition: float,
    use_ransac: bool,
    ransac_threshold_sigma: float,
    ransac_iterations: int,
    ransac_seed: int,
) -> FitOutcome:
    """Fit ``a * r_bar + b = h`` over the usable constraints; failures are recorded."""
    usable = [c for c in constraints if c.usable]
    outcome = FitOutcome(
        ok=False,
        n_constraints=len(constraints),
        n_usable=len(usable),
        rejection_reasons=count_reasons(c.rejection_reason for c in constraints),
    )
    if len(usable) < min_constraints:
        outcome.failure_reason = (
            f"underdetermined: {len(usable)} usable constraint(s), need {min_constraints}"
        )
        return outcome
    x = np.array([r_bar[c.building_id] for c in usable])
    y = np.array([c.height_m for c in usable])
    w = np.array([c.weight for c in usable])
    try:
        if use_ransac:
            dh = np.array([c.dh_m for c in usable])
            result = ransac_affine(
                x, y,
                threshold=ransac_threshold_sigma * dh,
                weights=w,
                iterations=ransac_iterations,
                seed=ransac_seed,
                min_inliers=min_constraints,
                max_condition=max_condition,
            )
            outcome.fit = result.fit
            outcome.n_inliers = result.n_inliers
            outcome.used_ids = tuple(c.building_id for c, keep in zip(usable, result.inliers) if keep)
            outcome.ransac = {k: v for k, v in result.to_dict().items() if k in
                              ("n_inliers", "n_outliers", "iterations", "seed")}
        else:
            outcome.fit = fit_affine_wls(x, y, w, min_points=min_constraints, max_condition=max_condition)
            outcome.used_ids = tuple(c.building_id for c in usable)
    except CalibrationError as exc:
        outcome.failure_reason = str(exc)
        return outcome
    outcome.ok = True
    return outcome


def fit_oracle(
    building_ids: Sequence[str],
    r_bar: dict[str, float],
    h_true: dict[str, float | None],
    *,
    min_constraints: int,
    max_condition: float,
) -> FitOutcome:
    """ORACLE: unweighted LS of ``a * r_bar + b`` on the buildings' TRUE AGL."""
    ids = [i for i in building_ids if h_true.get(i) is not None]
    outcome = FitOutcome(ok=False, n_constraints=len(building_ids), n_usable=len(ids))
    if len(ids) < min_constraints:
        outcome.failure_reason = f"underdetermined: {len(ids)} building(s) with ground truth"
        return outcome
    try:
        outcome.fit = fit_affine_wls(
            [r_bar[i] for i in ids], [h_true[i] for i in ids],
            min_points=min_constraints, max_condition=max_condition,
        )
    except CalibrationError as exc:
        outcome.failure_reason = str(exc)
        return outcome
    outcome.used_ids = tuple(ids)
    outcome.ok = True
    return outcome


def _fit_kwargs(cfg: Phase4Config, min_constraints: int | None = None) -> dict[str, Any]:
    return dict(
        min_constraints=cfg.fit.min_constraints if min_constraints is None else min_constraints,
        max_condition=cfg.fit.max_condition,
        use_ransac=cfg.fit.ransac,
        ransac_threshold_sigma=cfg.fit.ransac_threshold_sigma,
        ransac_iterations=cfg.fit.ransac_iterations,
        ransac_seed=cfg.fit.ransac_seed,
    )


def _measurement_params(cfg: Phase4Config) -> MeasurementParams:
    m = cfg.measurement
    return MeasurementParams(
        step_px=m.step_px,
        gap_tolerance_px=m.gap_tolerance_px,
        max_rays=m.max_rays,
        min_rays=m.min_rays,
        min_hit_fraction=m.min_hit_fraction,
        nominal_hit_fraction=m.nominal_hit_fraction,
        percentile=m.percentile,
    )


def _footprint_truth(
    footprints: Sequence[BuildingFootprint], agl: np.ndarray, valid: np.ndarray, min_valid_fraction: float
) -> dict[str, float | None]:
    truth: dict[str, float | None] = {}
    for fp in footprints:
        sel = fp.mask & valid
        truth[fp.building_id] = (
            float(np.median(agl[sel])) if sel.sum() >= min_valid_fraction * fp.pixel_count else None
        )
    return truth


# ---------------------------------------------------------------------------
# One real DFC2019 tile
# ---------------------------------------------------------------------------


def process_dfc_tile(
    tile_id: str,
    *,
    city: str,
    z_rel: np.ndarray,
    log_space: bool,
    agl: np.ndarray,
    agl_valid: np.ndarray,
    cls: np.ndarray,
    rgb_u8: np.ndarray,
    cfg: Phase4Config,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate and score one tile. Returns its self-contained record.

    ``debug``, when a dict is passed, is filled with the intermediate arrays
    the diagnostic figures need.
    """
    record: dict[str, Any] = {"tile_id": tile_id, "city": city}
    data_cfg, ref_cfg = cfg.dataset, cfg.reference
    agl0 = np.where(agl_valid, np.nan_to_num(agl), 0.0)

    extraction = extract_building_footprints(
        cls,
        tile_id=tile_id,
        building_class=data_cfg.building_class,
        min_area_px=cfg.footprints.min_area_px,
        connectivity=cfg.footprints.connectivity,
        reject_edge_touching=cfg.footprints.reject_edge_touching,
    )
    record["footprints"] = extraction.summary()
    h_true = _footprint_truth(extraction.footprints, agl0, agl_valid, ref_cfg.min_valid_fraction)
    heights = [h for h in h_true.values() if h is not None]
    # Diagnostic grouping only -- never an input to the method.
    record["footprint_heights_m"] = heights
    record["tile_max_building_agl_m"] = max(heights) if heights else None
    record["height_group"] = height_group(record["tile_max_building_agl_m"])

    detected = ClassicalShadowDetector().detect(rgb_u8, DetectionContext(band_order="rgb"))
    surface = np.isin(cls, np.asarray(data_cfg.shadow_surface_classes))
    shadow = detected.mask & surface
    record["shadow_mask"] = {
        "detector": detected.method,
        "threshold": detected.threshold,
        "detector_shadow_fraction": detected.shadow_fraction,
        "surface_restricted_shadow_fraction": float(shadow.mean()),
    }
    if debug is not None:
        debug.update(rgb=rgb_u8, building_mask=extraction.building_mask, shadow=shadow,
                     masks={fp.building_id: fp.mask for fp in extraction.footprints})

    def finish(status: str, reason: str | None = None) -> dict[str, Any]:
        record["status"] = status
        record["status_reason"] = reason
        return record

    azimuth = estimate_sun_azimuth(
        shadow,
        extraction.building_mask,
        band_px=cfg.azimuth.band_px,
        step_deg=cfg.azimuth.step_deg,
        min_building_px=cfg.azimuth.min_building_px,
        min_peak_to_median=cfg.azimuth.min_peak_to_median,
    )
    record["azimuth"] = azimuth.to_dict()
    if not azimuth.ok:
        kind = "no_buildings" if not extraction.footprints else "azimuth_unavailable"
        return finish(kind, azimuth.failure_reason)
    if debug is not None:
        debug["sun_azimuth_deg"] = azimuth.sun_azimuth_deg
    if not extraction.footprints:
        return finish("no_buildings", "no footprint survived filtering")

    shadows = measure_building_shadows(
        extraction.footprints,
        shadow,
        sun_azimuth_deg=float(azimuth.sun_azimuth_deg),
        params=_measurement_params(cfg),
        max_edge_terminated_fraction=cfg.measurement.max_edge_terminated_fraction,
        max_merged_fraction=cfg.measurement.max_merged_fraction,
        occluder_mask=extraction.building_mask,
    )
    record["shadows"] = {
        "measured": len(shadows),
        "usable": sum(s.usable for s in shadows),
        "nominal_usable": sum(s.usable and s.confidence.value == "nominal" for s in shadows),
        "rejection_reasons": count_reasons(s.rejection_reason for s in shadows),
    }
    if debug is not None:
        debug["shadows"] = shadows

    r = linearise_relative(z_rel, log_space=log_space)
    r_bar = dict(zip(
        (fp.building_id for fp in extraction.footprints),
        footprint_means(r, [fp.mask for fp in extraction.footprints]),
    ))

    candidates = assess_reference_candidates(
        shadows,
        agl=agl0,
        agl_valid=agl_valid,
        cls=cls,
        building_mask=extraction.building_mask,
        sun_azimuth_deg=float(azimuth.sun_azimuth_deg),
        surface_classes=data_cfg.shadow_surface_classes,
        min_area_px=ref_cfg.min_area_px,
        isolation_px=ref_cfg.isolation_px,
        min_solidity=ref_cfg.min_solidity,
        min_height_m=ref_cfg.min_height_m,
        min_valid_fraction=ref_cfg.min_valid_fraction,
        min_shadow_zone_ground_fraction=ref_cfg.min_shadow_zone_ground_fraction,
        max_relative_spread=ref_cfg.max_relative_spread,
    )
    eligible = [c for c in candidates if c.eligible]
    record["reference_candidates"] = {
        "eligible": len(eligible),
        "rejected": len(candidates) - len(eligible),
        "rejection_reasons": count_reasons(c.rejection_reason for c in candidates),
        "eligible_heights_m": [c.h_ref_m for c in eligible],
    }
    if debug is not None:
        debug["rejected_candidates"] = [c.building_id for c in candidates if not c.eligible]
    if len(eligible) < ref_cfg.min_count:
        return finish(
            "calibration_insufficient",
            f"{len(eligible)} valid reference building(s), need {ref_cfg.min_count}",
        )

    masks = {fp.building_id: fp.mask for fp in extraction.footprints}
    building_px = extraction.building_mask

    def ref_union(refs: Sequence[ReferenceCandidate]) -> np.ndarray:
        union = np.zeros_like(building_px)
        for c in refs:
            union |= masks[c.building_id]
        return union

    def calibrate(refs: Sequence[ReferenceCandidate], uncertainty: str) -> tuple[ShadowScale, list[HeightConstraint], FitOutcome]:
        scale = scale_from_references(refs, gsd_m=data_cfg.gsd_m, estimator=ref_cfg.k_estimator)
        constraints = shadow_height_constraints(
            shadows,
            scale,
            exclude_ids=[c.building_id for c in refs],
            uncertainty=uncertainty,
            explicit_dl_px=None,  # DFC2019 supplies no explicit shadow-length uncertainty
            reduced_weight_factor=cfg.fit.reduced_weight_factor,
        )
        return scale, constraints, fit_constraints(constraints, r_bar, **_fit_kwargs(cfg))

    def dense(fit: AffineFit, eval_mask: np.ndarray) -> dict[str, Any]:
        return dense_error_sums(apply_affine(r, fit.a, fit.b), agl0, eval_mask, building_px)

    # ---- primary calibration: 3 references, or 2 when only 2 exist ----------
    primary_refs = eligible[: min(ref_cfg.primary_count, len(eligible))]
    scale, constraints, outcome = calibrate(primary_refs, UNCERTAINTY_RAY_SPREAD_PROXY)
    record["sun_calibration"] = scale.to_dict()
    record["sun_calibration"]["references"] = [
        {**c.to_dict(), "r_bar": r_bar[c.building_id]} for c in primary_refs
    ]
    if debug is not None:
        debug["references"] = [c.building_id for c in primary_refs]

    usable = [c for c in constraints if c.usable]
    record["constraints"] = {
        "evaluation_buildings": len(constraints),
        "usable": len(usable),
        "rejection_reasons": count_reasons(c.rejection_reason for c in constraints),
        "r_bar_usable": distribution([r_bar[c.building_id] for c in usable]),
    }
    # Per-building: shadow height vs ground truth. No a,b involved.
    record["per_building"] = [
        [c.building_id, float(c.height_m), float(h_true[c.building_id])]
        for c in usable if h_true[c.building_id] is not None
    ]
    record["per_building_missing_truth"] = sum(1 for c in usable if h_true[c.building_id] is None)

    _, _, outcome_a = calibrate(primary_refs, UNCERTAINTY_EXPLICIT)
    record["fit_explicit_dh"] = outcome_a.to_dict()
    record["fit"] = outcome.to_dict()
    eval_mask = agl_valid & ~ref_union(primary_refs)
    record["dense"] = {}
    if not outcome.ok:
        log.warning("tile a,b fit is UNIDENTIFIABLE", extra={"tile": tile_id, "reason": outcome.failure_reason})
        return finish("fit_failed", outcome.failure_reason)
    record["dense"]["shadow"] = dense(outcome.fit, eval_mask)

    plain = fit_constraints(constraints, r_bar, **{**_fit_kwargs(cfg), "use_ransac": False})
    record["fit_wls_no_ransac"] = plain.to_dict()
    if plain.ok:
        record["dense"]["shadow_wls_no_ransac"] = dense(plain.fit, eval_mask)

    oracle = fit_oracle(
        [c.building_id for c in usable], r_bar, h_true,
        min_constraints=cfg.fit.min_constraints, max_condition=cfg.fit.max_condition,
    )
    record["oracle"] = oracle.to_dict()
    if oracle.ok:
        record["dense"]["oracle"] = dense(oracle.fit, eval_mask)

    # ---- reference-count sensitivity ----------------------------------------
    sensitivity: dict[str, Any] = {}
    fits_by_k: dict[int, AffineFit] = {}
    for k in ref_cfg.sensitivity_counts:
        if len(eligible) < k:
            sensitivity[str(k)] = {"status": "not enough valid reference buildings"}
            continue
        refs_k = eligible[:k]
        scale_k, _, outcome_k = calibrate(refs_k, UNCERTAINTY_RAY_SPREAD_PROXY)
        entry = {
            "status": "fitted" if outcome_k.ok else "fit_failed",
            "reference_ids": [c.building_id for c in refs_k],
            "metres_per_shadow_px": scale_k.metres_per_shadow_px,
            "fit": outcome_k.to_dict(),
        }
        if outcome_k.ok:
            fits_by_k[k] = outcome_k.fit
            entry["dense"] = dense(outcome_k.fit, agl_valid & ~ref_union(refs_k))
        sensitivity[str(k)] = entry
    largest = max(ref_cfg.sensitivity_counts)
    if all(k in fits_by_k for k in ref_cfg.sensitivity_counts):
        common = agl_valid & ~ref_union(eligible[:largest])
        for k, fit in fits_by_k.items():
            sensitivity[str(k)]["dense_common"] = dense(fit, common)
    record["reference_sensitivity"] = sensitivity
    return finish("calibrated")


# ---------------------------------------------------------------------------
# Summaries (from tile records only)
# ---------------------------------------------------------------------------


def _pool_dense(records: Iterable[dict[str, Any]], getter) -> dict[str, Any]:
    parts = {"building": ErrorAccumulator(), "ground": ErrorAccumulator(), "all": ErrorAccumulator()}
    for rec in records:
        sums = getter(rec)
        if not sums:
            continue
        for part, acc in parts.items():
            acc.add_sums(sums.get(part), rec["tile_id"])
    return {part: acc.to_dict() for part, acc in parts.items()}


def _gap(shadow: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    gap: dict[str, Any] = {}
    for part in ("building", "ground", "all"):
        s, o = shadow[part], oracle[part]
        if isinstance(s["mae_m"], float) and isinstance(o["mae_m"], float):
            gap[part] = {"mae_gap_m": s["mae_m"] - o["mae_m"], "rmse_gap_m": s["rmse_m"] - o["rmse_m"]}
        else:
            gap[part] = NOT_MEASURED
    return gap


def _fit_stats(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    fits = [r[key] for r in records if r.get(key)]
    ok = [f["fit"] for f in fits if f["ok"]]
    classes = {c: 0 for c in IDENTIFIABILITY_CLASSES}
    for f in fits:
        classes[f["identifiability"]] += 1
    return {
        "fits_attempted": len(fits),
        "fits_returned": len(ok),
        "identifiability_classes": classes,
        "identifiable_fraction_of_attempted": (classes["identifiable_positive"] / len(fits)) if fits else NOT_MEASURED,
        "a": distribution([f["a"] for f in ok]),
        "b_m": distribution([f["b"] for f in ok]),
        "a_se": distribution([f["a_se"] for f in ok]),
        "a_over_se": distribution([f["a_over_se"] for f in ok]),
        "weighted_r_bar_variance": distribution([f["weighted_r_bar_variance"] for f in ok]),
        "condition": distribution([f["condition"] for f in ok]),
    }


def _group_summary(records: Sequence[dict[str, Any]], cfg: Phase4Config) -> dict[str, Any]:
    status: dict[str, int] = {}
    for r in records:
        status[r["status"]] = status.get(r["status"], 0) + 1
    reached_selection = [r for r in records if "reference_candidates" in r]
    sun_cal = [r for r in records if r["status"] in SUN_CALIBRATED]
    calibrated = [r for r in records if r["status"] == "calibrated"]

    def merged(key_path: Sequence[str], rows: Sequence[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            d: Any = r
            for k in key_path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            for k, n in (d or {}).items():
                out[k] = out.get(k, 0) + n
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    total_footprints = sum(r["footprints"]["components_surviving"] for r in records)
    eval_buildings = sum(r["constraints"]["evaluation_buildings"] for r in sun_cal)
    usable = sum(r["constraints"]["usable"] for r in sun_cal)
    pairs = [p for r in sun_cal for p in r.get("per_building", [])]
    pred = np.array([p[1] for p in pairs]); true = np.array([p[2] for p in pairs])
    per_building = ErrorAccumulator(); per_building.add(pred, true)
    pairs_cal = [p for r in calibrated for p in r.get("per_building", [])]
    pb_cal = ErrorAccumulator(); pb_cal.add([p[1] for p in pairs_cal], [p[2] for p in pairs_cal])

    shadow = _pool_dense(calibrated, lambda r: r["dense"].get("shadow"))
    paired = [r for r in calibrated if "oracle" in r["dense"]]
    shadow_paired = _pool_dense(paired, lambda r: r["dense"].get("shadow"))
    oracle = _pool_dense(paired, lambda r: r["dense"].get("oracle"))
    by_class = {
        c: _pool_dense([r for r in calibrated if r["fit"]["identifiability"] == c], lambda r: r["dense"].get("shadow"))
        for c in ("identifiable_positive", "statistically_weak", "non_positive")
    }
    sens: dict[str, Any] = {}
    for k in cfg.reference.sensitivity_counts:
        key = str(k)
        sens[key] = {
            "tiles_fitted": sum(1 for r in calibrated if r["reference_sensitivity"].get(key, {}).get("status") == "fitted"),
            "tiles_without_enough_references": sum(
                1 for r in calibrated
                if r["reference_sensitivity"].get(key, {}).get("status") == "not enough valid reference buildings"),
            "tiles_fit_failed": sum(1 for r in calibrated if r["reference_sensitivity"].get(key, {}).get("status") == "fit_failed"),
            "own_tiles": _pool_dense(calibrated, lambda r, key=key: r["reference_sensitivity"].get(key, {}).get("dense")),
            "common_tiles": _pool_dense(calibrated, lambda r, key=key: r["reference_sensitivity"].get(key, {}).get("dense_common")),
        }

    return {
        "tiles": {
            "total": len(records),
            "images": len({r["image"] for r in records}),
            "geographic_tiles": len({r["scene_id"] for r in records}),
            "status": status,
            "funnel": {
                "total": len(records),
                "with_buildings": sum(1 for r in records if r["footprints"]["components_surviving"] > 0),
                "azimuth_recovered": sum(1 for r in records if r.get("azimuth", {}).get("ok")),
                "reached_reference_selection": len(reached_selection),
                "with_ge2_valid_references": sum(1 for r in reached_selection if r["reference_candidates"]["eligible"] >= cfg.reference.min_count),
                "sun_calibrated": len(sun_cal),
                "fit_returned": len(calibrated),
                "fit_identifiable_positive": sum(1 for r in calibrated if r["fit"]["identifiability"] == "identifiable_positive"),
            },
            "calibrated_images": len({r["image"] for r in calibrated}),
            "calibrated_geographic_tiles": sorted({r["scene_id"] for r in calibrated}),
        },
        "components": {
            "found": sum(r["footprints"]["components_found"] for r in records),
            "surviving": total_footprints,
            "rejection_reasons": merged(("footprints", "rejection_reasons"), records),
        },
        "shadow_measurement_rejections": merged(("shadows", "rejection_reasons"), records),
        "shadow_measurements": {
            "measured": sum(r.get("shadows", {}).get("measured", 0) for r in records),
            "usable": sum(r.get("shadows", {}).get("usable", 0) for r in records),
        },
        "reference_candidate_rejections": merged(("reference_candidates", "rejection_reasons"), reached_selection),
        "eligible_reference_heights_m": height_distribution(
            [h for r in reached_selection for h in r["reference_candidates"]["eligible_heights_m"]]),
        "calibration_buildings": {
            "count": sum(len(r["sun_calibration"]["references"]) for r in sun_cal),
            "agl_m": height_distribution([c["h_ref_m"] for r in sun_cal for c in r["sun_calibration"]["references"]]),
            "r_bar": distribution([c["r_bar"] for r in sun_cal for c in r["sun_calibration"]["references"]]),
            "metres_per_shadow_px": distribution([r["sun_calibration"]["metres_per_shadow_px"] for r in sun_cal]),
        },
        "evaluation_buildings": {
            "total_kept_footprints_in_group": total_footprints,
            "eligible_on_sun_calibrated_tiles": eval_buildings,
            "usable": usable,
            "rejected": eval_buildings - usable,
            "rejection_reasons": merged(("constraints", "rejection_reasons"), sun_cal),
            "used_in_fit_after_ransac": sum(r["fit"]["n_used_in_fit"] for r in calibrated),
        },
        "per_building_shadow_vs_agl": {
            "label": "shadow-measured building evaluation on successfully sun-calibrated held-out tiles",
            "evaluated_eligible_total": f"{per_building.count} / {eval_buildings} / {total_footprints}",
            **per_building.to_dict(),
            "on_fitted_tiles_only": pb_cal.to_dict(),
            "corr_true_height_vs_error": float(np.corrcoef(true, pred - true)[0, 1]) if true.size > 2 else NOT_MEASURED,
            "spearman_shadow_vs_true": _spearman(pred, true),
            "by_true_height": binned_errors(pred, true),
        },
        "fit_shadow": _fit_stats(calibrated + [r for r in sun_cal if r["status"] == "fit_failed"], "fit"),
        "fit_oracle": _fit_stats(calibrated, "oracle"),
        "dense_shadow_all_returned_fits": shadow,
        "dense_shadow_by_identifiability": by_class,
        "dense_shadow_wls_no_ransac": _pool_dense(calibrated, lambda r: r["dense"].get("shadow_wls_no_ransac")),
        "oracle_control": {
            "label": ORACLE_LABEL,
            "tiles": len(paired),
            "shadow_same_tiles": shadow_paired,
            "oracle": oracle,
            "gap_shadow_minus_oracle": _gap(shadow_paired, oracle),
        },
        "reference_sensitivity": sens,
        "explicit_dh_form": {
            "fits_attempted": sum(1 for r in sun_cal if r.get("fit_explicit_dh")),
            "fits_returned": sum(1 for r in sun_cal if r.get("fit_explicit_dh", {}).get("ok")),
            "note": "DFC2019 supplies no explicit shadow-length uncertainty; every constraint is dropped",
        },
        "by_tile_height_group": _height_group_table(records),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> Any:
    if a.size < 3:
        return NOT_MEASURED
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _height_group_table(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tall-building identifiability test (tile height group = diagnostic label only)."""
    table: dict[str, Any] = {}
    for label, *_ in HEIGHT_GROUPS:
        rows = [r for r in records if r.get("height_group") == label]
        cal = [r for r in rows if r["status"] == "calibrated"]
        status: dict[str, int] = {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
        table[label] = {
            "tiles": len(rows),
            "status": status,
            "reference_count": distribution([len(r["sun_calibration"]["references"]) for r in rows if "sun_calibration" in r]),
            "shadow_fit": _fit_stats(cal + [r for r in rows if r["status"] == "fit_failed"], "fit"),
            "oracle_fit": _fit_stats(cal, "oracle"),
        }
    return table


def summarise_records(records: Sequence[dict[str, Any]], cfg: Phase4Config) -> dict[str, Any]:
    groups = {city: [r for r in records if r["city"] == city] for city in CITIES}
    groups["combined"] = list(records)
    return {name: _group_summary(rows, cfg) for name, rows in groups.items()}


# ---------------------------------------------------------------------------
# The DFC2019 run
# ---------------------------------------------------------------------------


def _crop_offsets(extent: int, size: int) -> list[int]:
    if extent <= size:
        return [0]
    offsets = list(range(0, extent - size + 1, size))
    if offsets[-1] + size < extent:
        offsets.append(extent - size)
    return offsets


def select_views(pairs: Sequence[Any], views_per_scene: int | None) -> list[Any]:
    """Every pair, or ``views_per_scene`` per geographic tile, evenly spaced in sorted order."""
    pairs = sorted(pairs, key=lambda p: p.stem)
    if views_per_scene is None:
        return list(pairs)
    by_scene: dict[str, list[Any]] = {}
    for p in pairs:
        by_scene.setdefault(p.scene_id, []).append(p)
    chosen: list[Any] = []
    for scene in sorted(by_scene):
        views = by_scene[scene]
        idx = np.unique(np.linspace(0, len(views) - 1, min(views_per_scene, len(views))).round().astype(int))
        chosen += [views[i] for i in idx]
    return chosen


def _representative_tiles(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Pick figure tiles from structure only (never from accuracy)."""
    picks: dict[str, str] = {}
    for city in CITIES:
        rows = sorted((r for r in records if r["city"] == city and r.get("azimuth", {}).get("ok")),
                      key=lambda r: r["tile_id"])
        if not rows:
            continue
        n = lambda r: r["footprints"]["components_surviving"]  # noqa: E731
        picks[f"{city}_dense_urban"] = max(rows, key=n)["tile_id"]
        sparse = [r for r in rows if n(r) >= 3]
        if sparse:
            picks[f"{city}_sparse_urban"] = min(sparse, key=n)["tile_id"]
        low = [r for r in rows if r["height_group"] == "0-5 m"]
        if low:
            picks[f"{city}_low_rise"] = max(low, key=n)["tile_id"]
        tall = [r for r in rows if r["tile_max_building_agl_m"] is not None]
        if tall:
            picks[f"{city}_tallest"] = max(tall, key=lambda r: r["tile_max_building_agl_m"])["tile_id"]
    return picks


def run_dfc(
    cfg: Phase4Config,
    *,
    relative_config: Path | None = None,
    device: str | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    """Process the configured DFC2019 split and write the Phase 4a report."""
    from depthwizard.relative.config import load_relative_config
    from depthwizard.relative.data import discover_pairs, open_raster, raster_shape, split_pairs
    from depthwizard.relative.inference import infer_tile, load_model_from_checkpoint

    rel_cfg = load_relative_config(relative_config or cfg.relative_config)
    pairs = discover_pairs(rel_cfg.dataset)
    train, val = split_pairs(pairs, rel_cfg.dataset.split)
    if not rel_cfg.dataset.split.is_spatially_separated:
        raise CalibrationError("the Phase 3 split is not spatially separated; refusing to evaluate on it")
    held_out = cfg.dataset.split_side == "val"
    selected = select_views(val if held_out else train, cfg.dataset.views_per_scene)
    if not held_out:
        log.warning("Phase 4a is running on the Phase 3 TRAINING split: diagnostics only, not held out")
    if cfg.dataset.max_pairs is not None:
        selected = selected[: cfg.dataset.max_pairs]
    if held_out:
        leaked = sorted({p.scene_id for p in selected} & {p.scene_id for p in train})
        if leaked:
            raise CalibrationError(f"held-out scenes also appear in training: {leaked[:5]}")

    model, model_cfg = load_model_from_checkpoint(cfg.checkpoint)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.dataset.crop_size
    started = time.time()
    records: list[dict[str, Any]] = []
    by_stem = {p.stem: p for p in selected}

    def load_tile(pair: Any, row: int, col: int, cls_full: np.ndarray) -> dict[str, Any]:
        prediction = infer_tile(
            model, model_cfg, pair.image_path, height_path=pair.height_path,
            row_off=row, col_off=col, tile_size=size, device=device,
        )
        if prediction.is_fallback or prediction.fallback_kind:
            raise CalibrationError("Phase 4a requires the trained Phase 3 model, not a fallback")
        return dict(
            z_rel=prediction.relative_height,
            log_space=prediction.log_space,
            agl=np.nan_to_num(prediction.ground_truth),
            agl_valid=prediction.valid,
            cls=cls_full[row: row + size, col: col + size],
            rgb_u8=np.clip(np.rint(prediction.image_rgb * 255.0), 0, 255).astype(np.uint8),
        )

    def read_cls(pair: Any) -> np.ndarray:
        cls_path = pair.height_path.with_name(
            pair.height_path.name[: -len(rel_cfg.dataset.height_suffix)] + cfg.dataset.cls_suffix
        )
        if not cls_path.is_file():
            raise FileNotFoundError(f"CLS raster missing for {pair.stem}: {cls_path}")
        with open_raster(cls_path) as src:
            return src.read(1)

    with (out_dir / "per_tile.jsonl").open("w", encoding="utf-8") as per_tile:
        for index, pair in enumerate(selected):
            height, width = raster_shape(pair.image_path)
            cls_full = read_cls(pair)
            if cls_full.shape != (height, width):
                raise CalibrationError(f"{pair.stem}: CLS {cls_full.shape} != RGB {(height, width)}")
            for row in _crop_offsets(height, size):
                for col in _crop_offsets(width, size):
                    tile_id = f"{pair.stem}_r{row}_c{col}"
                    try:
                        record = process_dfc_tile(tile_id, city=pair.city, cfg=cfg, **load_tile(pair, row, col, cls_full))
                    except CalibrationError as exc:
                        log.error("tile calibration error", extra={"tile": tile_id, "error": str(exc)})
                        record = {"tile_id": tile_id, "city": pair.city, "status": "calibration_error",
                                  "status_reason": str(exc), "footprints": {"components_found": 0,
                                  "components_surviving": 0, "rejection_reasons": {}},
                                  "height_group": "no buildings", "tile_max_building_agl_m": None}
                    record.update({"image": pair.stem, "scene_id": pair.scene_id, "row_off": row, "col_off": col})
                    records.append(record)
                    per_tile.write(json.dumps(to_jsonable(record)) + "\n")
            if (index + 1) % 25 == 0 or index + 1 == len(selected):
                log.info("phase 4a progress", extra={"pairs_done": index + 1, "pairs": len(selected),
                                                     "tiles": len(records), "elapsed_s": round(time.time() - started, 1)})

    figure_paths: dict[str, str] = {}
    if figures:
        for name, tile_id in _representative_tiles(records).items():
            stem, rest = tile_id.rsplit("_r", 1)
            row, col = (int(v) for v in rest.split("_c"))
            pair = by_stem[stem]
            debug: dict[str, Any] = {}
            process_dfc_tile(tile_id, city=pair.city, cfg=cfg, debug=debug, **load_tile(pair, row, col, read_cls(pair)))
            figure_paths[name] = str(_plot_debug(out_dir / "figures" / f"{name}__{tile_id}.png", name, tile_id, debug))

    report = {
        "phase": "4a (v2)",
        "data": ("REAL DFC2019 Track 1 - held-out Phase 3 validation split" if held_out
                 else "REAL DFC2019 Track 1 - Phase 3 TRAINING split (DIAGNOSTICS ONLY, not held out; Phase 3 was trained on these tiles)"),
        "elapsed_s": time.time() - started,
        "config": cfg.to_dict(),
        "facts": {
            "building_class": cfg.dataset.building_class,
            "building_class_source": DFC2019_BUILDING_CLASS_SOURCE,
            "gsd_m": cfg.dataset.gsd_m,
            "gsd_note": cfg.dataset.gsd_note,
            "identifiable_threshold_a_over_se": IDENTIFIABLE_T,
            "uncertainty": PROXY_LABEL,
            "sun_elevation": "not determinable: DFC2019 tile GSD unverified" if cfg.dataset.gsd_m is None else "see per_tile",
            "sun_azimuth_accuracy": "not measurable on DFC2019 (no sun metadata)",
        },
        "split": {
            "mode": rel_cfg.dataset.split.mode,
            "seed": rel_cfg.dataset.split.seed,
            "side": cfg.dataset.split_side,
            "views_per_scene": cfg.dataset.views_per_scene,
            "pairs": len(selected),
        },
        "groups": summarise_records(records, cfg),
        "figures": figure_paths,
    }
    write_json(out_dir / "phase4a_report.json", report)
    return report


def _plot_debug(path: Path, name: str, tile_id: str, debug: dict[str, Any]) -> Path:
    masks = debug.get("masks", {})
    refs = debug.get("references", [])
    rejected = [i for i in debug.get("rejected_candidates", []) if i not in refs]
    segments = []
    for shadow in debug.get("shadows", []):
        d_row, d_col = shadow.measurement.direction_px
        for ray in shadow.measurement.rays:
            if ray.hit and ray.length_px:
                segments.append((ray.start_row, ray.start_col,
                                 ray.start_row + d_row * ray.length_px, ray.start_col + d_col * ray.length_px))
    return plot_tile_diagnostic(
        path,
        title=f"{name}: {tile_id}  (diagnostic figure; selected by structure, not accuracy)",
        rgb=debug["rgb"],
        building_mask=debug["building_mask"],
        shadow_mask=debug["shadow"],
        reference_masks=[masks[i] for i in refs if i in masks],
        rejected_masks=[masks[i] for i in rejected if i in masks],
        sun_azimuth_deg=debug.get("sun_azimuth_deg"),
        ray_segments=segments,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC control
# ---------------------------------------------------------------------------


def _synthetic_scene_run(
    scene: Any,
    cfg: Phase4Config,
    *,
    use_recovered_azimuth: bool,
) -> dict[str, Any]:
    from depthwizard.ingest.synthetic import render_scene

    render = render_scene(scene)
    image = np.clip(np.rint(render.reflectance * 255.0), 0, 255).astype(np.uint8)
    shape = image.shape
    footprints = [
        BuildingFootprint.from_bbox(
            name, shape=shape, row_min=fp.row_min, row_max=fp.row_max, col_min=fp.col_min, col_max=fp.col_max
        )
        for name, fp in render.footprints.items()
    ]
    truth_h = {b.name: float(b.height_m) for b in scene.buildings}
    gsd = scene.raster.gsd_m
    true_el, true_az = scene.sun.elevation_deg, scene.sun.azimuth_deg

    detected = ClassicalShadowDetector().detect(image, DetectionContext(gsd_m=gsd))
    azimuth = estimate_sun_azimuth(
        detected.mask, render.building_mask,
        band_px=cfg.azimuth.band_px, step_deg=cfg.azimuth.step_deg,
        min_building_px=cfg.azimuth.min_building_px, min_peak_to_median=cfg.azimuth.min_peak_to_median,
    )
    result: dict[str, Any] = {
        "label": "SYNTHETIC",
        "scene": scene.name,
        "true_sun_elevation_deg": true_el,
        "true_sun_azimuth_deg": true_az,
        "gsd_m": gsd,
        "detector_mask_differing_pixels": int((detected.mask != render.shadow_mask).sum()),
        "azimuth": {
            **azimuth.to_dict(),
            "true_sun_azimuth_deg": true_az,
            "abs_error_deg": angular_difference_deg(azimuth.sun_azimuth_deg, true_az) if azimuth.ok else NOT_MEASURED,
            "quantisation_bound_deg": angular_resolution_deg(cfg.azimuth.band_px),
        },
    }
    measure_az = azimuth.sun_azimuth_deg if (use_recovered_azimuth and azimuth.ok) else true_az
    result["measurement_azimuth_source"] = "recovered" if use_recovered_azimuth else "true metadata"

    shadows = measure_building_shadows(
        footprints, detected.mask,
        sun_azimuth_deg=measure_az, params=_measurement_params(cfg),
        max_edge_terminated_fraction=cfg.measurement.max_edge_terminated_fraction,
        max_merged_fraction=cfg.measurement.max_merged_fraction,
        gsd_m=gsd, sun_elevation_deg=true_el,
    )
    scale = scale_from_known_geometry(gsd_m=gsd, sun_elevation_deg=true_el)
    result["sun_calibration"] = {**scale.to_dict(), "calibrated_sun_elevation_deg": scale.sun_elevation_deg}

    # Constructed relative field with a KNOWN answer: z = log(AGL + 1) + c.
    c = cfg.synthetic.log_offset
    z = np.log(render.height_m.astype(np.float64) + 1.0) + c
    r = linearise_relative(z, log_space=True)
    true_a, true_b = math.exp(-c), -1.0
    r_bar = dict(zip((fp.building_id for fp in footprints), footprint_means(r, [fp.mask for fp in footprints])))
    result["constructed_field"] = {
        "z_rel": "log(AGL_true + 1) + c",
        "c": c,
        "true_a": true_a,
        "true_b": true_b,
        "note": "SYNTHETIC relative field built from the fixture truth, not a Phase 3 prediction",
    }

    forms: dict[str, Any] = {}
    everything = np.ones_like(render.building_mask)
    for form, dl in ((UNCERTAINTY_EXPLICIT, explicit_grid_dl_px(measure_az)), (UNCERTAINTY_RAY_SPREAD_PROXY, None)):
        constraints = shadow_height_constraints(
            shadows, scale, uncertainty=form, explicit_dl_px=dl,
            reduced_weight_factor=cfg.fit.reduced_weight_factor,
        )
        outcome = fit_constraints(constraints, r_bar, **_fit_kwargs(cfg, cfg.synthetic.min_constraints))
        entry: dict[str, Any] = {
            "explicit_dl_px": dl,
            "explicit_dl_source": (
                "half the pixel-centre pitch along the shadow direction (grid quantisation of the "
                "fixture's construction)" if dl is not None else None
            ),
            "constraints": [c_.to_dict() for c_ in constraints],
            "fit": outcome.to_dict(),
            "per_building": [
                {
                    "building_id": c_.building_id,
                    "shadow_height_m": c_.height_m,
                    "true_height_m": truth_h[c_.building_id],
                    "abs_error_m": abs(c_.height_m - truth_h[c_.building_id]) if c_.height_m is not None else None,
                }
                for c_ in constraints
            ],
        }
        if outcome.ok:
            fit = outcome.fit
            sums = dense_error_sums(apply_affine(r, fit.a, fit.b), render.height_m, everything, render.building_mask)
            dense = {}
            for part, s in sums.items():
                acc = ErrorAccumulator(); acc.add_sums(s); dense[part] = acc.to_dict()
            entry.update(recovered_a=fit.a, recovered_b=fit.b, a_error=fit.a - true_a,
                         b_error_m=fit.b - true_b, dense=dense)
        else:
            entry["dense"] = NOT_MEASURED
        forms[form] = entry
    result["uncertainty_forms"] = forms
    return result


def run_synthetic(cfg: Phase4Config) -> dict[str, Any]:
    """The SYNTHETIC control: true sun elevation, known a, b."""
    from depthwizard.config import load_config

    scene = load_config(cfg.synthetic.scene_config).scene
    report: dict[str, Any] = {
        "label": "SYNTHETIC CONTROL - specified, not measured, heights; says nothing about real-world accuracy",
        "primary_true_metadata": _synthetic_scene_run(scene, cfg, use_recovered_azimuth=False),
        "recovered_azimuth_variant": _synthetic_scene_run(scene, cfg, use_recovered_azimuth=True),
        "azimuth_sweep": [],
    }
    for az in (45.0, 90.0, 135.0, 200.0, 290.0):
        swept = replace(scene, name=f"{scene.name}_az{az:g}", sun=replace(scene.sun, azimuth_deg=az))
        run = _synthetic_scene_run(swept, cfg, use_recovered_azimuth=True)
        report["azimuth_sweep"].append({
            "true_sun_azimuth_deg": az,
            "recovered_sun_azimuth_deg": run["azimuth"]["sun_azimuth_deg"],
            "abs_error_deg": run["azimuth"]["abs_error_deg"],
        })
    write_json(cfg.output_dir / "phase4a_synthetic_report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    from depthwizard.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="DepthWizard Phase 4a: pixel-space metric calibration")
    parser.add_argument("target", choices=("dfc", "synthetic", "ablation"))
    parser.add_argument("--config", default="configs/phase4.yaml")
    parser.add_argument("--relative-config", default=None,
                        help="Phase 3 config with the local DFC2019 root (e.g. configs/phase3.local.yaml)")
    parser.add_argument("--split-side", choices=("val", "train"), default=None)
    parser.add_argument("--views-per-scene", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-format", choices=("text", "json"), default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging({"level": "INFO", "format": args.log_format})

    cfg = load_phase4_config(args.config)
    data = cfg.dataset
    if args.split_side is not None:
        data = replace(data, split_side=args.split_side)
    if args.views_per_scene is not None:
        data = replace(data, views_per_scene=args.views_per_scene)
    if args.max_pairs is not None:
        data = replace(data, max_pairs=args.max_pairs)
    cfg = replace(cfg, dataset=data)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=Path(args.output_dir))
    relative = Path(args.relative_config) if args.relative_config else None

    if args.target == "synthetic":
        report = run_synthetic(cfg)
        primary = report["primary_true_metadata"]
        print(json.dumps(to_jsonable({
            "azimuth": primary["azimuth"],
            "explicit": {k: primary["uncertainty_forms"]["explicit"].get(k)
                         for k in ("recovered_a", "recovered_b", "a_error", "b_error_m", "dense")},
            "sweep": report["azimuth_sweep"],
        }), indent=2))
    elif args.target == "ablation":
        from depthwizard.calibration.ablation import run_ablation

        report = run_ablation(cfg, relative_config=relative)
        print(json.dumps(to_jsonable(report["decision_inputs"]), indent=2))
    else:
        report = run_dfc(cfg, relative_config=relative, device=args.device, figures=not args.no_figures)
        print(json.dumps(to_jsonable({g: v["tiles"] for g, v in report["groups"].items()}), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
