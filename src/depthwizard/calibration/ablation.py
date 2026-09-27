"""TRAINING-ONLY threshold ablations for Phase 4a, plus a per-city height census.

Every threshold Phase 4a uses on real data (shadow gap tolerance, isolation,
merged-shadow check, the k estimator, the shadow threshold mode) is chosen
from this study, which reads **only the Phase 3 training split**. The held-out
split is touched once, by the final frozen run in :mod:`~depthwizard.calibration.run`.

What is measured, per building, per variant (no Phase 3 model is needed --
shadow measurement uses RGB and CLS, and the quality criteria use training AGL):

* measurement outcome: success, failure kind (zero hits / partial hits / too
  few launch pixels), NOMINAL/REDUCED, edge truncation, merged-shadow fraction;
* every reference gate evaluated independently, so isolation variants can be
  combined afterwards;
* **measurement validity**: within a tile, does the shadow length ``L`` track
  the true height ``h``? (Spearman of ``L`` vs ``h``; dispersion of
  ``log(h / L)``; leave-references-out prediction error of ``h = k * L``).

Selection criteria are measurement validity and robustness on training data,
never a held-out accuracy.

The height census (:func:`height_census`) reads CLS/AGL for one view per
geographic tile of **both** splits. It is a data description, labelled by
split, and nothing is tuned on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image

from depthwizard.calibration.azimuth import estimate_sun_azimuth
from depthwizard.calibration.config import Phase4Config
from depthwizard.calibration.diagnostics import height_distribution, to_jsonable, write_json
from depthwizard.calibration.footprints import extract_building_footprints
from depthwizard.calibration.shadow_anchor import _shadow_zone, estimate_k, measure_building_shadows
from depthwizard.logging_setup import get_logger
from depthwizard.shadows.detector import ClassicalShadowDetector, DetectionContext, _otsu_threshold_u8

__all__ = ["GAPS", "ISOLATIONS", "THRESHOLD_MODES", "building_gate_records", "summarise_ablation",
           "height_census", "run_ablation"]

log = get_logger(__name__)

GAPS: tuple[float, ...] = (2.0, 4.0, 6.0)
#: isolation_px values: the original 10 px, and 0 (disabled).
ISOLATIONS: tuple[int, ...] = (10, 0)
#: "image_otsu": the Phase 2 detector as shipped. "ground_otsu": Otsu computed over
#: the shadow-receiving surface only, passed to the same detector as a fixed threshold.
THRESHOLD_MODES: tuple[str, ...] = ("image_otsu", "ground_otsu")
K_ESTIMATORS: tuple[str, ...] = ("median", "mean", "inverse_variance")


def _shadow_mask(rgb: np.ndarray, surface: np.ndarray, mode: str) -> np.ndarray:
    if mode == "image_otsu":
        detector = ClassicalShadowDetector()
    elif mode == "ground_otsu":
        value = rgb.max(axis=2)
        threshold_u8 = _otsu_threshold_u8(value[surface])[0] if surface.sum() > 100 else 0
        detector = ClassicalShadowDetector(threshold=threshold_u8 / 255.0)
    else:
        raise ValueError(f"unknown threshold mode {mode!r}")
    return detector.detect(rgb, DetectionContext(band_order="rgb")).mask & surface


def building_gate_records(
    tile_id: str,
    city: str,
    *,
    rgb: np.ndarray,
    cls: np.ndarray,
    agl: np.ndarray,
    cfg: Phase4Config,
    variants: Sequence[tuple[str, Sequence[float]]] = (("image_otsu", GAPS), ("ground_otsu", (2.0,))),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure one training tile under every variant. Returns (tile record, building records).

    ``variants`` pairs a shadow threshold mode with the gap tolerances to run it
    at; by default the Phase 2 image Otsu at every gap, and the ground-only
    Otsu check at 2 px only.
    """
    ref = cfg.reference
    valid = np.isfinite(agl)
    surface = np.isin(cls, np.asarray(cfg.dataset.shadow_surface_classes))
    extraction = extract_building_footprints(
        cls, tile_id=tile_id, building_class=cfg.dataset.building_class,
        min_area_px=cfg.footprints.min_area_px, connectivity=cfg.footprints.connectivity,
        reject_edge_touching=cfg.footprints.reject_edge_touching,
    )
    tile: dict[str, Any] = {"tile": tile_id, "city": city, "footprints": len(extraction.footprints),
                            "components_found": extraction.n_components, "azimuth": {}}
    rows: list[dict[str, Any]] = []

    geometry: dict[str, dict[str, Any]] = {}
    for fp in extraction.footprints:
        mask = fp.mask
        r_idx, c_idx = np.nonzero(mask)
        pad = max(ISOLATIONS) + 1
        r0, r1 = max(0, r_idx.min() - pad), min(mask.shape[0], r_idx.max() + pad + 1)
        c0, c1 = max(0, c_idx.min() - pad), min(mask.shape[1], c_idx.max() + pad + 1)
        local = mask[r0:r1, c0:c1]
        isolated = {}
        for iso in ISOLATIONS:
            if iso == 0:
                isolated[iso] = True
            else:
                grown = ndimage.binary_dilation(local, iterations=iso)
                isolated[iso] = not (grown & extraction.building_mask[r0:r1, c0:c1] & ~local).any()
        sel = valid & mask
        geometry[fp.building_id] = {
            "area": int(mask.sum()),
            "solidity": float(local.sum() / convex_hull_image(local).sum()),
            "isolated": isolated,
            "valid_fraction": float(sel.sum() / mask.sum()),
            "h": float(np.median(agl[sel])) if sel.any() else None,
        }

    for mode, gaps in variants:
        shadow = _shadow_mask(rgb, surface, mode)
        azimuth = estimate_sun_azimuth(
            shadow, extraction.building_mask, band_px=cfg.azimuth.band_px, step_deg=cfg.azimuth.step_deg,
            min_building_px=cfg.azimuth.min_building_px, min_peak_to_median=cfg.azimuth.min_peak_to_median,
        )
        tile["azimuth"][mode] = {"ok": azimuth.ok, "reason": azimuth.failure_reason,
                                 "shadow_fraction": float(shadow.mean())}
        if not azimuth.ok or not extraction.footprints:
            continue
        for gap in gaps:
            params = _params_for_gap(cfg, gap)
            shadows = measure_building_shadows(
                extraction.footprints, shadow, sun_azimuth_deg=float(azimuth.sun_azimuth_deg), params=params,
                max_edge_terminated_fraction=cfg.measurement.max_edge_terminated_fraction,
                max_merged_fraction=None,  # recorded, applied per variant in the summary
                occluder_mask=extraction.building_mask,
            )
            for s in shadows:
                m = s.measurement
                g = geometry[s.building_id]
                failure = None
                if not m.ok:
                    reason = m.failure_reason or ""
                    failure = ("zero_hits" if reason.startswith("only 0/") else
                               "partial_hits" if reason.startswith("only") and "rays" in reason else
                               "few_launch_px" if "boundary" in reason else "other")
                zone_ground = None
                if m.ok:
                    zone = _shadow_zone(s.footprint.mask, float(azimuth.sun_azimuth_deg), float(m.shadow_length_px))
                    zone_ground = float(surface[zone].mean()) if zone.any() else 0.0
                rows.append({
                    "tile": tile_id, "city": city, "mode": mode, "gap": gap, "id": s.building_id,
                    "area": g["area"], "solidity": g["solidity"], "isolated": {str(k): v for k, v in g["isolated"].items()},
                    "valid_fraction": g["valid_fraction"], "h": g["h"],
                    "ok": m.ok, "failure": failure, "hit_fraction": m.hit_fraction,
                    "L": m.shadow_length_px, "spread": m.ray_length_spread_px,
                    "confidence": s.confidence.value, "rejection": s.rejection_reason and s.rejection_reason.split(":")[0],
                    "merged": s.merged_fraction, "zone_ground": zone_ground,
                })
    return tile, rows


def _params_for_gap(cfg: Phase4Config, gap: float):
    from depthwizard.calibration.run import _measurement_params

    return replace(_measurement_params(cfg), gap_tolerance_px=float(gap))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _reference_gate(row: dict[str, Any], cfg: Phase4Config, isolation: int, merged_check: bool) -> str | None:
    ref = cfg.reference
    if not row["ok"]:
        return "failed_measurement"
    if row["rejection"]:
        return row["rejection"]
    if merged_check and row["merged"] is not None and row["merged"] > (cfg.measurement.max_merged_fraction or 0.5):
        return "merged_shadow"
    if row["confidence"] != "nominal":
        return "not_nominal"
    if row["area"] < ref.min_area_px:
        return "below_reference_min_area"
    if row["solidity"] < ref.min_solidity:
        return "low_solidity"
    if not row["isolated"][str(isolation)]:
        return "not_isolated"
    if row["valid_fraction"] < ref.min_valid_fraction:
        return "insufficient_valid_agl"
    if row["h"] is None or row["h"] < ref.min_height_m:
        return "below_reference_min_height"
    if row["spread"] is None or row["spread"] / row["L"] > ref.max_relative_spread:
        return "unreliable_shadow_spread"
    if row["zone_ground"] is None or row["zone_ground"] < ref.min_shadow_zone_ground_fraction:
        return "shadow_zone_not_open_ground"
    return None


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _usable_with_truth(row: dict[str, Any], cfg: Phase4Config, merged_check: bool) -> bool:
    if not (row["ok"] and not row["rejection"] and row["h"] is not None):
        return False
    if row["valid_fraction"] < cfg.reference.min_valid_fraction:
        return False
    return not (merged_check and row["merged"] is not None and row["merged"] > (cfg.measurement.max_merged_fraction or 0.5))


def summarise_variant(rows: Sequence[dict[str, Any]], cfg: Phase4Config, *, isolation: int, merged_check: bool) -> dict[str, Any]:
    """Measurement quality + reference availability + validity for one variant."""
    by_tile: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_tile.setdefault(r["tile"], []).append(r)
    usable = [r for r in rows if r["ok"] and not r["rejection"]]
    rel = np.array([r["spread"] / r["L"] for r in usable if r["spread"] is not None and r["L"]])
    merged = np.array([r["merged"] for r in usable if r["merged"] is not None])
    failures: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    gate_counts: dict[str, int] = {}
    eligible_by_tile: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        gate = _reference_gate(r, cfg, isolation, merged_check)
        if gate is None:
            eligible_by_tile.setdefault(r["tile"], []).append(r)
        else:
            gate_counts[gate] = gate_counts.get(gate, 0) + 1

    spearman, mad_log_k = [], []
    for tile_rows in by_tile.values():
        good = [r for r in tile_rows if _usable_with_truth(r, cfg, merged_check) and r["L"] > 0]
        if len(good) >= 5:
            spearman.append(_spearman([r["L"] for r in good], [r["h"] for r in good]))
        logs = [math.log(r["h"] / r["L"]) for r in good if r["h"] >= cfg.reference.min_height_m]
        if len(logs) >= 3:
            logs = np.array(logs); mad_log_k.append(float(np.median(np.abs(logs - np.median(logs)))))

    # Leave-references-out: k from the production-ranked top references,
    # predict h = k * L for every other usable building with truth.
    k_errors: dict[str, list[float]] = {k: [] for k in K_ESTIMATORS}
    k_undefined = {k: 0 for k in K_ESTIMATORS}
    tiles_calibratable = 0
    for tile, elig in eligible_by_tile.items():
        if len(elig) < cfg.reference.min_count:
            continue
        tiles_calibratable += 1
        refs = sorted(elig, key=lambda r: (r["spread"] / r["L"], -r["area"], r["id"]))[: cfg.reference.primary_count]
        ids = {r["id"] for r in refs}
        targets = [r for r in by_tile[tile] if r["id"] not in ids and _usable_with_truth(r, cfg, merged_check)]
        ratios = [r["h"] / r["L"] for r in refs]
        rel_spreads = [r["spread"] / r["L"] for r in refs]
        for how in K_ESTIMATORS:
            try:
                k = estimate_k(ratios, rel_spreads, how)
            except ValueError:
                k_undefined[how] += 1
                continue
            k_errors[how] += [k * r["L"] - r["h"] for r in targets]

    def err(e: list[float]) -> dict[str, Any]:
        e = np.asarray(e)
        return {"n": int(e.size), "mae_m": float(np.abs(e).mean()) if e.size else None,
                "median_ae_m": float(np.median(np.abs(e))) if e.size else None,
                "bias_m": float(e.mean()) if e.size else None}

    return {
        "buildings": len(rows),
        "measured_ok": sum(r["ok"] for r in rows),
        "measured_ok_fraction": (sum(r["ok"] for r in rows) / len(rows)) if rows else None,
        "usable": len(usable),
        "failure_kinds": failures,
        "median_hit_fraction": float(np.median([r["hit_fraction"] for r in rows])) if rows else None,
        "median_relative_spread": float(np.median(rel)) if rel.size else None,
        "high_spread_fraction": float((rel > cfg.reference.max_relative_spread).mean()) if rel.size else None,
        "merged_fraction_of_usable": float((merged > (cfg.measurement.max_merged_fraction or 0.5)).mean()) if merged.size else None,
        "edge_truncated": sum(1 for r in rows if r["rejection"] == "shadow_truncated_by_tile_edge"),
        "reference_gate_rejections": dict(sorted(gate_counts.items(), key=lambda kv: -kv[1])),
        "eligible_references": sum(len(v) for v in eligible_by_tile.values()),
        "tiles_with_ge2_references": tiles_calibratable,
        "validity_L_vs_h": {
            "tiles_with_ge5_usable": len(spearman),
            "median_within_tile_spearman": float(np.median(spearman)) if spearman else None,
            "tiles_with_ge3_usable_ge_min_height": len(mad_log_k),
            "median_within_tile_mad_log_k": float(np.median(mad_log_k)) if mad_log_k else None,
        },
        "leave_references_out_h_error": {k: err(v) for k, v in k_errors.items()},
        "k_estimator_undefined_tiles": k_undefined,
    }


def height_scaling(rows: Sequence[dict[str, Any]], cfg: Phase4Config) -> list[dict[str, Any]]:
    """Median L / (tile median L) per true-height bin: does L scale with h at all?"""
    by_tile: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if _usable_with_truth(r, cfg, merged_check=False) and r["L"] > 0:
            by_tile.setdefault(r["tile"], []).append(r)
    pairs = []
    for tile_rows in by_tile.values():
        if len(tile_rows) < 5:
            continue
        median_l = float(np.median([r["L"] for r in tile_rows]))
        median_h = float(np.median([r["h"] for r in tile_rows]))
        pairs += [(r["h"], r["L"] / median_l, r["h"] / median_h) for r in tile_rows]
    out = []
    arr = np.array(pairs) if pairs else np.zeros((0, 3))
    for low, high in ((0, 3), (3, 5), (5, 10), (10, 20), (20, math.inf)):
        sel = (arr[:, 0] >= low) & (arr[:, 0] < high) if arr.size else np.zeros(0, bool)
        out.append({"true_height_m": f"[{low:g}, {high:g})", "n": int(sel.sum()),
                    "median_L_over_tile_median_L": float(np.median(arr[sel, 1])) if sel.any() else None,
                    "median_h_over_tile_median_h": float(np.median(arr[sel, 2])) if sel.any() else None})
    return out


def summarise_ablation(tiles: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], cfg: Phase4Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for city in ("JAX", "OMA", "combined"):
        t = [x for x in tiles if city == "combined" or x["city"] == city]
        r_city = [x for x in rows if city == "combined" or x["city"] == city]
        entry: dict[str, Any] = {"tiles": len(t), "tiles_with_footprints": sum(1 for x in t if x["footprints"] > 0)}
        for mode in THRESHOLD_MODES:
            az = [x["azimuth"].get(mode) for x in t if mode in x["azimuth"]]
            entry[f"azimuth_{mode}"] = {
                "ok": sum(1 for a in az if a["ok"]),
                "failed_no_buildings": sum(1 for a in az if not a["ok"] and "building pixels" in (a["reason"] or "")),
                "failed_weak_peak": sum(1 for a in az if not a["ok"] and "peak/median" in (a["reason"] or "")),
                "median_shadow_fraction": float(np.median([a["shadow_fraction"] for a in az])) if az else None,
            }
        variants: dict[str, Any] = {}
        for mode in THRESHOLD_MODES:
            for gap in sorted({x["gap"] for x in r_city if x["mode"] == mode}):
                sub = [x for x in r_city if x["mode"] == mode and x["gap"] == gap]
                for iso in ISOLATIONS:
                    for merged_check in (False, True):
                        key = f"{mode}|gap={gap:g}|isolation={iso}|merged_check={merged_check}"
                        variants[key] = summarise_variant(sub, cfg, isolation=iso, merged_check=merged_check)
                variants[f"{mode}|gap={gap:g}|height_scaling"] = height_scaling(sub, cfg)
        entry["variants"] = variants
        out[city] = entry
    return out


# ---------------------------------------------------------------------------
# Height census (both splits, one view per geographic tile; description only)
# ---------------------------------------------------------------------------


def height_census(pairs: Sequence[Any], cfg: Phase4Config, *, cls_suffix: str) -> dict[str, Any]:
    from depthwizard.relative.data import open_raster

    out: dict[str, Any] = {}
    structure4 = ndimage.generate_binary_structure(2, 1)
    for city in ("JAX", "OMA"):
        first: dict[str, Any] = {}
        for p in sorted((p for p in pairs if p.city == city), key=lambda p: p.stem):
            first.setdefault(p.scene_id, p)
        heights: list[float] = []
        speckle_px = building_px = small = comps = 0
        for p in first.values():
            with open_raster(p.height_path.with_name(p.stem + cls_suffix)) as src:
                cls = src.read(1)
            with open_raster(p.height_path) as src:
                agl = src.read(1).astype(np.float64)
            building = cls == cfg.dataset.building_class
            labels, n = ndimage.label(building, structure=structure4)
            if not n:
                continue
            sizes = np.bincount(labels.ravel())[1:]
            building_px += int(building.sum()); comps += int(n)
            small += int((sizes < cfg.footprints.min_area_px).sum())
            speckle_px += int(sizes[sizes < cfg.footprints.min_area_px].sum())
            finite = np.where(np.isfinite(agl), agl, np.nan)
            medians = ndimage.labeled_comprehension(
                finite, labels, np.arange(1, n + 1),
                lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan, float, np.nan)
            heights += [float(m) for m, s in zip(medians, sizes) if s >= cfg.footprints.min_area_px and np.isfinite(m)]
        out[city] = {
            "geographic_tiles": len(first),
            "building_heights_m (components >= min_area, 4-connected, median AGL)": height_distribution(heights),
            "components": comps,
            "components_below_min_area": small,
            "building_pixels": building_px,
            "building_pixels_in_components_below_min_area": speckle_px,
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_ablation(cfg: Phase4Config, *, relative_config: Path | None = None, views_per_scene: int = 4) -> dict[str, Any]:
    """Run the TRAINING-only ablation and the two-split height census."""
    from depthwizard.calibration.run import _crop_offsets, select_views
    from depthwizard.relative.config import load_relative_config
    from depthwizard.relative.data import discover_pairs, open_raster, split_pairs

    rel_cfg = load_relative_config(relative_config or cfg.relative_config)
    train, val = split_pairs(discover_pairs(rel_cfg.dataset), rel_cfg.dataset.split)
    chosen = select_views(train, views_per_scene)
    out_dir = cfg.output_dir.parent / f"{cfg.output_dir.name}_training_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.dataset.crop_size
    tiles: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(chosen):
        with open_raster(pair.image_path) as src:
            rgb_full = np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))
        with open_raster(pair.height_path.with_name(pair.stem + cfg.dataset.cls_suffix)) as src:
            cls_full = src.read(1)
        with open_raster(pair.height_path) as src:
            agl_full = src.read(1).astype(np.float64)
        for row in _crop_offsets(cls_full.shape[0], size):
            for col in _crop_offsets(cls_full.shape[1], size):
                window = (slice(row, row + size), slice(col, col + size))
                tile, building_rows = building_gate_records(
                    f"{pair.stem}_r{row}_c{col}", pair.city,
                    rgb=np.ascontiguousarray(rgb_full[window]), cls=cls_full[window], agl=agl_full[window], cfg=cfg,
                )
                tiles.append(tile)
                rows += building_rows
        if (index + 1) % 25 == 0 or index + 1 == len(chosen):
            log.info("ablation progress", extra={"pairs_done": index + 1, "pairs": len(chosen)})
    with (out_dir / "buildings.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(to_jsonable(r)) + "\n")

    summary = summarise_ablation(tiles, rows, cfg)
    report = {
        "data": "Phase 3 TRAINING split only (held-out split untouched)",
        "views_per_scene": views_per_scene,
        "training_images": len(chosen),
        "training_geographic_tiles": len({p.scene_id for p in chosen}),
        "training_tiles_512px": len(tiles),
        "ablation": summary,
        "height_census": {
            "note": "Description of the data (one view per geographic tile); nothing is tuned on it.",
            "train": height_census(train, cfg, cls_suffix=cfg.dataset.cls_suffix),
            "held_out": height_census(val, cfg, cls_suffix=cfg.dataset.cls_suffix),
        },
        "decision_inputs": {
            city: {k: {kk: v[kk] for kk in ("measured_ok_fraction", "tiles_with_ge2_references", "validity_L_vs_h")}
                   for k, v in summary[city]["variants"].items() if not k.endswith("height_scaling")}
            for city in ("combined",)
        },
    }
    write_json(out_dir / "ablation_report.json", report)
    return report
