"""Phase 4a v2: shadow-quality rules, reference selection, k estimation and
city-split diagnostics. Small hand-built scenes; no DFC2019 data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from depthwizard.calibration.config import load_phase4_config
from depthwizard.calibration.run import _measurement_params, process_dfc_tile, summarise_records
from depthwizard.calibration.shadow_anchor import (
    UNCERTAINTY_RAY_SPREAD_PROXY,
    ShadowScale,
    assess_reference_candidates,
    estimate_k,
    measure_building_shadows,
    shadow_height_constraints,
)
from depthwizard.ingest.synthetic import render_scene
from depthwizard.physics.height import Confidence
from depthwizard.shadows.measure import BuildingFootprint, MeasurementParams

REPO = Path(__file__).resolve().parents[1]
PARAMS = MeasurementParams(max_rays=64)


def _box(shape, r0, r1, c0, c1, name):
    return BuildingFootprint.from_bbox(name, shape=shape, row_min=r0, row_max=r1, col_min=c0, col_max=c1)


def test_frozen_config_uses_training_selected_values() -> None:
    cfg = load_phase4_config(REPO / "configs" / "phase4.yaml")
    assert cfg.measurement.gap_tolerance_px == 2.0
    assert _measurement_params(cfg).gap_tolerance_px == 2.0
    assert cfg.reference.isolation_px == 0
    assert cfg.measurement.max_merged_fraction == 0.5
    assert cfg.reference.k_estimator == "median"
    assert cfg.dataset.gsd_m is None


def test_merged_shadow_is_rejected_and_can_be_disabled() -> None:
    # Sun in the east (90 deg): shadows run west. A's shadow reaches B, and the
    # ray skips over B and carries on through B's shadow: a merged shadow.
    shape = (40, 80)
    a, b = _box(shape, 10, 30, 50, 60, "A"), _box(shape, 10, 30, 30, 40, "B")
    shadow = np.zeros(shape, bool)
    shadow[10:30, 20:50] = True
    shadow &= ~b.mask
    kwargs = dict(sun_azimuth_deg=90.0, params=PARAMS)

    checked = {s.building_id: s for s in measure_building_shadows([a, b], shadow, **kwargs)}
    assert checked["A"].merged_fraction == pytest.approx(1.0)
    assert checked["A"].rejection_reason == "merged_shadow"
    assert checked["B"].rejection_reason is None

    unchecked = {s.building_id: s for s in measure_building_shadows([a, b], shadow, max_merged_fraction=None, **kwargs)}
    assert unchecked["A"].rejection_reason is None


def _two_neighbours():
    # Sun in the south (180): shadows run north, 8 px each, onto CLS ground.
    shape = (60, 60)
    a, b = _box(shape, 30, 40, 10, 25, "A"), _box(shape, 30, 40, 28, 43, "B")
    shadow = np.zeros(shape, bool)
    shadow[22:30, 10:43] = True
    cls = np.full(shape, 2, np.uint8)
    cls[a.mask | b.mask] = 6
    agl = np.where(a.mask, 8.0, np.where(b.mask, 12.0, 0.0))
    shadows = measure_building_shadows([a, b], shadow, sun_azimuth_deg=180.0, params=PARAMS)
    return shadows, dict(agl=agl, agl_valid=np.ones(shape, bool), cls=cls, building_mask=cls == 6,
                         sun_azimuth_deg=180.0, surface_classes=(2,), min_area_px=50, min_solidity=0.8,
                         min_height_m=3.0, min_valid_fraction=0.9, min_shadow_zone_ground_fraction=0.8,
                         max_relative_spread=0.5)


def test_isolation_gate_and_its_ablation() -> None:
    shadows, kwargs = _two_neighbours()
    isolated = assess_reference_candidates(shadows, isolation_px=10, **kwargs)
    assert all(c.rejection_reason == "not_isolated" for c in isolated)
    relaxed = assess_reference_candidates(shadows, isolation_px=0, **kwargs)
    assert all(c.eligible for c in relaxed)


def test_reference_ranking_never_uses_ground_truth_height() -> None:
    shadows, kwargs = _two_neighbours()
    ranked = [c.building_id for c in assess_reference_candidates(shadows, isolation_px=0, **kwargs)]
    # Swap which building is taller: the ranking (spread, then area, then id) must not change.
    kwargs["agl"] = np.where(kwargs["agl"] == 8.0, 12.0, np.where(kwargs["agl"] == 12.0, 8.0, 0.0))
    swapped = [c.building_id for c in assess_reference_candidates(shadows, isolation_px=0, **kwargs)]
    assert ranked == swapped


def test_k_estimators() -> None:
    ratios = [0.40, 0.42, 1.60]  # one bad reference
    assert estimate_k(ratios, how="median") == pytest.approx(0.42)
    assert estimate_k(ratios, how="mean") == pytest.approx(np.mean(ratios))
    w = 1.0 / (np.array(ratios) * np.array([0.1, 0.2, 0.4])) ** 2
    assert estimate_k(ratios, [0.1, 0.2, 0.4], "inverse_variance") == pytest.approx((w * ratios).sum() / w.sum())
    with pytest.raises(ValueError):
        estimate_k(ratios, [0.1, 0.0, 0.4], "inverse_variance")


def test_failed_excluded_reduced_downweighted() -> None:
    # Sun in the south: rays launch from each box's top row and run north.
    shape = (60, 60)
    a = _box(shape, 30, 40, 10, 30, "full")      # 20/20 rays hit, lengths 10 and 5: NOMINAL
    b = _box(shape, 30, 40, 35, 55, "partial")   # 12/20 rays hit, lengths 8 and 4: REDUCED
    c = _box(shape, 45, 55, 10, 25, "none")      # nothing north of it: FAILED
    shadow = np.zeros(shape, bool)
    shadow[20:30, 10:20] = True
    shadow[25:30, 20:30] = True
    shadow[22:30, 35:41] = True
    shadow[26:30, 41:47] = True
    shadows = measure_building_shadows([a, b, c], shadow, sun_azimuth_deg=180.0, params=PARAMS)
    by_id = {s.building_id: s for s in shadows}
    assert by_id["none"].confidence is Confidence.FAILED
    scale = ShadowScale(metres_per_shadow_px=0.5, source="test")
    constraints = {c_.building_id: c_ for c_ in shadow_height_constraints(
        shadows, scale, uncertainty=UNCERTAINTY_RAY_SPREAD_PROXY, reduced_weight_factor=0.5)}
    assert not constraints["none"].usable
    for name in ("full", "partial"):
        con = constraints[name]
        factor = 1.0 if by_id[name].confidence is Confidence.NOMINAL else 0.5
        assert con.weight == pytest.approx(factor / con.dh_m**2)
    assert by_id["partial"].confidence is Confidence.REDUCED


def test_city_split_diagnostics(scene) -> None:
    cfg = load_phase4_config(REPO / "configs" / "phase4.yaml")
    cfg = replace(cfg, azimuth=replace(cfg.azimuth, min_building_px=100))
    render = render_scene(scene)
    grey = np.clip(np.rint(render.reflectance * 255), 0, 255).astype(np.uint8)
    cls = np.where(render.building_mask, 6, 2).astype(np.uint8)
    agl = render.height_m.astype(np.float64)
    tile = dict(z_rel=np.log(agl + 1.0) + 0.5, log_space=True, agl=agl, agl_valid=np.ones_like(render.building_mask),
                cls=cls, rgb_u8=np.stack([grey] * 3, axis=2), cfg=cfg)
    records = []
    for city, n in (("JAX", 1), ("OMA", 2)):
        for i in range(n):
            rec = process_dfc_tile(f"{city}_{i}", city=city, **tile)
            rec.update(image=f"{city}_{i}", scene_id=city)
            records.append(rec)
    summary = summarise_records(records, cfg)
    assert summary["JAX"]["tiles"]["total"] == 1
    assert summary["OMA"]["tiles"]["total"] == 2
    assert summary["combined"]["tiles"]["total"] == 3
    # 4 fixture buildings: 3 become references, 1 constraint remains -> the fit
    # is UNIDENTIFIABLE (insufficient constraints), reported per city.
    for city, n in (("JAX", 1), ("OMA", 2)):
        funnel = summary[city]["tiles"]["funnel"]
        assert funnel["sun_calibrated"] == n and funnel["fit_returned"] == 0
        assert summary[city]["fit_shadow"]["identifiability_classes"]["insufficient_constraints"] == n
