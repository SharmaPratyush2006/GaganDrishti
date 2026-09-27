"""Phase 5 metrics and validity masks.

Arrays here are either hand-built with a known answer, or the Phase 0
SYNTHETIC fixture's specified truth with a known error added. Nothing asserts
or implies a real-world accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthwizard.ingest.geotiff import read_raster
from depthwizard.validation.masks import (
    BUILDING,
    EXCLUSION_ORDER,
    OVERALL,
    TERRAIN,
    CategoryMasks,
    build_valid_mask,
    category_masks_from_labels,
)
from depthwizard.validation.metrics import (
    DEFAULT_DELTA_THRESHOLDS,
    MIN_PEARSON_SAMPLES,
    MetricAccumulator,
    MetricValue,
    RegionMetrics,
    compute_metrics,
    delta_accuracy,
    delta_name,
    mae,
    pearson,
    rmse,
)


def _all(shape):
    return np.ones(shape, dtype=bool)


# ---------------------------------------------------------------------------
# MAE / RMSE / bias
# ---------------------------------------------------------------------------


def test_mae_and_rmse_basic_case():
    p = np.array([[1.0, 2.0, 3.0]])
    r = np.array([[1.0, 1.0, 1.0]])
    assert mae(p, r, _all(p.shape), units="metres").value == pytest.approx(1.0)
    assert rmse(p, r, _all(p.shape), units="metres").value == pytest.approx(math.sqrt(5.0 / 3.0))
    m = compute_metrics(p, r, _all(p.shape), units="metres")
    assert m.bias.value == pytest.approx(1.0)
    assert m.valid_pixels == 3 and m.mae.n == 3
    assert m.mae.units == m.rmse.units == "metres"


def test_units_are_carried_not_assumed():
    p = np.array([[0.1, 0.2, 0.4]])
    m = compute_metrics(p, p * 2, _all(p.shape), units="relative (unitless)")
    assert m.mae.units == "relative (unitless)"
    assert m.pearson_r.units == "dimensionless"
    assert all(d.units == "fraction" for d in m.deltas)


def test_empty_valid_mask_gives_no_numbers_anywhere():
    p = np.array([[1.0, 2.0]])
    m = compute_metrics(p, p, np.zeros(p.shape, dtype=bool), units="metres")
    for metric in (m.mae, m.rmse, m.bias, m.pearson_r, *m.deltas):
        assert metric.value is None and metric.reason == "no valid pixels"
        assert metric.n == 0
    assert m.valid_pixels == 0


# ---------------------------------------------------------------------------
# Pearson
# ---------------------------------------------------------------------------


def test_pearson_basic_cases():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(20, 20))
    valid = _all(r.shape)
    assert pearson(2.0 * r + 1.0, r, valid).value == pytest.approx(1.0)
    assert pearson(-r, r, valid).value == pytest.approx(-1.0)
    p = r + rng.normal(size=r.shape)
    assert pearson(p, r, valid).value == pytest.approx(np.corrcoef(p.ravel(), r.ravel())[0, 1], abs=1e-12)


def test_pearson_is_stable_on_large_offsets():
    # Elevations around 540 m: a raw sum-of-squares formula loses digits here.
    rng = np.random.default_rng(1)
    r = 540.0 + rng.normal(scale=1e-3, size=(50, 50))
    p = r + rng.normal(scale=1e-3, size=r.shape)
    expected = np.corrcoef(p.ravel(), r.ravel())[0, 1]
    assert pearson(p, r, _all(r.shape)).value == pytest.approx(expected, abs=1e-9)


def test_pearson_insufficient_samples_is_not_a_number():
    p = np.array([[1.0, 2.0]])
    value = pearson(p, p * 3, _all(p.shape))
    assert MIN_PEARSON_SAMPLES > 2
    assert value.value is None and "insufficient samples" in value.reason


@pytest.mark.parametrize("which", ["prediction", "reference"])
def test_pearson_zero_variance_is_not_a_number(which):
    varying = np.array([[1.0, 2.0, 3.0, 4.0]])
    constant = np.full_like(varying, 7.0)
    p, r = (constant, varying) if which == "prediction" else (varying, constant)
    value = pearson(p, r, _all(p.shape))
    assert value.value is None
    assert f"zero variance in {which}" in value.reason


# ---------------------------------------------------------------------------
# delta thresholds
# ---------------------------------------------------------------------------


def test_delta_threshold_calculations():
    p = np.array([[1.0, 1.2, 1.3, 2.0, 10.0, 0.5]])
    r = np.ones_like(p)
    # ratios max(p/r, r/p): 1, 1.2, 1.3, 2, 10, 2
    d1, d2, d3 = delta_accuracy(p, r, _all(p.shape))
    assert d1.value == pytest.approx(2 / 6)   # < 1.25
    assert d2.value == pytest.approx(3 / 6)   # < 1.5625
    assert d3.value == pytest.approx(3 / 6)   # < 1.953125 (2.0 is not below)
    assert [d.name for d in (d1, d2, d3)] == ["delta<1.25", "delta<1.25^2", "delta<1.25^3"]
    assert d1.n == 6


def test_delta_excludes_non_positive_values_and_counts_them():
    p = np.array([[0.0, -1.0, 2.0, 4.0]])
    r = np.array([[1.0, 1.0, 2.0, -3.0]])
    m = compute_metrics(p, r, _all(p.shape), units="metres")
    assert m.delta_domain_pixels == 1
    assert m.delta_excluded_nonpositive == 3
    assert all(d.value == pytest.approx(1.0) and d.n == 1 for d in m.deltas)
    # MAE still covers every valid pixel -- only delta has the positive domain.
    assert m.mae.n == 4


def test_delta_with_no_positive_pixel_is_not_a_number():
    p = np.zeros((2, 2))
    m = compute_metrics(p, p, _all(p.shape), units="metres")
    for d in m.deltas:
        assert d.value is None and "no valid pixel has prediction > 0 and reference > 0" in d.reason


def test_delta_thresholds_must_exceed_one():
    with pytest.raises(ValueError):
        MetricAccumulator(delta_thresholds=(1.0,))
    assert delta_name(1.25**2) == "delta<1.25^2"


# ---------------------------------------------------------------------------
# NaN / nodata / streaming
# ---------------------------------------------------------------------------


def test_accumulator_refuses_nan_hidden_inside_the_valid_mask():
    p = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(p, np.ones_like(p), _all(p.shape), units="metres")


def test_nan_and_nodata_are_excluded_through_the_mask():
    p = np.array([[1.0, np.nan, -9999.0, 3.0]])
    r = np.array([[1.0, 1.0, 1.0, np.nan]])
    validity = build_valid_mask(p, r, prediction_nodata=-9999.0)
    m = compute_metrics(p, r, validity.valid, units="metres")
    assert m.valid_pixels == 1 and m.mae.value == 0.0


def test_streaming_equals_one_shot():
    rng = np.random.default_rng(2)
    p, r = rng.normal(5, 2, (40, 40)), rng.normal(5, 2, (40, 40))
    one = compute_metrics(p, r, _all(p.shape), units="m")
    acc = MetricAccumulator()
    acc.add(p[:15], r[:15], _all((15, 40)))
    acc.add(p[15:], r[15:], _all((25, 40)))
    two = acc.result(category="overall", units="m")
    assert two.valid_pixels == one.valid_pixels
    for a, b in ((one.mae, two.mae), (one.rmse, two.rmse), (one.pearson_r, two.pearson_r)):
        assert a.value == pytest.approx(b.value, abs=1e-12)
    for a, b in zip(one.deltas, two.deltas):
        assert a.value == b.value


def test_metric_value_cannot_be_absent_without_a_reason_or_non_finite():
    with pytest.raises(ValueError):
        MetricValue("mae", None, "m", 0)
    with pytest.raises(ValueError):
        MetricValue("mae", float("nan"), "m", 3)
    with pytest.raises(ValueError):
        MetricValue("mae", 1.0, "m", 3, reason="both")


def test_unavailable_region_has_only_absences():
    m = RegionMetrics.unavailable("terrain", "m", "no terrain label")
    for metric in (m.mae, m.rmse, m.bias, m.pearson_r, *m.deltas):
        assert metric.value is None and metric.reason == "no terrain label"
    assert m.to_dict()["mae"]["value"] == "not yet measured"
    assert len(m.deltas) == len(DEFAULT_DELTA_THRESHOLDS)


# ---------------------------------------------------------------------------
# On the Phase 0 SYNTHETIC fixture (specified heights)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def truth_height(generated):
    return read_raster(generated.height_path).array.astype(np.float64)


def test_perfect_prediction_on_fixture(truth_height):
    valid = _all(truth_height.shape)
    building = truth_height > 0
    overall = compute_metrics(truth_height, truth_height, valid, units="metres")
    assert overall.mae.value == 0.0 and overall.rmse.value == 0.0 and overall.bias.value == 0.0
    assert overall.pearson_r.value == pytest.approx(1.0)
    on_buildings = compute_metrics(truth_height, truth_height, building, units="metres")
    assert all(d.value == 1.0 for d in on_buildings.deltas)
    assert on_buildings.delta_domain_pixels == int(building.sum())


def test_known_errors_on_fixture_give_known_mae_and_rmse(truth_height):
    valid = _all(truth_height.shape)
    offset = compute_metrics(truth_height + 0.5, truth_height, valid, units="metres")
    assert offset.mae.value == pytest.approx(0.5) and offset.rmse.value == pytest.approx(0.5)
    assert offset.bias.value == pytest.approx(0.5)
    signs = np.where(np.indices(truth_height.shape).sum(axis=0) % 2 == 0, 1.0, -1.0)
    alternating = compute_metrics(truth_height + 2.0 * signs, truth_height, valid, units="metres")
    assert alternating.mae.value == pytest.approx(2.0) and alternating.rmse.value == pytest.approx(2.0)
    mixed = np.zeros_like(truth_height)
    mixed.flat[:2] = (3.0, 4.0)  # errors 3 and 4 on 2 pixels, 0 elsewhere
    m = compute_metrics(truth_height + mixed, truth_height, valid, units="metres")
    assert m.mae.value == pytest.approx(7.0 / truth_height.size)
    assert m.rmse.value == pytest.approx(math.sqrt(25.0 / truth_height.size))


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------


def test_valid_mask_counts_every_exclusion_once():
    p = np.array([[1.0, np.nan, -9999.0, 2.0, 3.0, np.inf]])
    r = np.array([[1.0, 1.0, 1.0, -1.0, np.nan, 1.0]])
    v = build_valid_mask(p, r, prediction_nodata=-9999.0, reference_nodata=-1.0)
    assert v.valid.tolist() == [[True, False, False, False, False, False]]
    assert v.exclusions == {
        "prediction_nodata": 1, "prediction_nonfinite": 2, "reference_nodata": 1,
        "reference_nonfinite": 1, "reference_invalid": 0, "outside_region": 0, "excluded": 0,
    }
    assert v.valid_pixels + sum(v.exclusions.values()) == v.total_pixels
    assert tuple(v.to_dict()["exclusion_precedence"]) == EXCLUSION_ORDER


def test_nodata_takes_precedence_over_nan():
    p = np.array([[-9999.0, np.nan]])
    r = np.array([[np.nan, -5.0]])
    v = build_valid_mask(p, r, prediction_nodata=-9999.0, reference_nodata=-5.0)
    # Pixel 0: prediction nodata wins over reference NaN; pixel 1: prediction NaN
    # wins over reference nodata. Each pixel is counted once.
    assert v.exclusions["prediction_nodata"] == 1
    assert v.exclusions["prediction_nonfinite"] == 1
    assert v.exclusions["reference_nodata"] == 0 and v.exclusions["reference_nonfinite"] == 0


def test_region_exclude_and_reference_validity():
    p = np.ones((2, 3))
    r = np.ones((2, 3))
    region = np.array([[True, True, True], [False, True, True]])
    exclude = np.array([[False, True, False], [False, False, False]])
    ref_valid = np.array([[True, True, True], [True, True, False]])
    v = build_valid_mask(p, r, reference_valid=ref_valid, region=region, exclude=exclude)
    assert v.exclusions["reference_invalid"] == 1
    assert v.exclusions["outside_region"] == 1
    assert v.exclusions["excluded"] == 1
    assert v.valid_pixels == 3


def test_valid_mask_rejects_grid_mismatch():
    with pytest.raises(ValueError, match="same grid"):
        build_valid_mask(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="region"):
        build_valid_mask(np.ones((2, 2)), np.ones((2, 2)), region=np.ones((3, 3), dtype=bool))


# ---------------------------------------------------------------------------
# Terrain / building masks
# ---------------------------------------------------------------------------


def test_category_masks_from_labels():
    cls = np.array([[2, 6, 9], [5, 6, 2]])
    c = category_masks_from_labels(cls, terrain_classes=[2], building_classes=[6], source="CLS")
    assert c.get(TERRAIN).tolist() == [[True, False, False], [False, False, True]]
    assert c.get(BUILDING).tolist() == [[False, True, False], [False, True, False]]
    assert c.get(OVERALL).all()
    with pytest.raises(KeyError):
        c.get("water")
    with pytest.raises(ValueError, match="both terrain and building"):
        category_masks_from_labels(cls, terrain_classes=[2, 6], building_classes=[6], source="CLS")


def test_terrain_and_building_metrics_are_separated():
    cls = np.array([[2, 2, 6, 6]])
    ref = np.array([[0.0, 0.0, 10.0, 20.0]])
    pred = np.array([[1.0, -1.0, 12.0, 20.0]])
    c = category_masks_from_labels(cls, terrain_classes=[2], building_classes=[6], source="CLS")
    valid = _all(ref.shape)
    terrain = compute_metrics(pred, ref, valid & c.terrain, units="m", category=TERRAIN)
    building = compute_metrics(pred, ref, valid & c.building, units="m", category=BUILDING)
    assert terrain.mae.value == pytest.approx(1.0) and terrain.valid_pixels == 2
    assert building.mae.value == pytest.approx(1.0) and building.valid_pixels == 2
    assert building.rmse.value == pytest.approx(math.sqrt(2.0))
    assert terrain.bias.value == pytest.approx(0.0)


def test_category_masks_refuse_overlap_and_allow_absence():
    with pytest.raises(ValueError, match="overlap"):
        CategoryMasks(shape=(1, 2), terrain=np.array([[True, True]]), building=np.array([[True, False]]), source="x")
    c = CategoryMasks(shape=(1, 2), terrain=None, building=np.array([[True, False]]), source="x")
    assert c.get(TERRAIN) is None
