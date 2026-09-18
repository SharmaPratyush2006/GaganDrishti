"""The reference-data interface and the evaluation report.

This module is the boundary through which *externally measured* heights enter
DepthWizard. The tests below are therefore as much about what must NOT happen
as about what must:

* no reference data ships with the repository, and none is generated;
* a reference cannot exist without a stated source;
* a synthetic reference can never be reported as a real-world one;
* with no references supplied, every derived figure is absent -- not zero, not
  a default, not an estimate.

Only the last group exercises metrics at all, and those run against references
the test itself constructs and tags SYNTHETIC.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from depthwizard.physics import Confidence
from depthwizard.shadows import BuildingFootprint, estimate_heights_from_mask
from depthwizard.validation import (
    NOT_MEASURED,
    AccuracyMetrics,
    EvaluationReport,
    ReferenceKind,
    ReferenceMeasurement,
    ReferenceSet,
    evaluate,
    load_reference_set,
)

from conftest import build_analytic_shadow_mask

SHAPE = (200, 200)


@pytest.fixture
def estimates():
    """Four predictions from an exactly-representable analytic scene."""
    boxes = {
        "tower_a": (20, 50, 150, 180, 60.0),
        "block_b": (70, 100, 150, 180, 24.0),
        "slab_c": (120, 150, 150, 180, 90.0),
        "low_d": (170, 195, 150, 180, 12.0),
    }
    mask = np.zeros(SHAPE, dtype=bool)
    footprints = []
    for name, (row_min, row_max, col_min, col_max, length_px) in boxes.items():
        mask |= build_analytic_shadow_mask(
            SHAPE,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
            sun_azimuth_deg=90.0,
            shadow_length_px=length_px,
        )
        footprints.append(
            BuildingFootprint.from_bbox(
                name,
                shape=SHAPE,
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
            )
        )
    for footprint in footprints:
        mask[footprint.mask] = False
    return estimate_heights_from_mask(
        footprints, mask, gsd_m=0.5, sun_azimuth_deg=90.0, sun_elevation_deg=45.0
    )


@pytest.fixture
def synthetic_references() -> ReferenceSet:
    """Heights that were SPECIFIED, not measured -- and tagged as such."""
    return ReferenceSet.from_records(
        [
            {"building_id": "tower_a", "reference_height_m": 30.0, "source": "fixture spec"},
            {"building_id": "block_b", "reference_height_m": 12.0, "source": "fixture spec"},
            {"building_id": "slab_c", "reference_height_m": 45.0, "source": "fixture spec"},
            {"building_id": "low_d", "reference_height_m": 6.0, "source": "fixture spec"},
        ],
        kind=ReferenceKind.SYNTHETIC,
        description="analytic scene, heights chosen not measured",
    )


# ---------------------------------------------------------------------------
# No reference data ships with this repository
# ---------------------------------------------------------------------------


def test_no_reference_files_are_committed():
    """The repository must not contain reference heights of any kind.

    If one ever appears it must arrive deliberately, with a source, not as a
    file that quietly starts backing an accuracy claim.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    suspicious = []
    for pattern in ("**/*reference*.json", "**/*reference*.csv", "**/*ground_truth*.csv"):
        for path in repo_root.glob(pattern):
            if any(part in (".venv", ".git", ".pytest_cache") for part in path.parts):
                continue
            suspicious.append(path)
    assert not suspicious, f"unexpected reference data committed: {suspicious}"


# ---------------------------------------------------------------------------
# A reference is inadmissible without a source
# ---------------------------------------------------------------------------


def test_a_reference_requires_a_non_empty_source():
    with pytest.raises(ValueError, match="source"):
        ReferenceMeasurement(building_id="a", reference_height_m=30.0, source="")


def test_a_reference_requires_a_whitespace_free_source():
    with pytest.raises(ValueError, match="source"):
        ReferenceMeasurement(building_id="a", reference_height_m=30.0, source="   ")


@pytest.mark.parametrize("bad_height", [0.0, -5.0, float("nan"), float("inf")])
def test_a_reference_height_must_be_finite_and_positive(bad_height):
    with pytest.raises(ValueError, match="finite"):
        ReferenceMeasurement(building_id="a", reference_height_m=bad_height, source="survey")


def test_a_reference_carries_its_provenance():
    measurement = ReferenceMeasurement(
        building_id="a",
        reference_height_m=31.4,
        source="total station",
        measured_by="survey team",
        measured_on="2026-03-14",
        note="north face",
    )
    record = measurement.to_dict()
    assert record["source"] == "total station"
    assert record["measured_by"] == "survey team"
    assert record["measured_on"] == "2026-03-14"
    assert record["note"] == "north face"


def test_from_records_rejects_a_row_missing_its_source():
    with pytest.raises(ValueError, match="missing"):
        ReferenceSet.from_records(
            [{"building_id": "a", "reference_height_m": 30.0}], kind=ReferenceKind.SYNTHETIC
        )


def test_duplicate_reference_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReferenceSet.from_records(
            [
                {"building_id": "a", "reference_height_m": 30.0, "source": "s"},
                {"building_id": "a", "reference_height_m": 31.0, "source": "s"},
            ],
            kind=ReferenceKind.SYNTHETIC,
        )


# ---------------------------------------------------------------------------
# Synthetic can never be mistaken for real
# ---------------------------------------------------------------------------


def test_reference_kinds_declare_whether_they_are_real_world():
    assert ReferenceKind.SYNTHETIC.is_real_world is False
    assert ReferenceKind.MANUAL_REAL.is_real_world is True


def test_metrics_carry_the_reference_kind(estimates, synthetic_references):
    metrics = evaluate(estimates, synthetic_references).metrics()
    assert metrics.kind is ReferenceKind.SYNTHETIC
    assert metrics.is_real_world is False


def test_synthetic_metrics_say_so_loudly(estimates, synthetic_references):
    description = evaluate(estimates, synthetic_references).metrics().describe()
    assert "SYNTHETIC" in description
    assert "NOT a real-world accuracy figure" in description


def test_the_summary_states_that_real_accuracy_is_unmeasured(estimates, synthetic_references):
    summary = evaluate(estimates, synthetic_references).summary()
    assert NOT_MEASURED in summary
    assert "real-world accuracy" in summary


def test_a_csv_without_an_explicit_kind_is_refused(tmp_path):
    """A CSV carries no kind, and the difference must never be guessed."""
    path = tmp_path / "refs.csv"
    path.write_text(
        "building_id,reference_height_m,source\ntower_a,30.0,tape measure\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="kind"):
        load_reference_set(path)


def test_a_json_without_a_kind_is_refused(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text(
        json.dumps(
            {
                "measurements": [
                    {"building_id": "a", "reference_height_m": 30.0, "source": "survey"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kind"):
        load_reference_set(path)


# ---------------------------------------------------------------------------
# Loading externally supplied references
# ---------------------------------------------------------------------------


def test_load_a_json_reference_set(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text(
        json.dumps(
            {
                "kind": "manual_real",
                "description": "rooftop GPS survey",
                "measurements": [
                    {
                        "building_id": "tower_a",
                        "reference_height_m": 31.4,
                        "source": "total station, 2026-03-14",
                        "measured_by": "survey team",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    references = load_reference_set(path)
    assert references.kind is ReferenceKind.MANUAL_REAL
    assert references.description == "rooftop GPS survey"
    assert len(references) == 1
    assert references.get("tower_a").reference_height_m == pytest.approx(31.4)


def test_load_a_csv_reference_set_with_an_explicit_kind(tmp_path):
    path = tmp_path / "refs.csv"
    path.write_text(
        "building_id,reference_height_m,source\n"
        "tower_a,31.4,laser rangefinder\n"
        "block_b,11.8,laser rangefinder\n",
        encoding="utf-8",
    )
    references = load_reference_set(path, kind=ReferenceKind.MANUAL_REAL)
    assert len(references) == 2
    assert references.kind is ReferenceKind.MANUAL_REAL
    assert references.get("block_b").source == "laser rangefinder"


def test_an_unknown_file_type_is_refused(tmp_path):
    path = tmp_path / "refs.txt"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported reference file"):
        load_reference_set(path)


def test_an_unknown_building_id_simply_does_not_match(estimates):
    """A reference for a building nobody predicted is not an error, and not a match."""
    references = ReferenceSet.from_records(
        [{"building_id": "not_in_scene", "reference_height_m": 20.0, "source": "s"}],
        kind=ReferenceKind.MANUAL_REAL,
    )
    report = evaluate(estimates, references)
    assert len(report) == len(estimates)
    assert report.matched == ()
    assert report.metrics() is None


# ---------------------------------------------------------------------------
# With no references, nothing is invented
# ---------------------------------------------------------------------------


def test_evaluate_with_no_references_still_reports_every_prediction(estimates):
    report = evaluate(estimates)
    assert len(report) == len(estimates)
    assert all(record.predicted_height_m is not None for record in report)


def test_no_references_means_no_metrics(estimates):
    """None is the point: there is no MAE to report, so none is manufactured."""
    report = evaluate(estimates)
    assert report.has_references is False
    assert report.metrics() is None
    assert report.reference_kind is None


def test_no_references_means_empty_reference_fields(estimates):
    report = evaluate(estimates)
    for record in report:
        assert record.reference_height_m is None
        assert record.absolute_error_m is None
        assert record.reference_kind is None
        assert record.has_reference is False


def test_missing_values_render_as_not_measured_not_as_zero(estimates):
    """A human-facing report says 'not yet measured', never 0.0."""
    record = evaluate(estimates).records[0].to_dict(render_missing=True)
    assert record["reference_height_m"] == NOT_MEASURED
    assert record["absolute_error_m"] == NOT_MEASURED
    assert record["reference_kind"] == NOT_MEASURED


def test_machine_facing_output_keeps_none(estimates):
    record = evaluate(estimates).records[0].to_dict()
    assert record["reference_height_m"] is None
    assert record["absolute_error_m"] is None


def test_the_table_marks_absent_values_rather_than_filling_them(estimates):
    table = evaluate(estimates).format_table()
    assert "--" in table
    assert NOT_MEASURED in table


def test_a_summary_without_references_claims_nothing(estimates):
    summary = evaluate(estimates).summary()
    assert NOT_MEASURED in summary
    for forbidden in ("MAE  ", "RMSE  ", "Pearson"):
        assert forbidden not in summary


# ---------------------------------------------------------------------------
# Metrics, where references do exist
# ---------------------------------------------------------------------------


def test_metrics_are_computed_over_matched_records_only(estimates, synthetic_references):
    partial = ReferenceSet(
        kind=ReferenceKind.SYNTHETIC,
        measurements=synthetic_references.measurements[:2],
    )
    report = evaluate(estimates, partial)
    assert len(report) == 4
    assert len(report.matched) == 2
    assert report.metrics().n == 2


def test_metric_arithmetic_is_correct():
    """Hand-checkable numbers, so the statistics cannot drift silently."""
    report = EvaluationReport(
        records=tuple(
            _record(building_id, predicted, reference)
            for building_id, predicted, reference in (
                ("a", 11.0, 10.0),  # error +1
                ("b", 18.0, 20.0),  # error -2
                ("c", 30.0, 30.0),  # error  0
            )
        ),
        reference_kind=ReferenceKind.SYNTHETIC,
    )
    metrics = report.metrics()
    assert metrics.n == 3
    assert metrics.mae_m == pytest.approx((1 + 2 + 0) / 3)
    assert metrics.rmse_m == pytest.approx(np.sqrt((1 + 4 + 0) / 3))
    assert metrics.bias_m == pytest.approx((1 - 2 + 0) / 3)
    assert metrics.max_absolute_error_m == pytest.approx(2.0)


def test_bias_is_signed_and_mae_is_not():
    report = EvaluationReport(
        records=(_record("a", 8.0, 10.0), _record("b", 18.0, 20.0)),
        reference_kind=ReferenceKind.SYNTHETIC,
    )
    metrics = report.metrics()
    assert metrics.bias_m == pytest.approx(-2.0)  # consistently under-predicting
    assert metrics.mae_m == pytest.approx(2.0)


def test_pearson_r_is_none_when_undefined():
    """One point, or no spread, has no correlation -- and none is invented."""
    single = EvaluationReport(
        records=(_record("a", 10.0, 10.0),), reference_kind=ReferenceKind.SYNTHETIC
    )
    assert single.metrics().pearson_r is None

    flat = EvaluationReport(
        records=(_record("a", 10.0, 10.0), _record("b", 10.0, 10.0)),
        reference_kind=ReferenceKind.SYNTHETIC,
    )
    assert flat.metrics().pearson_r is None


def test_perfect_predictions_score_zero_error(estimates, synthetic_references):
    """The analytic scene is exactly representable, so the errors are ~0."""
    metrics = evaluate(estimates, synthetic_references).metrics()
    assert metrics.n == 4
    assert metrics.mae_m == pytest.approx(0.0, abs=1e-9)
    assert metrics.rmse_m == pytest.approx(0.0, abs=1e-9)
    assert metrics.bias_m == pytest.approx(0.0, abs=1e-9)


def test_absolute_error_is_recorded_per_building(estimates, synthetic_references):
    report = evaluate(estimates, synthetic_references)
    for record in report.matched:
        assert record.absolute_error_m == pytest.approx(
            abs(record.predicted_height_m - record.reference_height_m)
        )


def test_metrics_to_dict_is_tagged(estimates, synthetic_references):
    record = evaluate(estimates, synthetic_references).metrics().to_dict()
    assert record["reference_kind"] == "synthetic"
    assert record["is_real_world"] is False
    assert record["n"] == 4


def test_report_to_dict_round_trips_through_json(estimates, synthetic_references):
    """The report must be serialisable, since that is how results leave here."""
    record = json.loads(json.dumps(evaluate(estimates, synthetic_references).to_dict()))
    assert record["reference_kind"] == "synthetic"
    assert record["has_references"] is True
    assert len(record["records"]) == 4
    assert record["metrics"]["n"] == 4


def test_synthetic_references_leave_real_world_accuracy_empty(estimates, synthetic_references):
    """Metrics exist, but the real-world slot stays empty: they are not the same.

    A synthetic reference produces a number under ``metrics``; it must never
    also populate ``real_world_accuracy``, which is the field a reader would
    quote.
    """
    record = evaluate(estimates, synthetic_references).to_dict()
    assert record["metrics"] is not None
    assert record["real_world_accuracy"] is None

    rendered = evaluate(estimates, synthetic_references).to_dict(render_missing=True)
    assert rendered["real_world_accuracy"] == NOT_MEASURED


def test_real_references_do_populate_real_world_accuracy(estimates):
    """The same arithmetic, tagged MANUAL_REAL, is allowed to be quoted.

    Nothing in the repository supplies such a set; this one is constructed by
    the test to prove the flag is wired, not to report an accuracy.
    """
    references = ReferenceSet.from_records(
        [{"building_id": "tower_a", "reference_height_m": 30.0, "source": "test-only, invented"}],
        kind=ReferenceKind.MANUAL_REAL,
    )
    record = evaluate(estimates, references).to_dict()
    assert record["real_world_accuracy"] is not None
    assert record["real_world_accuracy"]["is_real_world"] is True


# ---------------------------------------------------------------------------
# Failed predictions are carried, not dropped or defaulted
# ---------------------------------------------------------------------------


def test_a_failed_prediction_appears_with_its_reason_and_no_height():
    shape = (120, 120)
    footprint = BuildingFootprint.from_bbox(
        "unlit", shape=shape, row_min=40, row_max=80, col_min=40, col_max=80
    )
    failed = estimate_heights_from_mask(
        [footprint],
        np.zeros(shape, dtype=bool),
        gsd_m=0.5,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
    )
    references = ReferenceSet.from_records(
        [{"building_id": "unlit", "reference_height_m": 20.0, "source": "survey"}],
        kind=ReferenceKind.MANUAL_REAL,
    )
    report = evaluate(failed, references)
    record = report.records[0]
    assert record.predicted_height_m is None
    assert record.absolute_error_m is None
    assert record.confidence_flag == Confidence.FAILED.value
    assert record.failure_reason
    # It had a reference, but no prediction, so it is not scorable.
    assert report.matched == ()
    assert report.metrics() is None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _record(building_id: str, predicted: float, reference: float):
    from depthwizard.validation.evaluation import EvaluationRecord

    return EvaluationRecord(
        building_id=building_id,
        shadow_length_px=predicted * 2.0,
        gsd_m=0.5,
        shadow_length_m=predicted,
        sun_elevation_deg=45.0,
        predicted_height_m=predicted,
        reference_height_m=reference,
        absolute_error_m=abs(predicted - reference),
        uncertainty_m=None,
        confidence_flag="nominal",
        reference_kind=ReferenceKind.SYNTHETIC,
        reference_source="test",
    )


def test_accuracy_metrics_is_a_value_object():
    metrics = AccuracyMetrics(
        kind=ReferenceKind.MANUAL_REAL,
        n=3,
        mae_m=1.0,
        rmse_m=1.2,
        bias_m=-0.3,
        max_absolute_error_m=2.0,
        pearson_r=0.98,
    )
    assert metrics.is_real_world is True
    assert "REAL-WORLD" in metrics.describe()
