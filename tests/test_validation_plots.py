"""The predicted-vs-reference scatter plot.

The plot is a reporting utility, and the contract it has to keep is the same
one the rest of the validation package keeps: it draws what exists and nothing
else. With no reference measurements there is no scatter to draw, so it writes
no file and returns None rather than producing an empty or misleading figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthwizard.validation import (
    NOT_MEASURED,
    EvaluationReport,
    ReferenceKind,
    ReferenceSet,
    evaluate,
    plot_predicted_vs_reference,
)
from depthwizard.validation.evaluation import EvaluationRecord


def _record(building_id, predicted, reference, uncertainty=None):
    return EvaluationRecord(
        building_id=building_id,
        shadow_length_px=(predicted or 0.0) * 2.0,
        gsd_m=0.5,
        shadow_length_m=predicted,
        sun_elevation_deg=45.0,
        predicted_height_m=predicted,
        reference_height_m=reference,
        absolute_error_m=(
            abs(predicted - reference) if predicted is not None and reference is not None else None
        ),
        uncertainty_m=uncertainty,
        confidence_flag="nominal",
        reference_kind=ReferenceKind.SYNTHETIC if reference is not None else None,
        reference_source="test" if reference is not None else None,
    )


def _report(kind=ReferenceKind.SYNTHETIC, uncertainty=None):
    return EvaluationReport(
        records=(
            _record("tower_a", 29.7, 30.0, uncertainty),
            _record("block_b", 12.0, 12.0, uncertainty),
            _record("slab_c", 45.3, 45.0, uncertainty),
            _record("low_d", 5.7, 6.0, uncertainty),
        ),
        reference_kind=kind,
        reference_description="test set",
    )


def _png_header(path) -> bytes:
    return path.read_bytes()[:8]


# ---------------------------------------------------------------------------
# With references: a plot is drawn
# ---------------------------------------------------------------------------


def test_a_plot_is_written_when_references_exist(tmp_path):
    out_path = tmp_path / "scatter.png"
    result = plot_predicted_vs_reference(_report(), out_path)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert _png_header(out_path) == b"\x89PNG\r\n\x1a\n"


def test_the_plot_creates_missing_parent_directories(tmp_path):
    out_path = tmp_path / "nested" / "deeper" / "scatter.png"
    assert plot_predicted_vs_reference(_report(), out_path) == out_path
    assert out_path.exists()


def test_a_string_path_is_accepted(tmp_path):
    out_path = tmp_path / "scatter.png"
    result = plot_predicted_vs_reference(_report(), str(out_path))
    assert result == out_path
    assert out_path.exists()


def test_the_dpi_setting_changes_the_output(tmp_path):
    low = tmp_path / "low.png"
    high = tmp_path / "high.png"
    plot_predicted_vs_reference(_report(), low, dpi=50)
    plot_predicted_vs_reference(_report(), high, dpi=200)
    assert high.stat().st_size > low.stat().st_size


def test_annotations_can_be_turned_off(tmp_path):
    """Labelling every point is unreadable on a dense scene, so it is optional."""
    annotated = tmp_path / "annotated.png"
    plain = tmp_path / "plain.png"
    plot_predicted_vs_reference(_report(), annotated, annotate_ids=True)
    plot_predicted_vs_reference(_report(), plain, annotate_ids=False)
    assert annotated.exists() and plain.exists()
    assert annotated.read_bytes() != plain.read_bytes()


def test_a_custom_title_is_used(tmp_path):
    out_path = tmp_path / "titled.png"
    assert plot_predicted_vs_reference(_report(), out_path, title="Custom") == out_path


def test_uncertainty_is_drawn_as_error_bars_when_supplied(tmp_path):
    with_bars = tmp_path / "bars.png"
    without = tmp_path / "plain.png"
    plot_predicted_vs_reference(_report(uncertainty=0.5), with_bars)
    plot_predicted_vs_reference(_report(uncertainty=None), without)
    assert with_bars.read_bytes() != without.read_bytes()


def test_a_real_reference_set_is_plotted_too(tmp_path):
    out_path = tmp_path / "real.png"
    assert plot_predicted_vs_reference(_report(kind=ReferenceKind.MANUAL_REAL), out_path)


def test_a_single_matched_point_still_plots(tmp_path):
    """One point has no correlation, but it does have a position."""
    report = EvaluationReport(
        records=(_record("only", 30.0, 30.0),), reference_kind=ReferenceKind.SYNTHETIC
    )
    out_path = tmp_path / "single.png"
    assert plot_predicted_vs_reference(report, out_path) == out_path
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Without references: nothing is drawn
# ---------------------------------------------------------------------------


def test_no_references_means_no_file_and_no_plot(tmp_path):
    """The central contract: an empty scatter is worse than no scatter."""
    report = EvaluationReport(
        records=(_record("a", 30.0, None), _record("b", 12.0, None)), reference_kind=None
    )
    out_path = tmp_path / "scatter.png"
    assert plot_predicted_vs_reference(report, out_path) is None
    assert not out_path.exists()


def test_an_empty_report_draws_nothing(tmp_path):
    out_path = tmp_path / "scatter.png"
    assert plot_predicted_vs_reference(EvaluationReport(records=()), out_path) is None
    assert not out_path.exists()


def test_predictions_without_references_draw_nothing(tmp_path):
    """Half a pair is not a point: a prediction alone cannot be plotted."""
    report = EvaluationReport(
        records=(_record("a", 30.0, None),), reference_kind=ReferenceKind.MANUAL_REAL
    )
    out_path = tmp_path / "scatter.png"
    assert plot_predicted_vs_reference(report, out_path) is None
    assert not out_path.exists()


def test_references_without_predictions_draw_nothing(tmp_path):
    """Nor is a failed measurement plottable against its reference."""
    report = EvaluationReport(
        records=(_record("a", None, 30.0),), reference_kind=ReferenceKind.MANUAL_REAL
    )
    out_path = tmp_path / "scatter.png"
    assert plot_predicted_vs_reference(report, out_path) is None
    assert not out_path.exists()


def test_the_skipped_plot_is_reported_as_not_measured(tmp_path, caplog):
    """The absence must be stated, not silent."""
    import logging

    report = EvaluationReport(records=(_record("a", 30.0, None),))
    with caplog.at_level(logging.WARNING):
        assert plot_predicted_vs_reference(report, tmp_path / "scatter.png") is None
    assert NOT_MEASURED in caplog.text


def test_only_matched_records_are_plotted(tmp_path):
    """A mixed report plots the pairs and silently omits the rest, not the reverse."""
    mixed = EvaluationReport(
        records=(
            _record("matched", 30.0, 30.0),
            _record("no_reference", 12.0, None),
            _record("no_prediction", None, 45.0),
        ),
        reference_kind=ReferenceKind.SYNTHETIC,
    )
    out_path = tmp_path / "mixed.png"
    assert plot_predicted_vs_reference(mixed, out_path) == out_path
    assert len(mixed.matched) == 1


# ---------------------------------------------------------------------------
# End to end, from a real pipeline result
# ---------------------------------------------------------------------------


def test_plot_from_an_evaluated_pipeline_result(tmp_path):
    from conftest import build_analytic_shadow_mask
    from depthwizard.shadows import BuildingFootprint, estimate_heights_from_mask

    shape = (200, 200)
    mask = build_analytic_shadow_mask(
        shape,
        row_min=80,
        row_max=120,
        col_min=140,
        col_max=180,
        sun_azimuth_deg=90.0,
        shadow_length_px=60.0,
    )
    footprint = BuildingFootprint.from_bbox(
        "solo", shape=shape, row_min=80, row_max=120, col_min=140, col_max=180
    )
    estimates = estimate_heights_from_mask(
        [footprint],
        mask,
        gsd_m=0.5,
        sun_azimuth_deg=90.0,
        sun_elevation_deg=45.0,
        measurement_uncertainty_px=1.0,
    )
    references = ReferenceSet.from_records(
        [{"building_id": "solo", "reference_height_m": 30.0, "source": "analytic scene spec"}],
        kind=ReferenceKind.SYNTHETIC,
    )
    report = evaluate(estimates, references)
    out_path = tmp_path / "pipeline.png"
    assert plot_predicted_vs_reference(report, out_path) == out_path
    assert out_path.exists()
    assert np.isclose(report.metrics().mae_m, 0.0, atol=1e-9)
