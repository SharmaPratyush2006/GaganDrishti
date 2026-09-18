"""Visualisation: predicted height vs reference height.

One plot, with one rule attached to it: **it is only drawn when reference
measurements actually exist.** Called on a report with no references,
:func:`plot_predicted_vs_reference` returns None and logs
``"not yet measured"``. An empty scatter plot with an identity line through it
looks like a validation result, which is exactly the impression this project
must not give before anything has been validated.

Layout:

* x-axis = **reference** height in metres (the independent, known quantity)
* y-axis = **predicted** height in metres
* a dashed 1:1 identity line, so vertical distance from it *is* the error
* error bars where an uncertainty was supplied, omitted where it was not
* the reference kind printed in the title, so a synthetic plot is never
  mistaken for a real-world one

Rendered through the Agg canvas directly rather than ``pyplot``, so it is
headless-safe and holds no global figure state.
"""

from __future__ import annotations

from pathlib import Path

from depthwizard.logging_setup import get_logger
from depthwizard.validation.evaluation import NOT_MEASURED, EvaluationReport

__all__ = ["plot_predicted_vs_reference"]

log = get_logger(__name__)


def plot_predicted_vs_reference(
    report: EvaluationReport,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 150,
    annotate_ids: bool = True,
) -> Path | None:
    """Scatter predicted height against reference height.

    Args:
        report: An :class:`~depthwizard.validation.evaluation.EvaluationReport`.
        out_path: Where to write the PNG.
        title: Overrides the generated title.
        dpi: Output resolution.
        annotate_ids: Label each point with its ``building_id``.

    Returns:
        The path written, or **None** when the report has no reference
        measurements -- in which case nothing is drawn and nothing is written.
    """
    out_path = Path(out_path)
    matched = report.matched
    if not matched:
        log.warning(
            "no scatter plot drawn: predicted vs reference height is " + NOT_MEASURED,
            extra={"out_path": str(out_path), "buildings": len(report)},
        )
        return None

    # Imported here so that importing the package does not pull in matplotlib.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    reference = [r.reference_height_m for r in matched]
    predicted = [r.predicted_height_m for r in matched]
    uncertainty: list[float] = [r.uncertainty_m or 0.0 for r in matched]
    has_uncertainty = any(r.uncertainty_m is not None for r in matched)

    figure = Figure(figsize=(6.0, 6.0), dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)

    low = min(min(reference), min(predicted))
    high = max(max(reference), max(predicted))
    span = high - low if high > low else max(high, 1.0)
    pad = 0.1 * span
    limits = (max(0.0, low - pad), high + pad)

    axes.plot(limits, limits, linestyle="--", linewidth=1.0, color="0.4", label="1:1 (perfect)")
    if has_uncertainty:
        axes.errorbar(
            reference,
            predicted,
            yerr=uncertainty,
            fmt="o",
            markersize=6,
            capsize=3,
            linewidth=1.0,
            label="predicted (with supplied dh)",
        )
    else:
        axes.plot(reference, predicted, "o", markersize=6, label="predicted")

    if annotate_ids:
        for record in matched:
            axes.annotate(
                record.building_id,
                (record.reference_height_m, record.predicted_height_m),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    kind = report.reference_kind
    kind_label = kind.value if kind else "unknown"
    if title is None:
        title = f"Predicted vs reference height ({kind_label} references)"
        if kind is not None and not kind.is_real_world:
            title += "\nSYNTHETIC - not a real-world accuracy result"

    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("reference height (m)")
    axes.set_ylabel("predicted height (m)")
    axes.set_title(title, fontsize=10)
    axes.grid(True, linewidth=0.4, alpha=0.5)
    axes.legend(loc="lower right", fontsize=8)

    metrics = report.metrics()
    if metrics is not None:
        axes.text(
            0.04,
            0.96,
            f"n = {metrics.n}\nMAE = {metrics.mae_m:.4f} m\nRMSE = {metrics.rmse_m:.4f} m",
            transform=axes.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "linewidth": 0.4},
        )

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_figure(out_path, dpi=dpi)

    log.info(
        "wrote predicted vs reference scatter plot",
        extra={"out_path": str(out_path), "points": len(matched), "reference_kind": kind_label},
    )
    return out_path
