"""Validation: scoring predicted heights against reference data.

Phase 2 implements the **interface** through which reference heights arrive,
and the reporting and plotting that sit on top of it:

* :mod:`depthwizard.validation.evaluation` - reference measurements, per-building
  evaluation records, and error statistics *only where references exist*
* :mod:`depthwizard.validation.plots` - predicted vs reference scatter plot,
  drawn *only where references exist*

**No reference measurements ship with this repository, and none are generated.**
A reference height can only be constructed with a stated ``source``, and every
set is tagged ``SYNTHETIC`` or ``MANUAL_REAL`` so a fixture number can never be
reported as a real-world one. With nothing supplied, every reference-derived
value renders as :data:`~depthwizard.validation.evaluation.NOT_MEASURED`.

**Real-world accuracy for DepthWizard has not been measured.** No MAE, RMSE or
correlation against real buildings is claimed anywhere in this repository,
because no real reference data has been supplied to it.
"""

from depthwizard.validation.evaluation import (
    NOT_MEASURED,
    AccuracyMetrics,
    EvaluationRecord,
    EvaluationReport,
    ReferenceKind,
    ReferenceMeasurement,
    ReferenceSet,
    evaluate,
    load_reference_set,
)
from depthwizard.validation.plots import plot_predicted_vs_reference

__all__ = [
    "NOT_MEASURED",
    "ReferenceKind",
    "ReferenceMeasurement",
    "ReferenceSet",
    "AccuracyMetrics",
    "EvaluationRecord",
    "EvaluationReport",
    "evaluate",
    "load_reference_set",
    "plot_predicted_vs_reference",
]
