"""Validation: scoring predicted heights against reference data.

Phase 2 implemented the **interface** through which reference heights arrive,
and the reporting and plotting that sit on top of it:

* :mod:`depthwizard.validation.evaluation` - reference measurements, per-building
  evaluation records, and error statistics *only where references exist*
* :mod:`depthwizard.validation.plots` - predicted vs reference scatter plot,
  drawn *only where references exist*

Phase 5 adds the dense validation harness (``python -m depthwizard.validation.run``):

* :mod:`depthwizard.validation.masks` - explicit validity mask with a per-reason
  exclusion count; OVERALL / TERRAIN / BUILDING category masks from data labels
* :mod:`depthwizard.validation.metrics` - MAE, RMSE, bias, Pearson r and
  delta-threshold accuracy, streamable, each with units and a pixel count
* :mod:`depthwizard.validation.spatial_split` - a declared geographic split,
  audited against the split recorded inside the checkpoint
* :mod:`depthwizard.validation.confidence` - the rule-based confidence proxy
  (shadow occlusion, water, 25-45 deg sun band); flags, not probabilities
* :mod:`depthwizard.validation.error_map` - ``prediction - reference`` raster
  (CRS/transform preserved) and its figure
* :mod:`depthwizard.validation.report` - the Markdown validation report
* :mod:`depthwizard.validation.run` - the one-command harness

**No reference measurements ship with this repository, and none are generated.**
Every reported figure is tagged SYNTHETIC or real-world, and every value that
does not exist renders as :data:`~depthwizard.validation.evaluation.NOT_MEASURED`.
"""

from depthwizard.validation.confidence import ConfidenceResult, ConfidenceState, confidence_proxy
from depthwizard.validation.config import (
    Phase5Config,
    ValidationError,
    ValidationInputError,
    load_phase5_config,
)
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
from depthwizard.validation.masks import CategoryMasks, ValidityMask, build_valid_mask, category_masks_from_labels
from depthwizard.validation.metrics import MetricAccumulator, MetricValue, RegionMetrics, compute_metrics
from depthwizard.validation.plots import plot_predicted_vs_reference
from depthwizard.validation.spatial_split import SplitError, audit_region_split

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
    # Phase 5
    "Phase5Config",
    "load_phase5_config",
    "ValidationError",
    "ValidationInputError",
    "ValidityMask",
    "build_valid_mask",
    "CategoryMasks",
    "category_masks_from_labels",
    "MetricValue",
    "RegionMetrics",
    "MetricAccumulator",
    "compute_metrics",
    "SplitError",
    "audit_region_split",
    "ConfidenceState",
    "ConfidenceResult",
    "confidence_proxy",
]
