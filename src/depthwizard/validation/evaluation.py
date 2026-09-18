"""Evaluating predicted heights against **externally supplied** references.

This module is the interface through which manually measured real buildings
enter DepthWizard. It does not contain any reference measurements, and it will
not make any up.

The no-fabrication rule, in code
--------------------------------
Three mechanisms enforce it:

1. A reference height can only arrive inside a :class:`ReferenceMeasurement`,
   which **requires** a ``source`` string saying where the number came from. A
   height with no provenance cannot be constructed.
2. Every :class:`ReferenceSet` is tagged with a :class:`ReferenceKind`.
   ``SYNTHETIC`` references come from the fixture generator, where the heights
   were chosen rather than measured; ``MANUAL_REAL`` references come from
   someone measuring a real building. Accuracy figures are always reported with
   that tag attached, so a synthetic number can never be read as a real-world
   one.
3. With no references supplied, :meth:`EvaluationReport.metrics` returns None
   and every reference-derived field renders as :data:`NOT_MEASURED`. There is
   no code path that produces an MAE, an RMSE or a correlation out of
   predictions alone.

A record per building
---------------------
:class:`EvaluationRecord` carries exactly the fields the Phase 2 acceptance
criteria ask for::

    building_id, shadow_length_px, gsd_m, shadow_length_m, sun_elevation_deg,
    predicted_height_m, reference_height_m, absolute_error_m, uncertainty_m,
    confidence_flag

Supplying references
--------------------
From code::

    references = ReferenceSet(
        kind=ReferenceKind.MANUAL_REAL,
        description="rooftop GPS survey, 2026-03",
        measurements=[
            ReferenceMeasurement("tower_a", 31.4, source="total station, 2026-03-14"),
        ],
    )
    report = evaluate(result.estimates, references)

From a file -- :func:`load_reference_set` reads JSON or CSV::

    building_id,reference_height_m,source,note
    tower_a,31.4,"total station, 2026-03-14",north face
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from depthwizard.logging_setup import get_logger
from depthwizard.physics.height import Confidence

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
]

log = get_logger(__name__)

#: Rendered wherever a reference-derived value does not exist yet. This exact
#: string is what a report prints instead of a number, and it is the only
#: honest thing to print.
NOT_MEASURED = "not yet measured"


class ReferenceKind(str, Enum):
    """Where a set of reference heights came from.

    ``SYNTHETIC``
        The heights were *specified* when generating a fixture, not measured.
        Errors against them test the implementation's geometry, and say nothing
        about real-world accuracy.
    ``MANUAL_REAL``
        Someone measured a real building and supplied the number. Only these
        support a claim about real-world accuracy.
    """

    SYNTHETIC = "synthetic"
    MANUAL_REAL = "manual_real"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value

    @property
    def is_real_world(self) -> bool:
        return self is ReferenceKind.MANUAL_REAL


@dataclass(frozen=True)
class ReferenceMeasurement:
    """One externally supplied reference height for one building.

    Args:
        building_id: Must match the ``building_id`` of the footprint that was
            measured.
        reference_height_m: The reference height in metres (> 0, finite).
        source: **Required.** Where this number came from -- an instrument and
            a date, a survey, a fixture's ground truth sidecar. A reference with
            no stated origin is not admissible.
        measured_by: Optional name or role of whoever measured it.
        measured_on: Optional ISO date.
        note: Optional free text, e.g. which face was measured.
    """

    building_id: str
    reference_height_m: float
    source: str
    measured_by: str | None = None
    measured_on: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "building_id", str(self.building_id))
        height = float(self.reference_height_m)
        if not math.isfinite(height) or height <= 0.0:
            raise ValueError(
                f"reference_height_m for {self.building_id!r} must be finite and > 0, "
                f"got {self.reference_height_m}"
            )
        object.__setattr__(self, "reference_height_m", height)
        if not str(self.source).strip():
            raise ValueError(
                f"reference measurement for {self.building_id!r} needs a non-empty "
                "source; an unattributed height is not admissible as a reference"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "reference_height_m": self.reference_height_m,
            "source": self.source,
            "measured_by": self.measured_by,
            "measured_on": self.measured_on,
            "note": self.note,
        }


@dataclass(frozen=True)
class ReferenceSet:
    """A tagged collection of reference measurements.

    Args:
        kind: :class:`ReferenceKind`. Carried onto every metric so a synthetic
            figure can never be mistaken for a real-world one.
        measurements: The references themselves.
        description: Free text describing the campaign or fixture.
    """

    kind: ReferenceKind
    measurements: tuple[ReferenceMeasurement, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ReferenceKind(self.kind))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        ids = [m.building_id for m in self.measurements]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate reference building_id(s): {duplicates}")

    def __len__(self) -> int:
        return len(self.measurements)

    def __bool__(self) -> bool:
        return bool(self.measurements)

    def get(self, building_id: str) -> ReferenceMeasurement | None:
        for measurement in self.measurements:
            if measurement.building_id == building_id:
                return measurement
        return None

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        kind: ReferenceKind,
        description: str = "",
    ) -> "ReferenceSet":
        """Build from dict-like rows with at least ``building_id``,
        ``reference_height_m`` and ``source``."""
        measurements = []
        for index, row in enumerate(records):
            missing = [
                key for key in ("building_id", "reference_height_m", "source") if not row.get(key)
            ]
            if missing:
                raise ValueError(f"reference record {index} is missing {missing}")
            measurements.append(
                ReferenceMeasurement(
                    building_id=str(row["building_id"]),
                    reference_height_m=float(row["reference_height_m"]),
                    source=str(row["source"]),
                    measured_by=(str(row["measured_by"]) if row.get("measured_by") else None),
                    measured_on=(str(row["measured_on"]) if row.get("measured_on") else None),
                    note=(str(row["note"]) if row.get("note") else None),
                )
            )
        return cls(kind=kind, measurements=tuple(measurements), description=description)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "description": self.description,
            "measurements": [m.to_dict() for m in self.measurements],
        }


def load_reference_set(
    path: str | Path,
    *,
    kind: ReferenceKind | None = None,
    description: str | None = None,
) -> ReferenceSet:
    """Load externally supplied reference heights from JSON or CSV.

    JSON shape::

        {
          "kind": "manual_real",
          "description": "rooftop GPS survey, 2026-03",
          "measurements": [
            {"building_id": "tower_a", "reference_height_m": 31.4,
             "source": "total station, 2026-03-14"}
          ]
        }

    CSV shape -- a header row with at least ``building_id``,
    ``reference_height_m`` and ``source``. A CSV carries no ``kind``, so one
    must be passed explicitly; that is deliberate, since the difference between
    a synthetic and a real reference is exactly the thing that must never be
    guessed.

    Args:
        path: ``.json`` or ``.csv`` file.
        kind: Overrides (JSON) or supplies (CSV) the reference kind.
        description: Overrides the file's description.

    Raises:
        ValueError: for an unknown suffix, or a CSV with no ``kind`` supplied.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        resolved_kind = kind or (ReferenceKind(payload["kind"]) if "kind" in payload else None)
        if resolved_kind is None:
            raise ValueError(
                f"{path} has no 'kind' field; pass kind= explicitly so a synthetic "
                "reference is never reported as a real-world one"
            )
        return ReferenceSet.from_records(
            payload.get("measurements", []),
            kind=resolved_kind,
            description=(
                description if description is not None else str(payload.get("description", ""))
            ),
        )

    if suffix == ".csv":
        if kind is None:
            raise ValueError(
                f"{path} is a CSV and carries no reference kind; pass kind= explicitly "
                "so a synthetic reference is never reported as a real-world one"
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return ReferenceSet.from_records(
            rows, kind=kind, description=description if description is not None else str(path)
        )

    raise ValueError(f"unsupported reference file {path.name!r}; expected .json or .csv")


# ---------------------------------------------------------------------------
# Records and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccuracyMetrics:
    """Error statistics over the buildings that had a reference **and** a
    prediction.

    Always carries its :class:`ReferenceKind`, because the same arithmetic over
    synthetic references and over surveyed ones means two entirely different
    things.
    """

    kind: ReferenceKind
    n: int
    mae_m: float
    rmse_m: float
    bias_m: float
    max_absolute_error_m: float
    #: Pearson correlation, or None when it is undefined (n < 2, or no spread).
    pearson_r: float | None

    @property
    def is_real_world(self) -> bool:
        return self.kind.is_real_world

    def describe(self) -> str:
        label = (
            "REAL-WORLD (manually measured references)"
            if self.is_real_world
            else "SYNTHETIC fixture geometry -- NOT a real-world accuracy figure"
        )
        correlation = f"{self.pearson_r:.6f}" if self.pearson_r is not None else NOT_MEASURED
        return (
            f"{label}\n"
            f"  n                  : {self.n}\n"
            f"  MAE                : {self.mae_m:.6f} m\n"
            f"  RMSE               : {self.rmse_m:.6f} m\n"
            f"  bias (pred - ref)  : {self.bias_m:+.6f} m\n"
            f"  max |error|        : {self.max_absolute_error_m:.6f} m\n"
            f"  Pearson r          : {correlation}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_kind": self.kind.value,
            "is_real_world": self.is_real_world,
            "n": self.n,
            "mae_m": self.mae_m,
            "rmse_m": self.rmse_m,
            "bias_m": self.bias_m,
            "max_absolute_error_m": self.max_absolute_error_m,
            "pearson_r": self.pearson_r,
        }


@dataclass(frozen=True)
class EvaluationRecord:
    """One building's prediction, next to its reference if there is one."""

    building_id: str
    shadow_length_px: float | None
    gsd_m: float
    shadow_length_m: float | None
    sun_elevation_deg: float
    predicted_height_m: float | None
    reference_height_m: float | None
    absolute_error_m: float | None
    uncertainty_m: float | None
    confidence_flag: str
    reference_kind: ReferenceKind | None = None
    reference_source: str | None = None
    failure_reason: str | None = None
    reduced_confidence_reasons: tuple[str, ...] = ()

    @property
    def has_reference(self) -> bool:
        return self.reference_height_m is not None

    def to_dict(self, *, render_missing: bool = False) -> dict[str, Any]:
        """Flatten to a dict.

        Args:
            render_missing: When True, every value that does not exist yet is
                replaced by :data:`NOT_MEASURED` rather than None -- the shape a
                human-facing report wants. Machine-facing callers leave it False
                and get None.
        """
        record: dict[str, Any] = {
            "building_id": self.building_id,
            "shadow_length_px": self.shadow_length_px,
            "gsd_m": self.gsd_m,
            "shadow_length_m": self.shadow_length_m,
            "sun_elevation_deg": self.sun_elevation_deg,
            "predicted_height_m": self.predicted_height_m,
            "reference_height_m": self.reference_height_m,
            "absolute_error_m": self.absolute_error_m,
            "uncertainty_m": self.uncertainty_m,
            "confidence_flag": self.confidence_flag,
            "reference_kind": self.reference_kind.value if self.reference_kind else None,
            "reference_source": self.reference_source,
            "measurement_failure_reason": self.failure_reason,
            "reduced_confidence_reasons": list(self.reduced_confidence_reasons),
        }
        if render_missing:
            for key in (
                "reference_height_m",
                "absolute_error_m",
                "reference_kind",
                "reference_source",
            ):
                if record[key] is None:
                    record[key] = NOT_MEASURED
            if record["uncertainty_m"] is None:
                record["uncertainty_m"] = NOT_MEASURED
        return record


@dataclass(frozen=True)
class EvaluationReport:
    """Per-building records, plus metrics **only** where references exist."""

    records: tuple[EvaluationRecord, ...]
    reference_kind: ReferenceKind | None = None
    reference_description: str = ""

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def matched(self) -> tuple[EvaluationRecord, ...]:
        """Records with both a reference and a prediction -- the scorable ones."""
        return tuple(
            r for r in self.records if r.reference_height_m is not None and r.predicted_height_m is not None
        )

    @property
    def has_references(self) -> bool:
        return bool(self.matched)

    def metrics(self) -> AccuracyMetrics | None:
        """Error statistics, or **None** when no reference data was supplied.

        None is the whole point: with nothing to compare against there is no
        MAE, no RMSE and no correlation, and this method will not manufacture
        one. Callers render :data:`NOT_MEASURED` instead.
        """
        matched = self.matched
        if not matched or self.reference_kind is None:
            return None

        predicted = np.array([r.predicted_height_m for r in matched], dtype=np.float64)
        reference = np.array([r.reference_height_m for r in matched], dtype=np.float64)
        error = predicted - reference

        correlation: float | None = None
        if predicted.size >= 2 and predicted.std() > 0.0 and reference.std() > 0.0:
            correlation = float(np.corrcoef(predicted, reference)[0, 1])

        return AccuracyMetrics(
            kind=self.reference_kind,
            n=int(predicted.size),
            mae_m=float(np.abs(error).mean()),
            rmse_m=float(np.sqrt((error**2).mean())),
            bias_m=float(error.mean()),
            max_absolute_error_m=float(np.abs(error).max()),
            pearson_r=correlation,
        )

    # -- rendering ----------------------------------------------------------

    def format_table(self) -> str:
        """A fixed-width table of every record, missing values spelled out."""
        header = (
            f"{'building_id':<14}{'L_px':>10}{'GSD_m':>8}{'L_m':>10}{'sun_el':>8}"
            f"{'pred_h_m':>11}{'ref_h_m':>11}{'abs_err_m':>11}{'unc_m':>9}  confidence"
        )
        lines = [header, "-" * len(header)]

        def number(value: float | None, width: int, places: int = 3) -> str:
            return f"{value:>{width}.{places}f}" if value is not None else f"{'--':>{width}}"

        for record in self.records:
            lines.append(
                f"{record.building_id:<14}"
                f"{number(record.shadow_length_px, 10)}"
                f"{number(record.gsd_m, 8)}"
                f"{number(record.shadow_length_m, 10)}"
                f"{number(record.sun_elevation_deg, 8, 2)}"
                f"{number(record.predicted_height_m, 11)}"
                f"{number(record.reference_height_m, 11)}"
                f"{number(record.absolute_error_m, 11)}"
                f"{number(record.uncertainty_m, 9)}"
                f"  {record.confidence_flag}"
            )
        lines.append("")
        lines.append("'--' means the value does not exist: " + NOT_MEASURED)
        return "\n".join(lines)

    def summary(self) -> str:
        """The table, followed by metrics or an explicit statement of absence."""
        parts = [self.format_table(), ""]
        metrics = self.metrics()
        if metrics is None:
            parts.append(
                "accuracy vs reference heights: " + NOT_MEASURED + "\n"
                "  No reference measurements were supplied, so no MAE, RMSE or\n"
                "  correlation is reported. None is computed and none is implied."
            )
        else:
            label = self.reference_description or "(no description supplied)"
            parts.append(f"reference set: {label}")
            parts.append(metrics.describe())
            if not metrics.is_real_world:
                parts.append(
                    "\nreal-world accuracy: " + NOT_MEASURED + "\n"
                    "  The figures above are against SYNTHETIC references and measure\n"
                    "  the implementation's geometry, not performance on real imagery."
                )
        return "\n".join(parts)

    def to_dict(self, *, render_missing: bool = False) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "reference_kind": self.reference_kind.value if self.reference_kind else None,
            "reference_description": self.reference_description,
            "has_references": self.has_references,
            "records": [r.to_dict(render_missing=render_missing) for r in self.records],
            "metrics": metrics.to_dict() if metrics else (NOT_MEASURED if render_missing else None),
            "real_world_accuracy": (
                metrics.to_dict()
                if metrics and metrics.is_real_world
                else (NOT_MEASURED if render_missing else None)
            ),
        }


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------


def evaluate(
    estimates: Sequence[Any],
    references: ReferenceSet | None = None,
) -> EvaluationReport:
    """Pair height estimates with references and build an evaluation report.

    Args:
        estimates: :class:`~depthwizard.shadows.pipeline.BuildingHeightEstimate`
            objects, as returned by the Phase 2 pipeline.
        references: Externally supplied reference heights, or None. **None is a
            valid, expected state**: it means nobody has measured these
            buildings yet, and the report says so rather than inventing values.

    Returns:
        An :class:`EvaluationReport`. Buildings with no reference still appear,
        with their prediction and their reference fields empty.
    """
    records: list[EvaluationRecord] = []
    for estimate in estimates:
        reference = references.get(estimate.building_id) if references else None
        predicted = estimate.height_m
        absolute_error = (
            abs(predicted - reference.reference_height_m)
            if (reference is not None and predicted is not None)
            else None
        )
        confidence = estimate.confidence
        records.append(
            EvaluationRecord(
                building_id=estimate.building_id,
                shadow_length_px=estimate.shadow_length_px,
                gsd_m=estimate.measurement.gsd_m,
                shadow_length_m=estimate.shadow_length_m,
                sun_elevation_deg=estimate.measurement.sun_elevation_deg,
                predicted_height_m=predicted,
                reference_height_m=reference.reference_height_m if reference else None,
                absolute_error_m=absolute_error,
                uncertainty_m=estimate.height_uncertainty_m,
                confidence_flag=(
                    confidence.value if isinstance(confidence, Confidence) else str(confidence)
                ),
                reference_kind=references.kind if (references and reference) else None,
                reference_source=reference.source if reference else None,
                failure_reason=estimate.failure_reason,
                reduced_confidence_reasons=tuple(estimate.reduced_confidence_reasons),
            )
        )

    report = EvaluationReport(
        records=tuple(records),
        reference_kind=references.kind if references else None,
        reference_description=references.description if references else "",
    )
    log.info(
        "built evaluation report",
        extra={
            "buildings": len(report),
            "with_reference": len(report.matched),
            "reference_kind": report.reference_kind.value if report.reference_kind else None,
        },
    )
    return report
