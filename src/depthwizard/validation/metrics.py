"""Phase 5 metrics: MAE, RMSE, Pearson r and delta-threshold accuracy.

Every metric is computed over an **explicit** validity mask (see
:func:`depthwizard.validation.masks.build_valid_mask`), reports how many pixels
it used, and carries its units. A metric that cannot be computed comes back as
a :class:`MetricValue` whose ``value`` is None and whose ``reason`` says why.
It never falls back to 0.0: an unmeasurable metric is not a perfect one.

Definitions (``e = prediction - reference`` over valid pixels)
-------------------------------------------------------------
``MAE``   ``mean(|e|)``, in the data's units.
``RMSE``  ``sqrt(mean(e^2))``, in the data's units.
``bias``  ``mean(e)``, in the data's units (positive = over-prediction).
``r``     Pearson correlation of prediction with reference, dimensionless.
          Not computable with fewer than :data:`MIN_PEARSON_SAMPLES` pixels
          (with two points r is always +/-1 and carries no information), or when
          either side has zero variance (r is 0/0).
``delta < t``  Fraction of pixels with ``max(p / r, r / p) < t``, the standard
          depth-estimation threshold accuracy. Reported as a **fraction** in
          [0, 1]. The ratio is only meaningful for positive quantities, so the
          delta domain is the valid pixels where **both** prediction and
          reference are > 0; the rest are counted as ``delta_excluded_nonpositive``
          and never evaluated. For above-ground height this excludes flat
          ground (reference AGL <= 0) by construction -- which is why the
          terrain delta is usually not computable, and says so.

Streaming
---------
:class:`MetricAccumulator` keeps only additive sufficient statistics, so any
number of tiles can be pooled exactly without holding them in memory. Pearson's
sums are accumulated about a shift (the first batch's means) to avoid the
catastrophic cancellation a raw ``sum(x^2)`` suffers on elevations of hundreds
of metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from depthwizard.validation.evaluation import NOT_MEASURED

__all__ = [
    "DEFAULT_DELTA_THRESHOLDS",
    "MIN_PEARSON_SAMPLES",
    "DIMENSIONLESS",
    "FRACTION",
    "delta_name",
    "MetricValue",
    "RegionMetrics",
    "MetricAccumulator",
    "compute_metrics",
    "mae",
    "rmse",
    "pearson",
    "delta_accuracy",
]

#: delta < 1.25, 1.25^2, 1.25^3.
DEFAULT_DELTA_THRESHOLDS: tuple[float, ...] = (1.25, 1.25**2, 1.25**3)
#: Fewest pixels for which a Pearson r is reported.
MIN_PEARSON_SAMPLES = 3
DIMENSIONLESS = "dimensionless"
FRACTION = "fraction"


def delta_name(threshold: float) -> str:
    """``delta<1.25``, ``delta<1.25^2``, ``delta<1.25^3`` for the standard set."""
    for power in (1, 2, 3):
        if math.isclose(threshold, 1.25**power, rel_tol=1e-12):
            return "delta<1.25" if power == 1 else f"delta<1.25^{power}"
    return f"delta<{threshold:g}"


@dataclass(frozen=True)
class MetricValue:
    """One metric: a value with units and a pixel count, or a stated absence."""

    name: str
    value: float | None
    units: str
    #: Pixels the metric was computed over (for delta: its positive domain).
    n: int
    #: Why ``value`` is None. Always set when it is, never set otherwise.
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None and not self.reason:
            raise ValueError(f"metric {self.name!r} has no value and no reason")
        if self.value is not None:
            if not math.isfinite(self.value):
                raise ValueError(f"metric {self.name!r} is not finite: {self.value}")
            if self.reason is not None:
                raise ValueError(f"metric {self.name!r} has a value and a reason")

    @property
    def available(self) -> bool:
        return self.value is not None

    def render(self, places: int = 4) -> str:
        """Human-readable: the number, or ``not yet measured (reason)``."""
        if self.value is None:
            return f"{NOT_MEASURED} ({self.reason})"
        return f"{self.value:.{places}f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value if self.value is not None else NOT_MEASURED,
            "units": self.units,
            "n": self.n,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RegionMetrics:
    """All metrics for one category (OVERALL / TERRAIN / BUILDING / ...)."""

    category: str
    units: str
    valid_pixels: int
    mae: MetricValue
    rmse: MetricValue
    bias: MetricValue
    pearson_r: MetricValue
    deltas: tuple[MetricValue, ...]
    #: Valid pixels where prediction and reference are both > 0.
    delta_domain_pixels: int
    #: Valid pixels left out of the delta metrics because p <= 0 or r <= 0.
    delta_excluded_nonpositive: int
    #: Set when the category itself could not be measured (e.g. no label).
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "units": self.units,
            "valid_pixels": self.valid_pixels,
            "mae": self.mae.to_dict(),
            "rmse": self.rmse.to_dict(),
            "bias": self.bias.to_dict(),
            "pearson_r": self.pearson_r.to_dict(),
            "deltas": [d.to_dict() for d in self.deltas],
            "delta_domain": "valid pixels with prediction > 0 and reference > 0",
            "delta_domain_pixels": self.delta_domain_pixels,
            "delta_excluded_nonpositive": self.delta_excluded_nonpositive,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def unavailable(
        cls, category: str, units: str, reason: str, delta_thresholds: Sequence[float] = DEFAULT_DELTA_THRESHOLDS
    ) -> "RegionMetrics":
        """A category the data cannot measure at all. Every value is absent."""
        def gone(name: str, metric_units: str) -> MetricValue:
            return MetricValue(name, None, metric_units, 0, reason)

        return cls(
            category=category,
            units=units,
            valid_pixels=0,
            mae=gone("mae", units),
            rmse=gone("rmse", units),
            bias=gone("bias", units),
            pearson_r=gone("pearson_r", DIMENSIONLESS),
            deltas=tuple(gone(delta_name(t), FRACTION) for t in delta_thresholds),
            delta_domain_pixels=0,
            delta_excluded_nonpositive=0,
            unavailable_reason=reason,
        )


@dataclass
class MetricAccumulator:
    """Pooled sufficient statistics for every Phase 5 metric.

    Call :meth:`add` once per array (or per tile), then :meth:`result`.
    """

    delta_thresholds: tuple[float, ...] = DEFAULT_DELTA_THRESHOLDS
    n: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_err: float = 0.0
    # Pearson, about a shift fixed by the first non-empty batch.
    _shift: tuple[float, float] | None = None
    _sp: float = 0.0
    _sr: float = 0.0
    _spp: float = 0.0
    _srr: float = 0.0
    _spr: float = 0.0
    _p_range: list[float] = field(default_factory=lambda: [math.inf, -math.inf])
    _r_range: list[float] = field(default_factory=lambda: [math.inf, -math.inf])
    # Delta.
    delta_domain: int = 0
    delta_excluded_nonpositive: int = 0
    _under: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.delta_thresholds = tuple(float(t) for t in self.delta_thresholds)
        for t in self.delta_thresholds:
            if not (math.isfinite(t) and t > 1.0):
                raise ValueError(f"delta thresholds must be finite and > 1, got {t}")
        if not self._under:
            self._under = [0] * len(self.delta_thresholds)

    def add(self, prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> None:
        """Pool the pixels selected by ``valid``.

        Raises:
            ValueError: on a shape mismatch, or if ``valid`` admits a
                non-finite value -- the mask must exclude those explicitly
                (:func:`~depthwizard.validation.masks.build_valid_mask`), so no
                NaN is ever dropped out of sight here.
        """
        prediction = np.asarray(prediction)
        reference = np.asarray(reference)
        valid = np.asarray(valid, dtype=bool)
        if not (prediction.shape == reference.shape == valid.shape):
            raise ValueError(
                f"shape mismatch: prediction {prediction.shape}, reference {reference.shape}, valid {valid.shape}"
            )
        p = prediction[valid].astype(np.float64)
        r = reference[valid].astype(np.float64)
        if p.size == 0:
            return
        if not (np.isfinite(p).all() and np.isfinite(r).all()):
            raise ValueError("the validity mask admits non-finite values; exclude them with build_valid_mask")

        e = p - r
        self.n += int(p.size)
        self.sum_abs += float(np.abs(e).sum())
        self.sum_sq += float((e * e).sum())
        self.sum_err += float(e.sum())

        if self._shift is None:
            self._shift = (float(p.mean()), float(r.mean()))
        dp, dr = p - self._shift[0], r - self._shift[1]
        self._sp += float(dp.sum())
        self._sr += float(dr.sum())
        self._spp += float((dp * dp).sum())
        self._srr += float((dr * dr).sum())
        self._spr += float((dp * dr).sum())
        self._p_range = [min(self._p_range[0], float(p.min())), max(self._p_range[1], float(p.max()))]
        self._r_range = [min(self._r_range[0], float(r.min())), max(self._r_range[1], float(r.max()))]

        positive = (p > 0.0) & (r > 0.0)
        self.delta_domain += int(positive.sum())
        self.delta_excluded_nonpositive += int((~positive).sum())
        if positive.any():
            pp, rr = p[positive], r[positive]
            ratio = np.maximum(pp / rr, rr / pp)
            for i, t in enumerate(self.delta_thresholds):
                self._under[i] += int((ratio < t).sum())

    # -- finalisation --------------------------------------------------------

    def _pearson(self) -> MetricValue:
        name = "pearson_r"
        if self.n == 0:
            return MetricValue(name, None, DIMENSIONLESS, 0, "no valid pixels")
        if self.n < MIN_PEARSON_SAMPLES:
            return MetricValue(
                name, None, DIMENSIONLESS, self.n,
                f"insufficient samples: {self.n} valid pixel(s), need at least {MIN_PEARSON_SAMPLES}",
            )
        if self._p_range[0] == self._p_range[1]:
            return MetricValue(name, None, DIMENSIONLESS, self.n, "zero variance in prediction (constant field)")
        if self._r_range[0] == self._r_range[1]:
            return MetricValue(name, None, DIMENSIONLESS, self.n, "zero variance in reference (constant field)")
        n = float(self.n)
        var_p = self._spp / n - (self._sp / n) ** 2
        var_r = self._srr / n - (self._sr / n) ** 2
        cov = self._spr / n - (self._sp / n) * (self._sr / n)
        if var_p <= 0.0 or var_r <= 0.0:
            return MetricValue(name, None, DIMENSIONLESS, self.n, "variance is zero to numerical precision")
        r = cov / math.sqrt(var_p * var_r)
        return MetricValue(name, float(min(1.0, max(-1.0, r))), DIMENSIONLESS, self.n)

    def result(self, *, category: str, units: str) -> RegionMetrics:
        """Finalise into a :class:`RegionMetrics`. Absent values carry reasons."""
        empty = "no valid pixels"
        if self.n:
            mae_v = MetricValue("mae", self.sum_abs / self.n, units, self.n)
            rmse_v = MetricValue("rmse", math.sqrt(self.sum_sq / self.n), units, self.n)
            bias_v = MetricValue("bias", self.sum_err / self.n, units, self.n)
        else:
            mae_v = MetricValue("mae", None, units, 0, empty)
            rmse_v = MetricValue("rmse", None, units, 0, empty)
            bias_v = MetricValue("bias", None, units, 0, empty)

        deltas = []
        for t, under in zip(self.delta_thresholds, self._under):
            if self.n == 0:
                deltas.append(MetricValue(delta_name(t), None, FRACTION, 0, empty))
            elif self.delta_domain == 0:
                deltas.append(MetricValue(
                    delta_name(t), None, FRACTION, 0,
                    f"no valid pixel has prediction > 0 and reference > 0 "
                    f"({self.delta_excluded_nonpositive} non-positive pixel(s) excluded)",
                ))
            else:
                deltas.append(MetricValue(delta_name(t), under / self.delta_domain, FRACTION, self.delta_domain))

        return RegionMetrics(
            category=category,
            units=units,
            valid_pixels=self.n,
            mae=mae_v,
            rmse=rmse_v,
            bias=bias_v,
            pearson_r=self._pearson(),
            deltas=tuple(deltas),
            delta_domain_pixels=self.delta_domain,
            delta_excluded_nonpositive=self.delta_excluded_nonpositive,
        )


def compute_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    units: str,
    category: str = "overall",
    delta_thresholds: Sequence[float] = DEFAULT_DELTA_THRESHOLDS,
) -> RegionMetrics:
    """Every metric for one array pair, over ``valid`` pixels only.

    Args:
        prediction, reference: Same-shape arrays in the same ``units``.
        valid: Boolean evaluation mask; must exclude every non-finite pixel.
        units: Units of prediction and reference, e.g. ``"metres"``. Stated,
            never assumed: pass ``"relative (unitless)"`` for a relative field.
        category: Label carried into the result.
        delta_thresholds: Thresholds for the delta metrics (each > 1).
    """
    acc = MetricAccumulator(delta_thresholds=tuple(delta_thresholds))
    acc.add(prediction, reference, valid)
    return acc.result(category=category, units=units)


def mae(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray, *, units: str) -> MetricValue:
    """``mean(|prediction - reference|)`` over ``valid``."""
    return compute_metrics(prediction, reference, valid, units=units).mae


def rmse(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray, *, units: str) -> MetricValue:
    """``sqrt(mean((prediction - reference)^2))`` over ``valid``."""
    return compute_metrics(prediction, reference, valid, units=units).rmse


def pearson(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> MetricValue:
    """Pearson r over ``valid``, or an explained absence."""
    return compute_metrics(prediction, reference, valid, units=DIMENSIONLESS).pearson_r


def delta_accuracy(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    thresholds: Sequence[float] = DEFAULT_DELTA_THRESHOLDS,
) -> tuple[MetricValue, ...]:
    """Fraction of positive-domain pixels with ``max(p/r, r/p) < t``, per threshold."""
    return compute_metrics(prediction, reference, valid, units=DIMENSIONLESS, delta_thresholds=thresholds).deltas
