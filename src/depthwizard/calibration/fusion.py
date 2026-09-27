"""Fusing the Phase 3 relative field with shadow-derived heights.

Phase 3 predicts ``z_rel ~= log(AGL + 1) + c`` with an unknown per-tile ``c``.
Linearised, ``r = exp(z_rel)`` is proportional to ``AGL + 1``, so metric AGL is
an affine function of ``r``::

    AGL(x) = a * r(x) + b

and one ``(a, b)`` pair per tile is fitted from building constraints::

    a * r_bar_i + b = h_i        (weights w_i = 1 / dh_i^2, confidence-adjusted)

where ``r_bar_i`` is the mean of ``r`` over building ``i``'s footprint and
``h_i`` its shadow-derived height. Only ``a`` and ``b`` are unknown: the sun
geometry was fixed beforehand (see :mod:`depthwizard.calibration.shadow_anchor`),
which is what keeps the problem from being scale-degenerate.

A fit that is underdetermined or numerically unstable raises
:class:`CalibrationError`. It never returns an arbitrary ``(a, b)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CalibrationError",
    "AffineFit",
    "IDENTIFIABILITY_CLASSES",
    "IDENTIFIABLE_T",
    "classify_fit",
    "classify_failure",
    "linearise_relative",
    "footprint_means",
    "fit_affine_wls",
    "apply_affine",
]


class CalibrationError(RuntimeError):
    """A calibration could not be determined; the message says why."""


def linearise_relative(z_rel: np.ndarray, *, log_space: bool) -> np.ndarray:
    """``r = exp(z_rel)`` for a log-space Phase 3 field.

    Raises:
        CalibrationError: if the field is not log-space (the affine model
            ``a * r + b`` is only derived for the log-space head), or the
            exponential overflows.
    """
    if not log_space:
        raise CalibrationError(
            "the relative field is not log-space; a*exp(z)+b is only defined for "
            "the Phase 3 log-space head"
        )
    z = np.asarray(z_rel, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            r = np.exp(z)
        except FloatingPointError as exc:
            raise CalibrationError(f"exp(z_rel) is not finite: {exc}") from exc
    return r


def footprint_means(r: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of ``r`` over each boolean mask."""
    return np.array([float(r[np.asarray(m, dtype=bool)].mean()) for m in masks], dtype=np.float64)


@dataclass(frozen=True)
class AffineFit:
    """A fitted ``y = a * x + b`` and the numbers that justify trusting it."""

    a: float
    b: float
    n: int
    #: Condition number of the weighted design matrix ``sqrt(W) [x 1]``.
    condition: float
    #: ``sqrt(sum w r^2 / sum w)``, in the units of ``y``.
    weighted_rms_residual: float
    #: ``sum w r^2 / (n - 2)``; ~1 when the weights are honest 1/sigma^2.
    reduced_chi2: float | None
    #: Formal standard errors from ``(X^T W X)^-1``, unscaled.
    a_std: float
    b_std: float
    residuals: np.ndarray
    #: ``sum w (x - xbar_w)^2 / sum w`` -- the spread of r_bar the fit has to work with.
    weighted_x_variance: float = math.nan
    #: Standard errors scaled by the residual variance factor ``sqrt(reduced_chi2)``:
    #: the usual WLS errors when the weights are only relative (as the ray-spread
    #: proxy is). None when there are no residual degrees of freedom.
    a_se: float | None = None
    b_se: float | None = None

    @property
    def a_over_se(self) -> float | None:
        return self.a / self.a_se if self.a_se else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "condition": self.condition,
            "weighted_rms_residual": self.weighted_rms_residual,
            "reduced_chi2": self.reduced_chi2,
            "weighted_r_bar_variance": self.weighted_x_variance,
            "a_se": self.a_se,
            "b_se": self.b_se,
            "a_over_se": self.a_over_se,
            "a_std_formal": self.a_std,
            "b_std_formal": self.b_std,
            "identifiability": classify_fit(self),
        }


#: Fit classes. The first two are UNIDENTIFIABLE: no (a, b) is returned at all.
IDENT_INSUFFICIENT = "insufficient_constraints"
IDENT_UNSTABLE = "numerically_unstable"
IDENT_NON_POSITIVE = "non_positive"
IDENT_WEAK = "statistically_weak"
IDENT_OK = "identifiable_positive"
IDENTIFIABILITY_CLASSES = (IDENT_OK, IDENT_WEAK, IDENT_NON_POSITIVE, IDENT_UNSTABLE, IDENT_INSUFFICIENT)

#: |a / SE(a)| needed to call a significantly non-zero: the conventional
#: two-sided 95 % normal quantile, rounded. Not tuned on any data.
IDENTIFIABLE_T = 2.0


def classify_fit(fit: "AffineFit", t_threshold: float = IDENTIFIABLE_T) -> str:
    """Label a returned fit. A label, never a gate: every class is reported.

    ``non_positive`` (a <= 0) takes precedence; otherwise ``identifiable_positive``
    needs ``a / SE(a) >= t_threshold``, else ``statistically_weak``.
    """
    if fit.a <= 0.0:
        return IDENT_NON_POSITIVE
    ratio = fit.a_over_se
    if ratio is None or ratio < t_threshold:
        return IDENT_WEAK
    return IDENT_OK


def classify_failure(reason: str | None) -> str:
    """Map a :class:`CalibrationError` message to an UNIDENTIFIABLE class."""
    text = (reason or "").lower()
    if "underdetermined" in text or "at most" in text or "need at least" in text:
        return IDENT_INSUFFICIENT
    return IDENT_UNSTABLE


def fit_affine_wls(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    min_points: int = 3,
    max_condition: float = 1.0e6,
) -> AffineFit:
    """Weighted least squares for ``y = a * x + b``.

    Minimises ``sum_i w_i (a x_i + b - y_i)^2``, solved as an ordinary least
    squares problem on ``sqrt(w)``-scaled rows.

    Raises:
        CalibrationError: fewer than ``min_points`` points, non-finite input,
            non-positive weights, all ``x`` equal (``a`` is unidentifiable), or
            a weighted design matrix worse conditioned than ``max_condition``.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ws = np.ones_like(xs) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if not (xs.size == ys.size == ws.size):
        raise CalibrationError(f"x, y and weights differ in length ({xs.size}, {ys.size}, {ws.size})")
    if xs.size < max(2, min_points):
        raise CalibrationError(
            f"underdetermined: {xs.size} constraint(s), need at least {max(2, min_points)}"
        )
    if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(ws).all()):
        raise CalibrationError("non-finite constraint values or weights")
    if (ws <= 0.0).any():
        raise CalibrationError("weights must be strictly positive")
    if np.ptp(xs) <= 0.0:
        raise CalibrationError("all r_bar values are identical; the scale a is unidentifiable")

    root_w = np.sqrt(ws)
    design = np.column_stack([xs, np.ones_like(xs)]) * root_w[:, None]
    target = ys * root_w
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > max_condition:
        raise CalibrationError(
            f"weighted design matrix condition number {condition:.3g} exceeds {max_condition:.3g}; "
            "the fit is numerically unstable"
        )
    (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = a * xs + b - ys
    weighted_ss = float((ws * residuals**2).sum())
    covariance = np.linalg.inv(design.T @ design)
    dof = xs.size - 2
    x_mean_w = float((ws * xs).sum() / ws.sum())
    scale = math.sqrt(weighted_ss / dof) if dof > 0 else None
    return AffineFit(
        a=float(a),
        b=float(b),
        n=int(xs.size),
        condition=condition,
        weighted_rms_residual=math.sqrt(weighted_ss / float(ws.sum())),
        reduced_chi2=(weighted_ss / dof) if dof > 0 else None,
        a_std=float(math.sqrt(covariance[0, 0])),
        b_std=float(math.sqrt(covariance[1, 1])),
        residuals=residuals,
        weighted_x_variance=float((ws * (xs - x_mean_w) ** 2).sum() / ws.sum()),
        a_se=float(math.sqrt(covariance[0, 0]) * scale) if scale is not None else None,
        b_se=float(math.sqrt(covariance[1, 1]) * scale) if scale is not None else None,
    )


def apply_affine(r: np.ndarray, a: float, b: float) -> np.ndarray:
    """``AGL_pred = a * r + b``."""
    return a * np.asarray(r, dtype=np.float64) + b
