"""Seeded NumPy RANSAC for the affine calibration ``y = a * x + b``.

A shadow measurement that ran down a dark alley, or a footprint whose label
covers two buildings, produces a constraint that is not merely noisy but wrong.
Least squares -- weighted or not -- lets one such point drag ``(a, b)``
anywhere. RANSAC finds the largest set of constraints that agree with a single
line first, and only that set is handed to the weighted fit.

Algorithm (deterministic for a given ``seed``):

1. Draw two constraints with distinct ``x``; the line through them is a hypothesis.
2. Its inliers are the constraints with ``|a x_i + b - y_i| <= threshold_i``.
   ``threshold`` may be one number or one per constraint -- the calibration
   passes ``k * dh_i`` so a less certain constraint is allowed a larger residual.
3. Keep the hypothesis with the most inliers (ties: smaller total inlier residual).
4. Refit on the inliers with :func:`~depthwizard.calibration.fusion.fit_affine_wls`,
   recompute the inlier set under the refit line, and repeat until it stops
   changing (at most ``max_refinements`` times).

Every hypothesis is drawn from a :class:`numpy.random.Generator` seeded with
``seed``, so a run is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from depthwizard.calibration.fusion import AffineFit, CalibrationError, fit_affine_wls

__all__ = ["RansacResult", "ransac_affine"]


@dataclass(frozen=True)
class RansacResult:
    fit: AffineFit
    inliers: np.ndarray
    n_inliers: int
    n_outliers: int
    iterations: int
    seed: int

    @property
    def a(self) -> float:
        return self.fit.a

    @property
    def b(self) -> float:
        return self.fit.b

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_inliers": self.n_inliers,
            "n_outliers": self.n_outliers,
            "iterations": self.iterations,
            "seed": self.seed,
            **self.fit.to_dict(),
        }


def ransac_affine(
    x: Sequence[float],
    y: Sequence[float],
    *,
    threshold: float | Sequence[float],
    weights: Sequence[float] | None = None,
    iterations: int = 500,
    seed: int = 0,
    min_inliers: int = 3,
    max_condition: float = 1.0e6,
    max_refinements: int = 10,
) -> RansacResult:
    """Robustly fit ``y = a * x + b``.

    Args:
        x, y: Constraints.
        threshold: Inlier residual bound, scalar or per constraint (> 0).
        weights: WLS weights for the final refit; None = unweighted.
        iterations: Number of two-point hypotheses drawn.
        seed: Seed of the hypothesis generator.
        min_inliers: A consensus set smaller than this is a failure.
        max_condition: Passed to the final WLS fit.

    Raises:
        CalibrationError: fewer than two distinct ``x``, or no hypothesis
            reaching ``min_inliers``.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    n = xs.size
    if ys.size != n:
        raise CalibrationError(f"x and y differ in length ({n}, {ys.size})")
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n,)).copy()
    if not (np.isfinite(thresholds).all() and (thresholds > 0).all()):
        raise CalibrationError("RANSAC thresholds must be finite and positive")
    ws = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if n < 2 or np.unique(xs).size < 2:
        raise CalibrationError("RANSAC needs at least two constraints with distinct r_bar")

    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_key: tuple[int, float] | None = None
    drawn = 0
    for _ in range(int(iterations)):
        i, j = rng.choice(n, size=2, replace=False)
        drawn += 1
        if xs[i] == xs[j]:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - a * xs[i]
        residual = np.abs(a * xs + b - ys)
        inliers = residual <= thresholds
        key = (int(inliers.sum()), -float(residual[inliers].sum()))
        if best_key is None or key > best_key:
            best, best_key = inliers, key

    if best is None or int(best.sum()) < min_inliers:
        found = 0 if best is None else int(best.sum())
        raise CalibrationError(
            f"RANSAC found at most {found} mutually consistent constraint(s), need {min_inliers}"
        )

    inliers = best
    fit = fit_affine_wls(
        xs[inliers], ys[inliers], ws[inliers], min_points=min_inliers, max_condition=max_condition
    )
    for _ in range(int(max_refinements)):
        updated = np.abs(fit.a * xs + fit.b - ys) <= thresholds
        if np.array_equal(updated, inliers) or int(updated.sum()) < min_inliers:
            break
        inliers = updated
        fit = fit_affine_wls(
            xs[inliers], ys[inliers], ws[inliers], min_points=min_inliers, max_condition=max_condition
        )

    return RansacResult(
        fit=fit,
        inliers=inliers,
        n_inliers=int(inliers.sum()),
        n_outliers=int(n - inliers.sum()),
        iterations=drawn,
        seed=int(seed),
    )
