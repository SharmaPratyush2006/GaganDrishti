"""Phase 4a: seeded RANSAC for the affine calibration.

Construction: inliers ``y = a x + b + e`` with ``|e| <= delta``; outliers
displaced by at least ``3 * threshold``. The threshold is derived, not picked:
after the least-squares refit on the true inliers, an inlier residual is at
most ``|e_i| + |dA (x_i - xbar) + ebar| <= 2 delta + max|x - xbar| * bound_a``
(``bound_a`` as in test_calibration_wls). With the threshold equal to that
bound, every inlier is kept and every outlier, being 3x further, is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthwizard.calibration.ransac import ransac_affine

TRUE_A, TRUE_B = 0.61, -1.0


def test_exact_data_recovers_a_and_b() -> None:
    x = np.linspace(1.0, 30.0, 10)
    result = ransac_affine(x, TRUE_A * x + TRUE_B, threshold=1e-9, seed=1)
    assert result.n_outliers == 0
    assert result.a == pytest.approx(TRUE_A, rel=1e-9)
    assert result.b == pytest.approx(TRUE_B, abs=1e-9)


def test_rejects_injected_outliers_and_recovers_a_b() -> None:
    rng = np.random.default_rng(20260927)
    delta = 0.2
    x_in = rng.uniform(1.0, 30.0, 40)
    y_in = TRUE_A * x_in + TRUE_B + rng.uniform(-delta, delta, x_in.size)

    dx = x_in - x_in.mean()
    bound_a = delta * np.abs(dx).sum() / (dx**2).sum()
    bound_b = delta + abs(x_in.mean()) * bound_a
    threshold = 2 * delta + np.abs(dx).max() * bound_a

    n_out = 10
    x_out = rng.uniform(1.0, 30.0, n_out)
    sign = rng.choice([-1.0, 1.0], n_out)
    y_out = TRUE_A * x_out + TRUE_B + sign * rng.uniform(3 * threshold, 6 * threshold, n_out)

    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    order = rng.permutation(x.size)
    is_outlier = np.concatenate([np.zeros(x_in.size, bool), np.ones(n_out, bool)])[order]

    result = ransac_affine(x[order], y[order], threshold=threshold, iterations=500, seed=3)

    np.testing.assert_array_equal(result.inliers, ~is_outlier)
    assert abs(result.a - TRUE_A) <= bound_a
    assert abs(result.b - TRUE_B) <= bound_b
    # Seeded: the same call gives the same answer.
    again = ransac_affine(x[order], y[order], threshold=threshold, iterations=500, seed=3)
    assert (again.a, again.b) == (result.a, result.b)
