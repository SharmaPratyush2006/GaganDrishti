"""Phase 4a: the weighted least-squares a, b fit.

Tolerances are derived from the construction, never picked:

* exact data  -> machine precision (1e-9 relative);
* noise bounded by |e_i| <= delta -> the least-squares error is linear in the
  noise, dA = sum((x_i - xbar) e_i) / Sxx, so |dA| <= delta * sum|x_i - xbar| / Sxx
  and |dB| = |ebar - dA * xbar| <= delta + |xbar| * that bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthwizard.calibration.fusion import (
    IDENTIFIABLE_T,
    CalibrationError,
    classify_failure,
    classify_fit,
    fit_affine_wls,
)

TRUE_A, TRUE_B = 0.61, -1.0


def test_exact_data_recovers_a_and_b() -> None:
    x = np.linspace(1.5, 40.0, 12)
    fit = fit_affine_wls(x, TRUE_A * x + TRUE_B)
    assert fit.a == pytest.approx(TRUE_A, rel=1e-9)
    assert fit.b == pytest.approx(TRUE_B, abs=1e-9)
    assert fit.weighted_rms_residual < 1e-9


def test_bounded_noise_stays_within_the_least_squares_bound() -> None:
    rng = np.random.default_rng(20260927)
    delta = 0.25
    x = rng.uniform(1.5, 40.0, 60)
    y = TRUE_A * x + TRUE_B + rng.uniform(-delta, delta, x.size)

    fit = fit_affine_wls(x, y)

    dx = x - x.mean()
    bound_a = delta * np.abs(dx).sum() / (dx**2).sum()
    bound_b = delta + abs(x.mean()) * bound_a
    assert abs(fit.a - TRUE_A) <= bound_a
    assert abs(fit.b - TRUE_B) <= bound_b


def test_weights_match_the_closed_form_normal_equations() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(1.0, 30.0, 25)
    y = TRUE_A * x + TRUE_B + rng.normal(0.0, 0.5, x.size)
    w = 1.0 / rng.uniform(0.2, 3.0, x.size) ** 2

    fit = fit_affine_wls(x, y, w)

    sw, swx, swy = w.sum(), (w * x).sum(), (w * y).sum()
    swxx, swxy = (w * x * x).sum(), (w * x * y).sum()
    a = (sw * swxy - swx * swy) / (sw * swxx - swx**2)
    b = (swy - a * swx) / sw
    assert fit.a == pytest.approx(a, rel=1e-10)
    assert fit.b == pytest.approx(b, rel=1e-10)
    # The weights matter: the unweighted solution is a different line.
    assert abs(fit_affine_wls(x, y).a - fit.a) > 1e-6


def test_scaled_standard_error_matches_closed_form() -> None:
    rng = np.random.default_rng(11)
    x = rng.uniform(1.0, 30.0, 20)
    y = TRUE_A * x + TRUE_B + rng.normal(0.0, 0.8, x.size)

    fit = fit_affine_wls(x, y)

    rss = float((fit.residuals**2).sum())
    sxx = float(((x - x.mean()) ** 2).sum())
    assert fit.a_se == pytest.approx(math.sqrt(rss / (x.size - 2) / sxx), rel=1e-10)
    assert fit.weighted_x_variance == pytest.approx(x.var(), rel=1e-12)
    assert fit.a_over_se == pytest.approx(fit.a / fit.a_se)


def test_identifiability_classes() -> None:
    rng = np.random.default_rng(5)
    wide = np.linspace(1.0, 30.0, 15)
    strong = fit_affine_wls(wide, TRUE_A * wide + TRUE_B + rng.normal(0, 0.3, wide.size))
    assert classify_fit(strong) == "identifiable_positive"

    # Insufficient r_bar dynamic range: noise swamps the slope.
    narrow = np.linspace(1.00, 1.02, 15)
    weak = fit_affine_wls(narrow, 0.5 * narrow + 3.0 + rng.normal(0, 1.0, narrow.size))
    assert classify_fit(weak) in ("statistically_weak", "non_positive")
    assert abs(weak.a_over_se) < IDENTIFIABLE_T

    negative = fit_affine_wls(wide, -0.4 * wide + 10.0)
    assert classify_fit(negative) == "non_positive"

    with pytest.raises(CalibrationError) as unstable:
        fit_affine_wls([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    assert classify_failure(str(unstable.value)) == "numerically_unstable"
    with pytest.raises(CalibrationError) as few:
        fit_affine_wls([1.0, 2.0], [1.0, 2.0], min_points=3)
    assert classify_failure(str(few.value)) == "insufficient_constraints"


@pytest.mark.parametrize(
    "x, y",
    [
        ([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]),  # r_bar all equal: a unidentifiable
        ([1.0, 2.0], [1.0, 2.0]),  # fewer constraints than min_points
    ],
)
def test_underdetermined_fit_fails_loudly(x, y) -> None:
    with pytest.raises(CalibrationError):
        fit_affine_wls(x, y, min_points=3)
