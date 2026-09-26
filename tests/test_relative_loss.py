"""The scale-invariant loss: its defining invariances, and its masking.

Skipped in full when torch is not installed, so the rest of the suite still
runs in a plain environment.

These tests assert *mathematical properties* of the loss -- what it ignores,
what it penalises, what it refuses to score -- on tensors built by hand. They
involve no dataset, no model and no training, and therefore report no accuracy.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="Phase 3 loss requires PyTorch")

from depthwizard.relative.config import LossConfig  # noqa: E402
from depthwizard.relative.loss import (  # noqa: E402
    MIN_LINEAR_PREDICTION,
    ScaleInvariantLoss,
    align_log_shift,
    log_height,
    relative_metrics,
    scale_invariant_loss,
    to_log_space,
)


def _target(batch=2, size=8, value=10.0):
    return torch.full((batch, 1, size, size), float(value))


def _all_valid(batch=2, size=8):
    return torch.ones((batch, 1, size, size), dtype=torch.bool)


def _log_of(target: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """The exact log-space prediction that matches ``target`` perfectly."""
    return torch.log(target + cfg.epsilon)


# ---------------------------------------------------------------------------
# The defining property: invariance to a uniform log-space offset
# ---------------------------------------------------------------------------


def test_perfect_prediction_costs_nothing():
    cfg = LossConfig()
    target = _target()
    out = scale_invariant_loss(_log_of(target, cfg), target, _all_valid(), cfg)
    assert float(out.loss) == pytest.approx(0.0, abs=1e-6)


def test_uniform_offset_is_free_when_lambda_is_one():
    """This is the whole point of the loss: a global scale costs nothing.

    A prediction uniformly shifted in log space is uniformly *multiplied* in
    linear space -- exactly the degree of freedom a single image cannot fix.
    """
    cfg = LossConfig(lambda_si=1.0)
    target = _target()
    perfect = _log_of(target, cfg)
    for offset in (-2.0, -0.5, 0.5, 3.0):
        out = scale_invariant_loss(perfect + offset, target, _all_valid(), cfg)
        assert float(out.loss) == pytest.approx(0.0, abs=1e-5), offset


def test_uniform_offset_costs_its_square_when_lambda_is_zero():
    """With lambda = 0 the loss is plain log MSE, so an offset costs offset^2."""
    cfg = LossConfig(lambda_si=0.0)
    target = _target()
    perfect = _log_of(target, cfg)
    out = scale_invariant_loss(perfect + 0.5, target, _all_valid(), cfg)
    assert float(out.loss) == pytest.approx(0.25, abs=1e-5)


def test_half_lambda_charges_half_the_offset():
    """The documented default sits exactly halfway between the two extremes."""
    cfg = LossConfig(lambda_si=0.5)
    target = _target()
    out = scale_invariant_loss(_log_of(target, cfg) + 0.5, target, _all_valid(), cfg)
    assert float(out.loss) == pytest.approx(0.25 * 0.5, abs=1e-5)


def test_relative_structure_is_still_penalised_under_full_invariance():
    """Scale invariance must not mean 'anything goes'."""
    cfg = LossConfig(lambda_si=1.0)
    target = _target(batch=1, size=4)
    target[0, 0, :2, :] = 30.0  # two different height levels
    perfect = _log_of(target, cfg)
    wrong = perfect.clone()
    wrong[0, 0, :2, :] -= 1.0  # squash the contrast: a real structural error
    assert float(scale_invariant_loss(wrong, target, _all_valid(1, 4), cfg).loss) > 0.01


def test_loss_is_never_negative():
    cfg = LossConfig(lambda_si=1.0)
    target = torch.rand(4, 1, 8, 8) * 50.0
    prediction = torch.randn(4, 1, 8, 8)
    out = scale_invariant_loss(prediction, target, _all_valid(4, 8), cfg)
    assert float(out.loss) >= 0.0
    assert (out.per_sample >= 0).all()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_invalid_pixels_do_not_affect_the_loss():
    cfg = LossConfig()
    target = _target(batch=1, size=8)
    prediction = _log_of(target, cfg)
    valid = _all_valid(1, 8)
    valid[0, 0, :4, :] = False

    baseline = float(scale_invariant_loss(prediction, target, valid, cfg).loss)

    # Put arbitrary garbage in the prediction where the mask says "no data".
    polluted = prediction.clone()
    polluted[0, 0, :4, :] = 1234.5
    assert float(scale_invariant_loss(polluted, target, valid, cfg).loss) == pytest.approx(
        baseline, abs=1e-6
    )


def test_a_sentinel_left_in_the_target_cannot_poison_the_loss():
    """log(-9999 + 1) is NaN, and NaN * 0 is NaN -- so masking alone is not enough.

    The dataset zeroes invalid heights, but the loss must not *depend* on that.
    """
    cfg = LossConfig()
    target = _target(batch=1, size=8)
    target[0, 0, :4, :] = -9999.0
    valid = _all_valid(1, 8)
    valid[0, 0, :4, :] = False

    out = scale_invariant_loss(_log_of(_target(1, 8), cfg), target, valid, cfg)
    assert math.isfinite(float(out.loss))
    assert float(out.loss) == pytest.approx(0.0, abs=1e-6)


def _with_negative_heights(batch=1, size=8):
    """Mostly buildings, plus VALID negative AGL pixels, including below -epsilon."""
    target = torch.rand(batch, 1, size, size) * 30.0
    target[:, 0, 0, 0] = -0.5
    target[:, 0, 0, 1] = -1.0  # exactly -epsilon: log(h + 1) would be -inf
    target[:, 0, 0, 2] = -3.0  # below -epsilon: log(h + 1) would be NaN
    return target


def test_valid_negative_heights_give_a_finite_loss():
    """Regression: one valid -3 m AGL pixel used to turn the whole loss into NaN."""
    cfg = LossConfig()
    target = _with_negative_heights()
    out = scale_invariant_loss(torch.zeros_like(target), target, _all_valid(1, 8), cfg)
    assert math.isfinite(float(out.loss))
    assert torch.isfinite(out.per_sample).all()


def test_valid_negative_heights_are_scored_not_dropped():
    """They are measurements: a perfect fit costs 0 and a wrong one costs > 0."""
    cfg = LossConfig(lambda_si=1.0)
    target = _with_negative_heights()
    _, log_target, _ = to_log_space(torch.zeros_like(target), target, cfg)
    perfect = scale_invariant_loss(log_target, target, _all_valid(1, 8), cfg)
    assert float(perfect.loss) == pytest.approx(0.0, abs=1e-6)
    assert perfect.usable_count == 1

    wrong = log_target.clone()
    wrong[0, 0, 0, 2] += 2.0  # misplace only the -3 m pixel
    assert float(scale_invariant_loss(wrong, target, _all_valid(1, 8), cfg).loss) > 0.0


def test_log_height_is_unchanged_for_non_negative_heights():
    """The documented log(h + epsilon) must still hold wherever it was defined."""
    heights = torch.tensor([0.0, 0.25, 1.0, 7.5, 300.0])
    for epsilon in (0.5, 1.0, 2.0):
        assert torch.equal(log_height(heights, epsilon), torch.log(heights + epsilon))


def test_log_height_is_finite_ordered_and_continuous_below_zero():
    heights = torch.tensor([-500.0, -3.0, -1.0, -0.5, -1e-6, 0.0, 1e-6, 2.0])
    values = log_height(heights, 1.0)
    assert torch.isfinite(values).all()
    assert (values[1:] > values[:-1]).all()  # strictly increasing: order kept
    assert float(values[4]) == pytest.approx(float(values[5]), abs=1e-5)
    assert float(values[6]) == pytest.approx(float(values[5]), abs=1e-5)


def test_gradient_is_finite_with_valid_negative_heights():
    cfg = LossConfig()
    target = _with_negative_heights(batch=2)
    prediction = torch.randn(2, 1, 8, 8).requires_grad_(True)
    scale_invariant_loss(prediction, target, _all_valid(2, 8), cfg).loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad[0, 0, 0, 2].abs()) > 0.0  # it is trained on


def test_metrics_are_finite_with_valid_negative_heights():
    cfg = LossConfig()
    target = _with_negative_heights(batch=2)
    metrics = relative_metrics(torch.randn(2, 1, 8, 8), target, _all_valid(2, 8), cfg)
    for key, value in metrics.items():
        assert math.isfinite(value), key
    assert metrics["abs_rel_aligned"] >= 0.0


def test_a_negative_sentinel_is_still_masked_not_treated_as_negative_height():
    """The negative-height support must not let -9999 back in as a measurement."""
    cfg = LossConfig()
    target = _target(batch=1, size=8)
    target[0, 0, :4, :] = -9999.0
    valid = _all_valid(1, 8)
    valid[0, 0, :4, :] = False
    polluted = scale_invariant_loss(_log_of(_target(1, 8), cfg), target, valid, cfg)
    assert float(polluted.loss) == pytest.approx(0.0, abs=1e-6)


def test_samples_below_min_valid_pixels_are_dropped():
    cfg = LossConfig(min_valid_pixels=32)
    target = _target(batch=2, size=8)
    prediction = _log_of(target, cfg) + 1.0
    valid = _all_valid(2, 8)
    valid[1] = False
    valid[1, 0, 0, :4] = True  # only 4 valid pixels: below the threshold

    out = scale_invariant_loss(prediction, target, valid, cfg)
    assert out.usable_count == 1
    assert out.stats["dropped_samples"] == 1
    assert bool(out.usable[0]) and not bool(out.usable[1])
    assert float(out.per_sample[1]) == 0.0


def test_a_batch_with_nothing_usable_scores_zero_and_says_so():
    """Not a good loss -- an unmeasurable one. The trainer skips the step."""
    cfg = LossConfig(min_valid_pixels=32)
    target = _target(batch=2, size=8)
    valid = torch.zeros_like(target, dtype=torch.bool)
    out = scale_invariant_loss(_log_of(target, cfg), target, valid, cfg)
    assert out.usable_count == 0
    assert out.stats["usable_samples"] == 0
    assert float(out.loss) == 0.0


def test_stats_report_what_was_actually_scored():
    cfg = LossConfig()
    target = _target(batch=2, size=8)
    valid = _all_valid(2, 8)
    valid[0, 0, :4, :] = False
    out = scale_invariant_loss(_log_of(target, cfg), target, valid, cfg)
    assert out.stats["batch_size"] == 2
    assert out.stats["valid_fraction"] == pytest.approx(0.75)
    assert "mean_log_offset" in out.stats


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradient_flows_to_the_prediction():
    cfg = LossConfig()
    target = _target()
    prediction = (_log_of(target, cfg) + 0.3).requires_grad_(True)
    scale_invariant_loss(prediction, target, _all_valid(), cfg).loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_masked_pixels_receive_no_gradient():
    cfg = LossConfig()
    target = _target(batch=1, size=8)
    valid = _all_valid(1, 8)
    valid[0, 0, :4, :] = False
    prediction = (_log_of(target, cfg) + 0.3).requires_grad_(True)
    scale_invariant_loss(prediction, target, valid, cfg).loss.backward()
    assert float(prediction.grad[0, 0, :4, :].abs().sum()) == 0.0
    assert float(prediction.grad[0, 0, 4:, :].abs().sum()) > 0.0


def test_gradient_is_finite_when_a_batch_is_unusable():
    """A skipped batch must not produce NaN gradients."""
    cfg = LossConfig(min_valid_pixels=32)
    target = _target()
    prediction = _log_of(target, cfg).requires_grad_(True)
    valid = torch.zeros_like(target, dtype=torch.bool)
    scale_invariant_loss(prediction, target, valid, cfg).loss.backward()
    assert torch.isfinite(prediction.grad).all()


# ---------------------------------------------------------------------------
# predict_log = false
# ---------------------------------------------------------------------------


def test_linear_prediction_is_clamped_and_the_clamping_is_counted():
    cfg = LossConfig(predict_log=False)
    target = _target(batch=1, size=4)
    prediction = torch.full((1, 1, 4, 4), -50.0)  # far below zero
    out = scale_invariant_loss(prediction, target, _all_valid(1, 4), cfg)
    assert math.isfinite(float(out.loss))
    assert out.stats["clamped_pixels"] == 16


def test_predict_log_true_never_clamps():
    cfg = LossConfig(predict_log=True)
    target = _target(batch=1, size=4)
    _, _, clamped = to_log_space(torch.full((1, 1, 4, 4), -50.0), target, cfg)
    assert not clamped.any()


def test_linear_mode_matches_log_mode_on_equivalent_inputs():
    target = _target(batch=1, size=4, value=10.0)
    valid = _all_valid(1, 4)
    linear = LossConfig(predict_log=False)
    log_mode = LossConfig(predict_log=True)
    # A linear prediction of 10.0 and a log prediction of log(11) describe the
    # same height under epsilon = 1.
    linear_loss = scale_invariant_loss(torch.full((1, 1, 4, 4), 10.0), target, valid, linear)
    log_loss = scale_invariant_loss(_log_of(target, log_mode), target, valid, log_mode)
    assert float(linear_loss.loss) == pytest.approx(float(log_loss.loss), abs=1e-6)


def test_min_linear_prediction_is_positive():
    assert MIN_LINEAR_PREDICTION > 0.0


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_accepts_both_channel_conventions():
    cfg = LossConfig()
    target = _target(batch=1, size=4)
    prediction = _log_of(target, cfg)
    with_channel = scale_invariant_loss(prediction, target, _all_valid(1, 4), cfg)
    without = scale_invariant_loss(
        prediction[:, 0], target[:, 0], _all_valid(1, 4)[:, 0], cfg
    )
    assert float(with_channel.loss) == pytest.approx(float(without.loss))


def test_multi_channel_prediction_is_refused():
    cfg = LossConfig()
    with pytest.raises(ValueError, match="exactly 1 channel"):
        scale_invariant_loss(
            torch.zeros(1, 2, 4, 4), _target(1, 4), _all_valid(1, 4), cfg
        )


def test_shape_mismatch_is_refused_rather_than_resized():
    cfg = LossConfig()
    with pytest.raises(ValueError, match="same shape"):
        scale_invariant_loss(torch.zeros(1, 1, 8, 8), _target(1, 4), _all_valid(1, 4), cfg)


def test_rank_one_input_is_refused():
    cfg = LossConfig()
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        scale_invariant_loss(torch.zeros(4), torch.zeros(4), torch.ones(4), cfg)


# ---------------------------------------------------------------------------
# The nn.Module wrapper
# ---------------------------------------------------------------------------


def test_module_matches_the_function():
    cfg = LossConfig()
    target = _target()
    prediction = _log_of(target, cfg) + 0.2
    module = ScaleInvariantLoss(cfg)
    assert float(module(prediction, target, _all_valid()).loss) == pytest.approx(
        float(scale_invariant_loss(prediction, target, _all_valid(), cfg).loss)
    )


def test_module_has_no_parameters():
    assert list(ScaleInvariantLoss().parameters()) == []


def test_module_repr_states_its_settings():
    text = repr(ScaleInvariantLoss(LossConfig(lambda_si=0.75)))
    assert "lambda_si=0.75" in text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_alignment_recovers_a_known_offset():
    cfg = LossConfig()
    target = _target(batch=1, size=8)
    log_target = _log_of(target, cfg)
    shift = align_log_shift(log_target - 0.75, log_target, _all_valid(1, 8)[:, 0].float())
    assert float(shift[0]) == pytest.approx(0.75, abs=1e-6)


def test_metrics_are_perfect_for_a_perfect_prediction():
    cfg = LossConfig()
    target = torch.rand(2, 1, 8, 8) * 40.0
    metrics = relative_metrics(_log_of(target, cfg), target, _all_valid(2, 8), cfg)
    assert metrics["si_rmse"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["rmse_log_aligned"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["abs_rel_aligned"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["delta_1.250_aligned"] == pytest.approx(1.0)


def test_si_rmse_ignores_a_uniform_offset():
    cfg = LossConfig()
    target = torch.rand(2, 1, 8, 8) * 40.0
    perfect = _log_of(target, cfg)
    assert relative_metrics(perfect + 1.5, target, _all_valid(2, 8), cfg)[
        "si_rmse"
    ] == pytest.approx(
        relative_metrics(perfect, target, _all_valid(2, 8), cfg)["si_rmse"], abs=1e-5
    )


def test_metrics_are_nan_not_zero_when_nothing_is_valid():
    """An unmeasurable metric must not look like a perfect one."""
    cfg = LossConfig()
    target = _target()
    metrics = relative_metrics(
        _log_of(target, cfg), target, torch.zeros_like(target, dtype=torch.bool), cfg
    )
    assert math.isnan(metrics["si_rmse"])
    assert math.isnan(metrics["abs_rel_aligned"])
    assert metrics["valid_pixels"] == 0.0


def test_an_empty_sample_does_not_count_as_a_perfect_one():
    """A sample with no valid pixel is excluded, not averaged in as si_rmse 0."""
    cfg = LossConfig()
    target = torch.rand(2, 1, 8, 8) * 40.0 + 1.0
    prediction = torch.randn(2, 1, 8, 8)
    valid = _all_valid(2, 8).clone()
    valid[1] = False

    both = relative_metrics(prediction, target, valid, cfg)
    alone = relative_metrics(prediction[:1], target[:1], valid[:1], cfg)
    assert both["scored_samples"] == 1.0
    assert both["si_rmse"] == pytest.approx(alone["si_rmse"], rel=1e-5)
    assert both["rmse_log_aligned"] == pytest.approx(alone["rmse_log_aligned"], rel=1e-5)


def test_metrics_record_that_they_used_the_ground_truth_scale():
    cfg = LossConfig()
    target = _target()
    metrics = relative_metrics(_log_of(target, cfg), target, _all_valid(), cfg)
    assert metrics["scale_aligned"] == 1.0
    assert all(
        key.endswith("_aligned")
        for key in metrics
        if key.startswith(("rmse_log", "abs_rel", "delta_"))
    )


def test_delta_thresholds_are_ordered():
    cfg = LossConfig()
    target = torch.rand(2, 1, 8, 8) * 40.0
    metrics = relative_metrics(torch.randn(2, 1, 8, 8), target, _all_valid(2, 8), cfg)
    assert (
        metrics["delta_1.250_aligned"]
        <= metrics["delta_1.562_aligned"]
        <= metrics["delta_1.953_aligned"]
    )
