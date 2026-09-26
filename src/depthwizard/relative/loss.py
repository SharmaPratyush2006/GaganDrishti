"""Masked scale-invariant loss for the relative-height baseline.

What this loss optimises, and what it deliberately leaves undetermined
---------------------------------------------------------------------
A single satellite image does not contain the information needed to fix an
absolute height scale. Two scenes with identical pixels can sit at different
ground sample distances, and nothing in the RGB alone distinguishes a 30 m
tower at one GSD from a 60 m tower at another. Training a plain regression on
metres therefore asks the network to guess a number it cannot see, and it
learns the dataset's height prior instead of the image's height structure.

The scale-invariant log loss of Eigen et al. (NeurIPS 2014) removes exactly
that unlearnable degree of freedom. Writing ``d = log(pred) - log(target)`` over
the valid pixels of one sample::

    L = (1/n) * sum(d^2)  -  lambda * ( (1/n) * sum(d) )^2

The first term is ordinary log-space MSE. The second subtracts the part of the
error that is *common to the whole tile* -- a uniform offset in log space, which
is a uniform multiplicative factor in linear space. With ``lambda = 1`` a
prediction that is uniformly twice the truth costs nothing at all: only the
*relative* arrangement of heights is scored. With ``lambda = 0`` the loss is
scale sensitive again. ``lambda = 0.5`` is the usual compromise and this
project's default: it rewards getting the relative structure right while still
discouraging the scale from drifting arbitrarily far.

The direct consequence, which is stated here because everything downstream
depends on it: **the output of a model trained with this loss is unitless.**
It is defined only up to an additive constant in log space. It is not metres,
it is not elevation, and converting it to metres is Phase 4's problem, not
something this module quietly does.

Why the logarithm, and why epsilon
----------------------------------
Working in log space makes the error *relative*: being 2 m wrong on a 6 m house
costs the same as being 10 m wrong on a 30 m tower. That matches how the
prediction will be judged, and stops the handful of tall buildings in a tile
from owning the entire gradient.

But ground-level pixels have a true AGL height of exactly 0, and ``log(0)`` is
``-inf``. :attr:`~depthwizard.relative.config.LossConfig.epsilon` (1.0 by
default) is added before the logarithm, so ground maps to ``log(1) = 0`` and
stays a usable, finite training signal. The alternative -- masking out every
zero-height pixel -- would throw away the ground, which is most of the image and
the reference level the rest is measured against.

A *valid* AGL pixel can also be slightly negative (AGL is a DSM minus a terrain
model, and both have error). ``log(h + 1)`` is NaN below -1, so the target
transform is :func:`log_height`: exactly ``log(h + epsilon)`` for ``h >= 0``,
reflected symmetrically below zero so negative heights stay finite, ordered and
in the loss rather than being discarded.

Why the head predicts log-height by default
-------------------------------------------
With ``loss.predict_log`` true the network's raw output *is* the log-height, and
no logarithm is taken of the prediction. That is the safe direction:

* a linear-space head is free to emit negative numbers, which have no
  logarithm, so the loss would have to clamp them -- and a clamp produces a
  zero gradient exactly where the model is most wrong, which is the worst place
  to stop learning;
* log space is unbounded in both directions, so the head never has to fight an
  activation function to reach the value it wants;
* the loss is defined in log space anyway, so predicting there removes a
  non-linearity from the gradient path.

``predict_log: false`` is supported for comparison, and then the prediction is
clamped to a small positive value before the logarithm. The clamping is counted
and reported in the statistics rather than hidden.

Masking
-------
Every reduction here is over the **valid** pixels only, per sample. Invalid
pixels (nodata, sentinel, padding -- see :mod:`depthwizard.relative.data`) are
excluded from both sums and from ``n``. They are never imputed and never
contribute a fabricated target.

A sample with fewer than ``loss.min_valid_pixels`` valid pixels is dropped from
the batch entirely: its scale-invariant term is estimated from too few pixels to
mean anything, and letting it through would inject noise that looks like signal.
The threshold is capped at the sample's own pixel count, so it drops samples
for *missing data*, never for their tile size (see
:func:`_effective_min_valid_pixels`); at real tile sizes the cap never engages.
Dropped samples are counted in the returned statistics, so an epoch that quietly
consists mostly of empty tiles is visible rather than mysterious.

Metrics
-------
:func:`relative_metrics` reports the accuracy numbers that are *meaningful for a
scale-invariant prediction*, and each one says what it did about the free scale.
There is no metre-space MAE or RMSE here. Producing one would require choosing a
scale, and any scale chosen after seeing the ground truth measures something
other than what the model would do at inference time. Phase 4 calibrates the
scale from geometry; only then is a metric error meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from depthwizard.relative.config import LossConfig

__all__ = [
    "LossOutput",
    "ScaleInvariantLoss",
    "scale_invariant_loss",
    "to_log_space",
    "log_height",
    "align_log_shift",
    "relative_metrics",
    "MIN_LINEAR_PREDICTION",
]

#: Floor applied to a linear-space prediction before taking its logarithm, used
#: only when ``loss.predict_log`` is false. Small enough not to distort a real
#: near-ground prediction, large enough that its log is finite in float32.
MIN_LINEAR_PREDICTION = 1e-6


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------


def _squeeze_channel(tensor: Tensor, name: str) -> Tensor:
    """Accept ``[B, 1, H, W]`` or ``[B, H, W]``; return ``[B, H, W]``."""
    if tensor.ndim == 4:
        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must have exactly 1 channel, got {tensor.shape[1]}. "
                "Phase 3 has a single height head; a multi-channel tensor here "
                "means something upstream is predicting more than height."
            )
        return tensor[:, 0]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"{name} must be [B, 1, H, W] or [B, H, W], got {tuple(tensor.shape)}")


def _neutralise_invalid(target: Tensor, mask: Tensor) -> Tensor:
    """Replace the target at invalid pixels with a value whose log is finite.

    Masking by multiplication is only safe if the masked-out values are finite:
    ``log(-9999 + 1)`` is NaN, and ``NaN * 0.0`` is still NaN, so a single
    sentinel left in the target would poison the whole batch's sum rather than
    being ignored.

    :mod:`depthwizard.relative.data` already zeroes invalid pixels, so in the
    normal pipeline this changes nothing. It is done here as well so the loss is
    correct for *any* caller -- a hand-built tensor in a test, a different
    reader, a future dataset -- rather than silently depending on an invariant
    established in another module.

    This does not invent a target: the substituted pixels are excluded from
    every sum by the same mask. Their value is arithmetically irrelevant and is
    chosen only to be finite.
    """
    return torch.where(mask > 0, target, torch.zeros_like(target))


def _check_alignment(prediction: Tensor, target: Tensor, valid: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} "
            "must have the same shape. The loss never resizes either one: a resize "
            "here would silently break the pixel correspondence that the dataset "
            "went to some trouble to guarantee."
        )
    if valid.shape != target.shape:
        raise ValueError(
            f"valid mask {tuple(valid.shape)} must match the target {tuple(target.shape)}"
        )


# ---------------------------------------------------------------------------
# Log-space conversion
# ---------------------------------------------------------------------------


def log_height(height: Tensor, epsilon: float) -> Tensor:
    """The loss's log transform of a height, finite for every real input.

    For ``h >= 0`` this is exactly ``log(h + epsilon)``, the documented
    transform. For ``h < 0`` it is the odd reflection of that curve about
    ``h = 0``::

        f(h) = log(epsilon) - log1p(-h / epsilon)          (h < 0)

    Why not simply ``log(h + epsilon)`` everywhere: above-ground-level rasters
    are a DSM minus a terrain model, and both carry error, so a *valid* AGL
    pixel can be slightly negative (a road cut, a terrain-model overshoot).
    ``log(h + 1)`` is NaN for ``h < -1`` and ``-inf`` at ``-1``, and one such
    pixel poisons the whole batch. Marking every negative height invalid would
    throw real measurements away; clamping would flatten their ordering.

    The reflection keeps the value and slope continuous at 0 (both sides are
    ``log(epsilon)`` with slope ``1/epsilon``), is strictly increasing -- so a
    deeper negative stays lower -- and is finite for all finite ``h``. Sentinel
    nodata values never reach here as measurements: they are masked, and
    :func:`_neutralise_invalid` zeroes them before the transform.
    """
    log_eps = math.log(epsilon)
    positive = torch.log(height.clamp(min=0.0) + epsilon)
    negative = log_eps - torch.log1p((-height).clamp(min=0.0) / epsilon)
    return torch.where(height >= 0, positive, negative)


def to_log_space(
    prediction: Tensor,
    target: Tensor,
    cfg: LossConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Put prediction and target in the common log space the loss works in.

    Returns ``(log_prediction, log_target, clamped)`` where ``clamped`` is a
    boolean tensor marking prediction pixels that had to be floored at
    :data:`MIN_LINEAR_PREDICTION` -- always all-False when ``cfg.predict_log``,
    because then no logarithm is taken of the prediction at all.

    The target goes through :func:`log_height`: exactly ``log(target +
    epsilon)`` for ``target >= 0``, extended symmetrically below zero so that a
    legitimately negative above-ground height stays finite and ordered. The
    target is a real measurement in its own units and is never clamped or
    rescaled.
    """
    log_target = log_height(target, cfg.epsilon)

    if cfg.predict_log:
        # The head already lives in log space. Nothing to do, and in particular
        # nothing to clamp -- which is the whole point of this being the default.
        return prediction, log_target, torch.zeros_like(prediction, dtype=torch.bool)

    shifted = prediction + cfg.epsilon
    clamped = shifted < MIN_LINEAR_PREDICTION
    log_prediction = torch.log(shifted.clamp(min=MIN_LINEAR_PREDICTION))
    return log_prediction, log_target, clamped


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


@dataclass
class LossOutput:
    """The differentiable loss plus everything worth logging about it."""

    #: Scalar, batch-mean over usable samples. This is what ``.backward()`` runs on.
    loss: Tensor
    #: Per-sample scale-invariant loss, ``[B]``. Zero for dropped samples.
    per_sample: Tensor
    #: ``[B]`` bool: samples that met ``min_valid_pixels`` and were scored.
    usable: Tensor
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def usable_count(self) -> int:
        return int(self.usable.sum().item())

    def __float__(self) -> float:
        return float(self.loss.detach().item())


def _effective_min_valid_pixels(cfg: LossConfig, pixels_per_sample: int) -> int:
    """The per-sample valid-pixel threshold actually applied.

    ``loss.min_valid_pixels`` guards against samples that *masking* has left
    nearly empty -- nodata, sentinels, padding. It is not a minimum tile size:
    the tile size is fixed by ``training.image_size`` (a multiple of 32, so at
    least 1024 pixels) and validated there. The threshold is therefore capped
    at the number of pixels a sample has, so a sample whose every pixel is
    valid is never dropped for being "insufficient" -- that would reject the
    sample for its shape, not for missing data.

    At any real training tile size the cap is inactive and the configured
    threshold applies unchanged.
    """
    if pixels_per_sample <= 0:
        return cfg.min_valid_pixels
    return min(cfg.min_valid_pixels, pixels_per_sample)


def scale_invariant_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    cfg: LossConfig,
) -> LossOutput:
    """Masked scale-invariant log loss over a batch.

    Args:
        prediction: ``[B, 1, H, W]`` or ``[B, H, W]``. Log-height when
            ``cfg.predict_log``, linear height otherwise.
        target: same shape, ground-truth height in the dataset's own units.
        valid: same shape, bool or float. True where ``target`` is a real
            measurement.
        cfg: the Phase 3 loss configuration.

    Returns:
        A :class:`LossOutput`. When no sample in the batch has enough valid
        pixels the loss is an exact zero that still carries a gradient path, and
        ``stats['usable_samples']`` is 0 -- the trainer checks that rather than
        taking a meaningless optimiser step.
    """
    prediction = _squeeze_channel(prediction, "prediction")
    target = _squeeze_channel(target, "target")
    valid = _squeeze_channel(valid, "valid")
    _check_alignment(prediction, target, valid)

    # Promote to float32 for the reductions regardless of autocast: the sums run
    # over up to 65k pixels, and in fp16 that accumulation loses enough
    # precision to make the (small) difference between the two terms unreliable.
    mask = valid.to(torch.float32)
    log_prediction, log_target, clamped = to_log_space(
        prediction.float(), _neutralise_invalid(target.float(), mask), cfg
    )

    # Masked difference. Multiplying by the mask (rather than indexing) keeps
    # the shape static, which is what makes this cheap on GPU and safe under
    # torch.compile. Invalid pixels contribute exactly 0 to both sums.
    diff = (log_prediction - log_target) * mask

    n_valid = mask.sum(dim=(1, 2))
    n_safe = n_valid.clamp(min=1.0)
    sum_d = diff.sum(dim=(1, 2))
    sum_d2 = (diff * diff).sum(dim=(1, 2))

    mean_d = sum_d / n_safe
    mse_log = sum_d2 / n_safe
    # L = mean(d^2) - lambda * mean(d)^2. Non-negative for lambda <= 1, since
    # mean(d^2) >= mean(d)^2 by Jensen; clamped anyway so that float round-off
    # near a perfect fit cannot hand the optimiser a negative loss.
    per_sample = (mse_log - cfg.lambda_si * mean_d * mean_d).clamp(min=0.0)

    batch = int(prediction.shape[0])
    pixels_per_sample = int(mask[0].numel()) if batch else 0
    min_valid = _effective_min_valid_pixels(cfg, pixels_per_sample)
    usable = n_valid >= float(min_valid)
    weights = usable.to(per_sample.dtype)
    per_sample = per_sample * weights
    n_usable = weights.sum()
    loss = per_sample.sum() / n_usable.clamp(min=1.0)

    with torch.no_grad():
        n_usable_int = int(n_usable.item())
        stats: dict[str, Any] = {
            "loss": float(loss.item()),
            "batch_size": batch,
            "usable_samples": n_usable_int,
            "dropped_samples": batch - n_usable_int,
            "lambda_si": cfg.lambda_si,
            "min_valid_pixels": min_valid,
        }
        if batch and pixels_per_sample:
            stats["valid_fraction"] = float(
                n_valid.sum().item() / (batch * pixels_per_sample)
            )
        if n_usable_int:
            stats["mse_log"] = float((mse_log * weights).sum().item() / n_usable_int)
            # The mean log-space offset the loss is choosing to ignore. Not an
            # error to fix -- it is the free scale, and watching it is how you
            # see the model drifting away from the target's units.
            stats["mean_log_offset"] = float((mean_d * weights).sum().item() / n_usable_int)
        if not cfg.predict_log:
            stats["clamped_pixels"] = int((clamped & (mask > 0)).sum().item())

    return LossOutput(loss=loss, per_sample=per_sample, usable=usable, stats=stats)


class ScaleInvariantLoss(nn.Module):
    """:func:`scale_invariant_loss` as a module, so it can hold its config.

    Stateless apart from the config -- it has no parameters and no buffers, and
    is registered as a module purely so a trainer can move it around with the
    rest of the model without special-casing it.
    """

    def __init__(self, cfg: LossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else LossConfig()

    def forward(self, prediction: Tensor, target: Tensor, valid: Tensor) -> LossOutput:
        return scale_invariant_loss(prediction, target, valid, self.cfg)

    def extra_repr(self) -> str:
        return (
            f"lambda_si={self.cfg.lambda_si}, epsilon={self.cfg.epsilon}, "
            f"predict_log={self.cfg.predict_log}, "
            f"min_valid_pixels={self.cfg.min_valid_pixels}"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def align_log_shift(log_prediction: Tensor, log_target: Tensor, valid: Tensor) -> Tensor:
    """The per-sample additive log offset that best matches prediction to target.

    This is the closed-form least-squares solution -- simply the masked mean of
    ``log_target - log_prediction`` -- and it is the one quantity a
    scale-invariant loss deliberately never learns.

    It exists here **only** so that :func:`relative_metrics` can report how good
    the relative structure is without the free scale swamping the number. It is
    fitted on the ground truth, so any metric derived from it is an *upper
    bound* on what the model could achieve at inference time, where no ground
    truth exists. Every metric computed this way is named ``*_aligned`` and the
    returned dict records ``scale_aligned``. Nothing in the training path calls
    this, and it must never be used to produce a metric height.

    Each input may be ``[B, 1, H, W]`` or ``[B, H, W]`` (independently); the
    result is always a ``[B]`` tensor, one shift per sample. A sample with no
    valid pixel gets a shift of 0.
    """
    log_prediction = _squeeze_channel(log_prediction, "log_prediction")
    log_target = _squeeze_channel(log_target, "log_target")
    valid = _squeeze_channel(valid, "valid")
    _check_alignment(log_prediction, log_target, valid)
    mask = valid.to(log_target.dtype)
    n = mask.sum(dim=(1, 2)).clamp(min=1.0)
    return ((log_target - log_prediction) * mask).sum(dim=(1, 2)) / n


@torch.no_grad()
def relative_metrics(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    cfg: LossConfig,
    *,
    delta_thresholds: tuple[float, ...] = (1.25, 1.25**2, 1.25**3),
) -> dict[str, float]:
    """Accuracy of a *relative* height prediction, after removing the free scale.

    All of these are computed in log space over valid pixels only, and all but
    ``si_rmse`` first apply :func:`align_log_shift`, because comparing an
    unaligned scale-invariant prediction to the truth measures the arbitrary
    offset rather than the model.

    Returned keys:

    ``si_rmse``
        Root of the fully scale-invariant loss (``lambda = 1``). Invariant to
        the offset by construction, so it needs no alignment and is the number
        to watch across epochs.
    ``rmse_log_aligned``
        RMSE in log space after alignment.
    ``abs_rel_aligned``
        Mean of ``|pred - target| / target`` in *linear* space after alignment,
        with the epsilon shift applied to both sides.
    ``delta_<t>_aligned``
        Fraction of valid pixels whose ratio ``max(p/t, t/p)`` is below ``t``,
        the standard depth-estimation threshold accuracy.
    ``valid_pixels``, ``scored_samples``
        What the above were computed over. ``si_rmse`` and ``rmse_log_aligned``
        are means over the ``scored_samples`` that have at least one valid
        pixel; the others are pooled over ``valid_pixels``. Zero valid pixels
        yields ``nan`` rather than 0.0 -- an unmeasurable metric is not a
        perfect one.
    ``scale_aligned``
        1.0, recorded numerically so it survives a float-only metrics dict: a
        flag that these numbers used the ground truth to fix the scale.
    """
    prediction = _squeeze_channel(prediction, "prediction")
    target = _squeeze_channel(target, "target")
    valid = _squeeze_channel(valid, "valid")
    _check_alignment(prediction, target, valid)

    mask = valid.to(torch.float32)
    target = _neutralise_invalid(target.float(), mask)
    log_prediction, log_target, _ = to_log_space(prediction.float(), target, cfg)
    total_valid = mask.sum()

    if float(total_valid.item()) == 0.0:
        nan = float("nan")
        metrics = {
            "si_rmse": nan,
            "rmse_log_aligned": nan,
            "abs_rel_aligned": nan,
            "valid_pixels": 0.0,
            "scored_samples": 0.0,
            "scale_aligned": 1.0,
        }
        for threshold in delta_thresholds:
            metrics[f"delta_{threshold:.3f}_aligned"] = nan
        return metrics

    per_sample_valid = mask.sum(dim=(1, 2))
    n = per_sample_valid.clamp(min=1.0)
    # Samples with no valid pixel have nothing to score. Their per-sample terms
    # come out as exactly 0.0, so averaging over them would count an
    # unmeasurable sample as a perfect one; they are excluded instead.
    measured = per_sample_valid > 0

    # Scale-invariant RMSE: the lambda = 1 loss, which needs no alignment.
    diff = (log_prediction - log_target) * mask
    mean_d = diff.sum(dim=(1, 2)) / n
    si_term = (diff * diff).sum(dim=(1, 2)) / n - mean_d * mean_d
    si_rmse = torch.sqrt(si_term.clamp(min=0.0))

    # Aligned comparisons.
    shift = align_log_shift(log_prediction, log_target, mask)
    aligned = log_prediction + shift[:, None, None]
    residual = (aligned - log_target) * mask
    rmse_log = torch.sqrt((residual * residual).sum(dim=(1, 2)) / n)

    # Both sides back through exp of the SAME transform: for h >= 0 this is
    # exactly h + epsilon; for a valid negative h it stays positive, where
    # h + epsilon could be <= 0 and turn abs_rel and the delta ratio into
    # nonsense.
    linear_prediction = torch.exp(aligned)
    linear_target = torch.exp(log_target)
    abs_rel = (
        (linear_prediction - linear_target).abs() / linear_target * mask
    ).sum() / total_valid

    ratio = torch.maximum(
        linear_prediction / linear_target, linear_target / linear_prediction
    )

    metrics = {
        "si_rmse": float(si_rmse[measured].mean().item()),
        "rmse_log_aligned": float(rmse_log[measured].mean().item()),
        "abs_rel_aligned": float(abs_rel.item()),
        "valid_pixels": float(total_valid.item()),
        "scored_samples": float(measured.sum().item()),
        "scale_aligned": 1.0,
    }
    for threshold in delta_thresholds:
        under = ((ratio < threshold).to(torch.float32) * mask).sum() / total_valid
        metrics[f"delta_{threshold:.3f}_aligned"] = float(under.item())
    return metrics
