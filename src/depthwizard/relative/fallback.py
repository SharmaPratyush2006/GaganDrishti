"""Two clearly separated substitutes for a trained Phase 3 model.

1. The Phase 3 relative-field fallback: raw PRETRAINED output
--------------------------------------------------------------
If training stalls -- no DFC2019, a loss that will not come down, no usable
checkpoint -- Phase 3 still has to produce a relative field. The fallback is
:class:`PretrainedRelativeFallback`: a **frozen, pretrained** monocular model
whose raw output is passed through unchanged. The default is Depth Anything
V2-Small (``depth-anything/Depth-Anything-V2-Small-hf``, Apache-2.0), whose
backbone is the same DINOv2-Small that Phase 3 trains on, with the DPT head it
was published with.

What its output is, stated as plainly as the model itself states it:

* **relative inverse depth** -- larger means nearer the sensor. Seen from
  nadir, nearer is taller, so it orders heights, but it is *not* a height;
* **unitless**, defined only up to an unknown scale and shift per tile;
* **not metres**, not calibrated, and nothing here converts it;
* **not trained on satellite imagery or DFC2019** -- it is the published model's
  output on imagery it was not trained for.

Weights are fetched with ``transformers`` (the ``ml`` extra) or read from a
local directory. If they cannot be obtained, :class:`PretrainedWeightsError` is
raised. It is never replaced with random weights.

2. The random-initialisation architecture stub (NOT a fallback result)
----------------------------------------------------------------------
:class:`~depthwizard.relative.model.FrozenEncoder` refuses to substitute random
weights for pretrained ones. On a machine with no network access you still want
to exercise the training loop, check shapes, measure VRAM and run the tests.
:func:`build_fallback_model` builds the Phase 3 architecture with a **random**
encoder for exactly that. Its historical name says "fallback", but it is only
an offline architecture/testing stub:

It is not a trained model and it is not a pretrained model. Its encoder is
random noise with the right tensor shapes. Predictions from it are
meaningless, and it is labelled ``random_init`` everywhere -- never
"pretrained", and never as the Phase 3 fallback above.

:func:`assert_not_fallback` is the guard to call before reporting a number as a
result. It raises for a random-init stub rather than warning, because a
warning in a log is not enough to stop a meaningless figure being copied into
a report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import ModelConfig
from depthwizard.relative.model import PretrainedWeightsError, RelativeHeightModel, build_model

__all__ = [
    "FALLBACK_WEIGHTS_SOURCE",
    "RANDOM_INIT_STUB_KIND",
    "PRETRAINED_FALLBACK_KIND",
    "DEFAULT_PRETRAINED_FALLBACK",
    "FallbackModel",
    "PretrainedRelativeFallback",
    "build_fallback_model",
    "build_pretrained_fallback",
    "is_fallback",
    "is_pretrained_fallback",
    "fallback_kind",
    "assert_not_fallback",
]

log = get_logger(__name__)

#: The exact weight source of a random-init stub. Checked by :func:`is_fallback`
#: rather than re-derived, so there is one spelling of it.
FALLBACK_WEIGHTS_SOURCE = "random_init"

#: ``fallback_kind`` of the random-init architecture stub.
RANDOM_INIT_STUB_KIND = "random_init_stub"
#: ``fallback_kind`` of the Phase 3 relative-field fallback.
PRETRAINED_FALLBACK_KIND = "pretrained_raw_output"

#: The published pretrained model used for the relative-field fallback.
DEFAULT_PRETRAINED_FALLBACK = "depth-anything/Depth-Anything-V2-Small-hf"

#: Patch size of the fallback's ViT backbone; its input must be a multiple.
_FALLBACK_PATCH = 14

#: Output record shared by everything the pretrained fallback describes.
_FALLBACK_OUTPUT = (
    "relative inverse depth (raw pretrained output; larger = nearer the sensor) - "
    "unitless, NOT metres, NOT a height"
)


@dataclass(frozen=True)
class FallbackModel:
    """A random-init architecture stub plus the reason it was built.

    Offline architecture/testing only -- NOT the Phase 3 relative-field
    fallback (that is :class:`PretrainedRelativeFallback`). Deliberately not a
    bare ``nn.Module``: a caller has to unpack this, and in unpacking it they
    see :attr:`reason`.
    """

    model: RelativeHeightModel
    #: Why the pretrained path was not taken, in the caller's own words.
    reason: str

    @property
    def is_fallback(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        """The model's description, with the fallback status welded on."""
        description = dict(self.model.describe())
        description.update(
            {
                "is_fallback": True,
                "fallback_kind": RANDOM_INIT_STUB_KIND,
                "fallback_reason": self.reason,
                "encoder_pretrained": False,
                "predictions_are_meaningless": True,
            }
        )
        return description


def build_fallback_model(
    cfg: ModelConfig,
    *,
    reason: str,
    output_size: int = 256,
) -> FallbackModel:
    """Build a shape-correct model whose encoder is randomly initialised.

    An offline architecture/testing stub, labelled ``random_init``. It is NOT
    the Phase 3 relative-field fallback -- use :func:`build_pretrained_fallback`
    for that.

    Args:
        cfg: the model configuration. It is copied with ``pretrained=False`` and
            ``allow_random_init=True``; the caller's own config object is not
            modified, so a fallback build cannot leak into a later real one.
        reason: why the pretrained encoder was unavailable. Required, and
            carried through every description this model produces.
        output_size: prediction edge length, as for
            :func:`~depthwizard.relative.model.build_model`.

    Raises:
        ValueError: if ``reason`` is empty. An unexplained fallback is exactly
            the thing this module exists to prevent.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "build_fallback_model requires a non-empty reason. A model with an "
            "untrained encoder must always carry the explanation of why it was "
            "built that way."
        )

    fallback_cfg = replace(cfg, pretrained=False, allow_random_init=True)
    log.warning(
        "BUILDING A RANDOM-INIT ARCHITECTURE STUB: the encoder is RANDOMLY "
        "INITIALISED and this is NOT a trained or pretrained model, and NOT the "
        "Phase 3 pretrained fallback. Any prediction or metric it produces "
        "describes random weights, not the imagery, and must not be reported "
        "as a result.",
        extra={"encoder": fallback_cfg.encoder, "reason": reason.strip()},
    )
    model = build_model(fallback_cfg, output_size=output_size)
    return FallbackModel(model=model, reason=reason.strip())


def is_fallback(model: Any) -> bool:
    """True when ``model``'s encoder weights are RANDOM (a random-init stub).

    False for :class:`PretrainedRelativeFallback`, whose weights are genuinely
    pretrained -- use :func:`is_pretrained_fallback` or :func:`fallback_kind`.
    """
    if isinstance(model, FallbackModel):
        return True
    return getattr(model, "weights_source", None) == FALLBACK_WEIGHTS_SOURCE


# ---------------------------------------------------------------------------
# The Phase 3 relative-field fallback: raw pretrained output
# ---------------------------------------------------------------------------


def _load_depth_model(source: str) -> nn.Module:
    """Load a pretrained depth model with ``transformers``. Never random."""
    try:
        from transformers import AutoModelForDepthEstimation
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PretrainedWeightsError(
            "transformers is not installed, so the pretrained fallback cannot be "
            "built. Install it with `pip install -e \".[ml]\"`."
        ) from exc
    return AutoModelForDepthEstimation.from_pretrained(source)


class PretrainedRelativeFallback(nn.Module):
    """A frozen pretrained depth model whose RAW output is the relative field.

    ``forward`` takes the same ImageNet-normalised ``[B, 3, H, W]`` tile the
    Phase 3 model takes (Depth Anything was trained on ImageNet-normalised
    input too) and returns ``[B, 1, H, W]``: the model's ``predicted_depth``,
    unscaled and unshifted. The only resampling is the one the backbone
    forces, to a multiple of its 14 px patch (512 -> 504) and back.

    Every parameter is frozen and the model stays in eval mode even under
    ``.train()``: this is inference-only and has nothing to learn.
    """

    #: The complete list of outputs. One relative field, like the Phase 3 model.
    HEAD_NAMES = ("relative_depth",)

    def __init__(self, depth_model: nn.Module, *, weights_source: str, reason: str) -> None:
        super().__init__()
        if weights_source == FALLBACK_WEIGHTS_SOURCE:
            raise ValueError(
                "a pretrained fallback cannot carry the random_init weight source"
            )
        self.depth_model = depth_model
        self.weights_source = weights_source
        self.reason = reason
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)
        self.depth_model.eval()

    def train(self, mode: bool = True) -> "PretrainedRelativeFallback":
        super().train(False)
        self.depth_model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: Tensor, *, output_size: int | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected an [B, 3, H, W] RGB batch, got shape {tuple(x.shape)}")
        height, width = int(x.shape[-2]), int(x.shape[-1])
        size = (
            max(_FALLBACK_PATCH, (height // _FALLBACK_PATCH) * _FALLBACK_PATCH),
            max(_FALLBACK_PATCH, (width // _FALLBACK_PATCH) * _FALLBACK_PATCH),
        )
        model_input = x
        if (height, width) != size:
            model_input = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        depth = self.depth_model(pixel_values=model_input).predicted_depth
        if depth.ndim == 3:
            depth = depth[:, None]
        target = (int(output_size), int(output_size)) if output_size else (height, width)
        if tuple(depth.shape[-2:]) != target:
            depth = F.interpolate(depth, size=target, mode="bilinear", align_corners=False)
        return depth

    def describe(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder": "depth_anything_v2_small",
            "weights_source": self.weights_source,
            "encoder_frozen": True,
            "is_fallback": True,
            "fallback_kind": PRETRAINED_FALLBACK_KIND,
            "fallback_reason": self.reason,
            "encoder_pretrained": True,
            "trained_on_dfc2019": False,
            "heads": list(self.HEAD_NAMES),
            "output_channels": 1,
            "output_quantity": _FALLBACK_OUTPUT,
            "output_units": "relative (unitless) - NOT metres",
            "params_total": total,
            "params_trainable": 0,
            "params_frozen": total,
        }


def build_pretrained_fallback(
    *,
    reason: str,
    source: str | Path = DEFAULT_PRETRAINED_FALLBACK,
    loader: Callable[[str], nn.Module] | None = None,
) -> PretrainedRelativeFallback:
    """Build the Phase 3 relative-field fallback from published pretrained weights.

    Args:
        reason: why the trained model is not being used (e.g. "training
            stalled: val loss flat for 5 epochs"). Required, and carried into
            every description, JSON sidecar and figure.
        source: a Hugging Face model id, or a local directory holding the same
            model for offline machines.
        loader: ``source -> nn.Module`` returning a model whose
            ``forward(pixel_values=...)`` has a ``predicted_depth``. Defaults
            to ``transformers.AutoModelForDepthEstimation.from_pretrained``;
            replaceable so tests can run without a download.

    Raises:
        ValueError: if ``reason`` is empty.
        PretrainedWeightsError: if the weights cannot be obtained. Never
            substituted with random weights.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "build_pretrained_fallback requires a non-empty reason: a fallback "
            "output must always say why the trained model was not used."
        )
    source = str(source)
    try:
        depth_model = (loader or _load_depth_model)(source)
    except PretrainedWeightsError:
        raise
    except Exception as exc:  # network, hub, or checkpoint failure
        raise PretrainedWeightsError(
            f"could not obtain the pretrained fallback weights {source!r}: "
            f"{type(exc).__name__}: {exc}\n"
            "Download them on a connected machine and pass the local directory as "
            "the source. DepthWizard will NOT substitute random weights and call "
            "them a pretrained fallback."
        ) from exc
    weights_source = f"local:{source}" if Path(source).is_dir() else f"hf:{source}"
    log.warning(
        "using the PRETRAINED FALLBACK: raw output of a frozen pretrained depth "
        "model, not the trained Phase 3 model. Output is relative inverse depth, "
        "unitless and NOT metres; the model was not trained on DFC2019.",
        extra={"weights_source": weights_source, "reason": reason.strip()},
    )
    return PretrainedRelativeFallback(
        depth_model, weights_source=weights_source, reason=reason.strip()
    )


def is_pretrained_fallback(model: Any) -> bool:
    """True for the Phase 3 relative-field fallback (raw pretrained output)."""
    return isinstance(model, PretrainedRelativeFallback)


def fallback_kind(model: Any) -> str:
    """``"pretrained_raw_output"``, ``"random_init_stub"``, or ``""`` for a real model."""
    if is_pretrained_fallback(model):
        return PRETRAINED_FALLBACK_KIND
    if is_fallback(model):
        return RANDOM_INIT_STUB_KIND
    return ""


def assert_not_fallback(
    model: RelativeHeightModel | FallbackModel, *, context: str
) -> None:
    """Refuse to continue if ``model`` has an untrained encoder.

    Call this immediately before anything that turns a prediction into a
    reported figure -- an evaluation, an exported raster, a README table.

    Raises:
        RuntimeError: naming ``context``, if the model is a fallback.
    """
    if is_fallback(model):
        reason = model.reason if isinstance(model, FallbackModel) else "random_init encoder"
        raise RuntimeError(
            f"refusing to {context}: this model's encoder is randomly initialised "
            f"({reason}). Its output is not a measurement of anything and must "
            "not be reported. Build the model with pretrained weights, or supply "
            "a local checkpoint via model.weights."
        )
