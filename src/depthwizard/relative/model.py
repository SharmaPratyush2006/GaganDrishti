"""Frozen encoder + DPT-style decoder + one relative-height head.

Architecture
------------
::

    satellite RGB                       [B, 3, 256, 256]
          |
          |  (bilinear resize to the encoder's required input size;
          |   the TARGET is never resized -- only the network input)
          v
    frozen pretrained encoder           [B, 3, S, S]  -> 4 feature maps
          |
          v
    DPT-style decoder                   4 maps -> [B, D, S/4, S/4]
          |
          v
    ONE height regression head          [B, 1, S/4, S/4]
          |
          |  (bilinear upsample to the requested output resolution)
          v
    relative height field               [B, 1, 256, 256]

Concrete tensor shapes, for the two supported encoders at a 256 px tile:

===================  =========================  ==========================
stage                DINOv2-Small (patch 14)    ConvNeXt-Tiny
===================  =========================  ==========================
input tile           [B, 3, 256, 256]           [B, 3, 256, 256]
encoder input        [B, 3, 252, 252]           [B, 3, 256, 256]
pyramid level 0      [B, 384, 72, 72]           [B,  96, 64, 64]
pyramid level 1      [B, 384, 36, 36]           [B, 192, 32, 32]
pyramid level 2      [B, 384, 18, 18]           [B, 384, 16, 16]
pyramid level 3      [B, 384,  9,  9]           [B, 768,  8,  8]
after fusion         [B, 64, 144, 144]          [B, 64, 128, 128]
head output          [B, 1, 144, 144]           [B, 1, 128, 128]
final (upsampled)    [B, 1, 256, 256]           [B, 1, 256, 256]
===================  =========================  ==========================

Each of the four fusion blocks doubles the resolution, so the decoder finishes
one level *finer* than its finest input (144 = 72 x 2, 128 = 64 x 2); the head
runs there and a single bilinear step takes the result to the tile size.

Inference at 512 px (``inference.tile_size``) is a genuine 512 px forward pass,
not an upsampled 256 px one. Everything after the encoder is convolutional, and
the encoder input scales with the tile (:meth:`EncoderSpec.encoder_size_for`;
the ViT interpolates its position embedding to the larger token grid)::

    DINOv2-Small  512 -> 504 (36x36 patches) -> levels 144/72/36/18 -> 288 -> 512
    ConvNeXt-Tiny 512 -> 512                 -> levels 128/64/32/16 -> 256 -> 512

DINOv2 uses 14 px patches, and 256 is not a multiple of 14. Rather than pad or
crop the tile, the encoder input is bilinearly resized 256 -> 252 (18 patches).
This resize applies to the **network input only**: the height target keeps its
original 256 px grid and the prediction is upsampled back to it, so image and
target correspondence is never broken. ConvNeXt-Tiny needs no such resize.

Why the encoder is frozen
-------------------------
Phase 3 is a baseline, and a frozen encoder is what makes it a *stable* one on
an RTX 4050 with 6 GB of VRAM:

* **VRAM.** No gradients, no optimiser moments and no stored activations for
  the backbone -- by far the largest part of the model. On this GPU that is the
  difference between training at batch 8 and not training at all.
* **Trainable parameters.** ~1.5 M trainable (decoder + head) instead of ~24 M.
  An optimiser state that size fits comfortably beside the activations.
* **Stability.** One learning rate, one schedule, no backbone/head LR ratio to
  tune, no risk of destroying pretrained features with an early large step.
* **Overfitting.** A DFC2019 subset is small relative to a 22 M-parameter ViT.
  Freezing the backbone constrains the hypothesis class to "what can be read
  off general-purpose visual features", which is exactly what a baseline should
  measure before anything fancier is justified.

Only the decoder and the height head are trainable;
:meth:`RelativeHeightModel.trainable_parameters` is the single source the
optimiser is built from.

Scope
-----
There is exactly **one** output head, and it predicts a relative height field.
No shadow head, no semantic segmentation head, no uncertainty head, no
classification head, no metric calibration. The output is unitless: metric
calibration is Phase 4 and lives nowhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import ModelConfig

__all__ = [
    "PretrainedWeightsError",
    "EncoderSpec",
    "FrozenEncoder",
    "DPTDecoder",
    "RelativeHeightModel",
    "build_model",
    "ENCODER_SPECS",
]

log = get_logger(__name__)


class PretrainedWeightsError(RuntimeError):
    """Raised when requested pretrained weights cannot be obtained.

    Never swallowed into a random initialisation. If this is raised because the
    machine is offline, the honest options are: supply a local checkpoint via
    ``model.weights``, switch ``model.encoder`` to the other backbone, or use
    the documented fallback in :mod:`depthwizard.relative.fallback` -- which is
    labelled as a fallback everywhere it appears and is not a trained model.
    """


@dataclass(frozen=True)
class EncoderSpec:
    """Static facts about one supported backbone."""

    name: str
    #: The timm model id whose published weights we load.
    timm_name: str
    #: Encoder input edge length for a 256 px training tile.
    input_size: int
    #: Channels of the four pyramid levels handed to the decoder, fine -> coarse.
    feature_channels: tuple[int, int, int, int]
    #: Downsampling factor of each level relative to the encoder input.
    feature_strides: tuple[int, int, int, int]
    is_vit: bool
    #: The encoder input must be a multiple of this (ViT patch size, or the
    #: CNN's total stride).
    size_multiple: int

    def encoder_size_for(self, tile_edge: int) -> int:
        """Encoder input edge for a tile edge: the largest multiple <= the tile.

        256 -> 252 (DINOv2) / 256 (ConvNeXt), the training sizes; 512 -> 504 /
        512 at inference. The encoder therefore always sees the tile at (almost
        exactly) its native resolution, never a 512 tile squeezed down to 256.
        """
        multiple = self.size_multiple
        return max(multiple, (int(tile_edge) // multiple) * multiple)


ENCODER_SPECS: dict[str, EncoderSpec] = {
    # 252 = 18 patches of 14 px. The nearest multiple of 14 below 256.
    "dinov2_small": EncoderSpec(
        name="dinov2_small",
        timm_name="vit_small_patch14_dinov2.lvd142m",
        input_size=252,
        feature_channels=(384, 384, 384, 384),
        feature_strides=(4, 8, 14, 28),
        is_vit=True,
        size_multiple=14,
    ),
    "convnext_tiny": EncoderSpec(
        name="convnext_tiny",
        timm_name="convnext_tiny.fb_in22k_ft_in1k",
        input_size=256,
        feature_channels=(96, 192, 384, 768),
        feature_strides=(4, 8, 16, 32),
        is_vit=False,
        size_multiple=32,
    ),
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _require_timm():
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PretrainedWeightsError(
            "timm is not installed, so no pretrained encoder can be built. "
            "Install it with `pip install timm` (it is declared in the 'ml' extra)."
        ) from exc
    return timm


class FrozenEncoder(nn.Module):
    """A pretrained backbone with all parameters frozen, in permanent eval mode.

    Exposes a uniform interface regardless of backbone family:
    :meth:`forward` returns four feature maps, finest first, with channels and
    strides given by :attr:`spec`.

    For the ViT backbone the four maps come from four transformer blocks, all
    at the same token resolution, and are *reassembled* to four different
    resolutions by the DPT recipe: transposed convolutions upsample the two
    shallow levels, the third is passed through, and the deepest is strided
    down. Those reassembly layers are trainable -- they are part of the
    decoder, not the frozen backbone.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.encoder not in ENCODER_SPECS:  # pragma: no cover - config validated
            raise ValueError(f"unknown encoder {cfg.encoder!r}")
        self.spec = ENCODER_SPECS[cfg.encoder]
        self.cfg = cfg
        self.vit_layers = tuple(cfg.vit_intermediate_layers)

        timm = _require_timm()
        want_download = bool(cfg.pretrained) and cfg.weights is None
        kwargs: dict[str, Any] = {"pretrained": want_download, "num_classes": 0}
        if self.spec.is_vit:
            kwargs["img_size"] = self.spec.input_size
            # Interpolate the position embedding to whatever token grid the
            # input produces, so a 512 px inference tile (36x36 patches) runs
            # at full resolution. Adds no parameters; weights load unchanged.
            kwargs["dynamic_img_size"] = True
        else:
            kwargs["features_only"] = True
            kwargs["out_indices"] = (0, 1, 2, 3)

        try:
            self.backbone = timm.create_model(self.spec.timm_name, **kwargs)
        except Exception as exc:  # network, hub, or checkpoint failure
            if want_download:
                raise PretrainedWeightsError(
                    f"could not obtain pretrained weights for {self.spec.timm_name!r}: "
                    f"{type(exc).__name__}: {exc}\n"
                    "This machine may be offline or the checkpoint may be unavailable. "
                    "Options: (1) set model.weights to a local checkpoint path, "
                    "(2) switch model.encoder to the other supported backbone, "
                    "(3) use the documented frozen-encoder fallback in "
                    "depthwizard.relative.fallback. DepthWizard will NOT substitute "
                    "random weights and call them pretrained."
                ) from exc
            raise

        if cfg.weights is not None:
            self.weights_source = self._load_local_weights(Path(cfg.weights))
        elif want_download:
            self.weights_source = f"timm:{self.spec.timm_name}"
        else:
            # Only reachable with model.allow_random_init, which ModelConfig
            # forces the user to set explicitly.
            self.weights_source = "random_init"
            log.warning(
                "encoder built with RANDOM weights (model.allow_random_init is true). "
                "This is not a pretrained encoder and must not be reported as one.",
                extra={"encoder": self.spec.name},
            )

        self.freeze()

    # -- weights ------------------------------------------------------------

    def _load_local_weights(self, path: Path) -> str:
        if not path.is_file():
            raise PretrainedWeightsError(
                f"model.weights points at {path}, which does not exist. "
                "Supply a real checkpoint or set model.weights: null to download."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            raise PretrainedWeightsError(
                f"{path}: checkpoint is missing {len(missing)} backbone parameter(s), "
                f"e.g. {missing[:3]}. Refusing to leave them randomly initialised "
                "inside an encoder reported as pretrained."
            )
        log.info(
            "loaded local encoder weights",
            extra={"path": str(path), "unexpected_keys": len(unexpected)},
        )
        return f"local:{path}"

    # -- freezing -----------------------------------------------------------

    def freeze(self) -> None:
        """Disable gradients for every backbone parameter and switch to eval."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenEncoder":
        """Keep the backbone in eval mode even when the model is training.

        Overridden because ``model.train()`` recurses into children: without
        this, BatchNorm/LayerNorm running statistics and any stochastic depth
        in the backbone would start updating, which would make a "frozen"
        encoder quietly non-deterministic.
        """
        super().train(False)
        self.backbone.eval()
        return self

    # -- forward ------------------------------------------------------------

    def forward(self, x: Tensor) -> list[Tensor]:
        """``[B, 3, S, S]`` -> four feature maps, finest first.

        Runs under ``torch.no_grad``: the backbone is frozen, so its activation
        graph is pure overhead. Not building it is the single biggest VRAM
        saving in the model.
        """
        with torch.no_grad():
            if self.spec.is_vit:
                features = self.backbone.get_intermediate_layers(
                    x, n=self.vit_layers, reshape=True, norm=True
                )
                return [f.detach() for f in features]
            return [f.detach() for f in self.backbone(x)]


# ---------------------------------------------------------------------------
# DPT-style decoder
# ---------------------------------------------------------------------------


#: Upper bound on GroupNorm groups in the decoder. With the default 64 decoder
#: channels this gives 8 groups of 8 channels.
_MAX_NORM_GROUPS = 8


def _norm_groups(channels: int) -> int:
    """Largest divisor of ``channels`` <= :data:`_MAX_NORM_GROUPS` that still
    leaves at least two channels per group (when ``channels`` allows it).

    Two channels per group guarantees every group normalises over more than one
    value even on a 1x1 feature map, where a one-channel group would have zero
    variance and collapse to its bias.
    """
    limit = min(_MAX_NORM_GROUPS, max(1, channels // 2))
    return max(g for g in range(1, limit + 1) if channels % g == 0)


def _decoder_norm(channels: int) -> nn.GroupNorm:
    """Per-sample normalisation for the decoder.

    GroupNorm rather than BatchNorm: its statistics are computed within each
    sample, so it behaves identically in train and eval, needs no running
    averages, and is well defined at batch size 1 on the coarsest (possibly
    1x1) pyramid level -- where BatchNorm has a single value per channel and
    refuses to train. Small batches are the norm on a 6 GB GPU, which is
    exactly where batch statistics are least reliable anyway.
    """
    return nn.GroupNorm(_norm_groups(channels), channels)


class _ResidualConvUnit(nn.Module):
    """Two 3x3 convolutions with a skip connection, the DPT fusion primitive."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _decoder_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _decoder_norm(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm1(self.conv1(self.act(x)))
        out = self.norm2(self.conv2(self.act(out)))
        return out + x


class _FusionBlock(nn.Module):
    """Add a coarser feature map into a finer one, then upsample by 2.

    This is the DPT fusion stage, trimmed to a single residual unit per input
    instead of the original two. The saving matters on a 6 GB card and the
    baseline does not need the capacity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = _ResidualConvUnit(channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = x + skip
        x = self.refine(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.project(x)


class _ViTReassemble(nn.Module):
    """Turn same-resolution ViT token grids into a four-level pyramid.

    DPT's "reassemble" operation. Level 0 is upsampled 4x by a transposed
    convolution, level 1 by 2x, level 2 is passed through, and level 3 is
    strided down by 2 -- giving the decoder the resolution pyramid a
    convolutional backbone would have produced natively.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for _ in range(4)]
        )
        self.resamplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels, out_channels, 4, stride=4),
                nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            ]
        )

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        return [
            resample(project(feature))
            for feature, project, resample in zip(features, self.projections, self.resamplers)
        ]


class DPTDecoder(nn.Module):
    """Project, reassemble and fuse four encoder levels into one dense map.

    Flow (coarse to fine, four levels indexed 0=finest .. 3=coarsest)::

        level 3 --1x1--> D ch --fuse--> up2 --+
        level 2 --1x1--> D ch --------------> fuse --> up2 --+
        level 1 --1x1--> D ch ---------------------------> fuse --> up2 --+
        level 0 --1x1--> D ch ------------------------------------------> fuse --> up2

    Every level is projected to the same ``decoder_channels`` (D) first, so the
    additions are well defined and no level dominates by width. A coarser map is
    bilinearly matched to its finer partner's size before being added, which
    keeps the fusion correct even for the ViT pyramid where the sizes are not
    exact powers of two of each other.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        decoder_channels: int,
        *,
        reassemble: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError(f"expected 4 encoder levels, got {len(feature_channels)}")
        self.reassemble = reassemble
        if reassemble is None:
            self.projections = nn.ModuleList(
                [nn.Conv2d(c, decoder_channels, 1) for c in feature_channels]
            )
        else:
            # The reassemble module already projects to decoder_channels.
            self.projections = None
        self.fusions = nn.ModuleList([_FusionBlock(decoder_channels) for _ in range(4)])
        self.out_channels = decoder_channels

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if self.reassemble is not None:
            levels = self.reassemble(features)
        else:
            levels = [proj(f) for proj, f in zip(self.projections, features)]

        # Coarsest first; each step folds in the next-finer level.
        x = self.fusions[3](levels[3])
        x = self.fusions[2](x, levels[2])
        x = self.fusions[1](x, levels[1])
        x = self.fusions[0](x, levels[0])
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class RelativeHeightModel(nn.Module):
    """Frozen encoder -> DPT decoder -> one relative-height head.

    The forward pass returns ``[B, 1, H, W]``. That field is **relative and
    unitless**. It is not metres, not elevation and not a DSM. With the default
    ``loss.predict_log``, it is a log-space relative height, defined only up to
    an additive constant (a multiplicative one in linear space) -- which is
    precisely what a scale-invariant loss leaves undetermined, and precisely
    what Phase 4 exists to pin down.
    """

    #: The complete list of prediction heads. Phase 3 has exactly one.
    HEAD_NAMES = ("height",)

    def __init__(self, cfg: ModelConfig, *, output_size: int = 256) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_size = int(output_size)
        self.encoder = FrozenEncoder(cfg)
        spec = self.encoder.spec

        reassemble = (
            _ViTReassemble(spec.feature_channels[0], cfg.decoder_channels)
            if spec.is_vit
            else None
        )
        self.decoder = DPTDecoder(
            spec.feature_channels, cfg.decoder_channels, reassemble=reassemble
        )
        self.head = nn.Sequential(
            nn.Conv2d(cfg.decoder_channels, cfg.head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.head_channels, 1, 1),
        )

    # -- introspection ------------------------------------------------------

    @property
    def weights_source(self) -> str:
        """Where the encoder's parameters came from. Never guessed."""
        return self.encoder.weights_source

    @property
    def encoder_input_size(self) -> int:
        return self.encoder.spec.input_size

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The decoder and head parameters -- everything the optimiser may touch."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def describe(self) -> dict[str, Any]:
        """A compact record of the architecture, embedded in checkpoints and logs."""
        spec = self.encoder.spec
        counts = self.parameter_counts()
        return {
            "encoder": spec.name,
            "encoder_timm_name": spec.timm_name,
            "encoder_input_size": spec.input_size,
            "encoder_frozen": True,
            "weights_source": self.weights_source,
            "feature_channels": list(spec.feature_channels),
            "decoder_channels": self.cfg.decoder_channels,
            "head_channels": self.cfg.head_channels,
            "heads": list(self.HEAD_NAMES),
            "output_channels": 1,
            "output_size": self.output_size,
            "output_units": "relative (unitless) - NOT metres",
            "params_total": counts["total"],
            "params_trainable": counts["trainable"],
            "params_frozen": counts["frozen"],
        }

    # -- forward ------------------------------------------------------------

    def forward(self, x: Tensor, *, output_size: int | None = None) -> Tensor:
        """``[B, 3, H, W]`` -> ``[B, 1, out, out]`` relative height.

        ``output_size`` defaults to the size the model was built for. The input
        is resized only as far as the encoder's size multiple requires
        (:meth:`EncoderSpec.encoder_size_for`: 256 -> 252 for DINOv2, 512 ->
        504), so the encoder works at the tile's own resolution whatever its
        size. The prediction is then resized to the output size. Both resizes
        are on the network path only -- no target is ever touched here.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected an [B, 3, H, W] RGB batch, got shape {tuple(x.shape)}")
        target_size = int(output_size or self.output_size)

        encoder_input = x
        spec = self.encoder.spec
        required = (spec.encoder_size_for(x.shape[-2]), spec.encoder_size_for(x.shape[-1]))
        if tuple(x.shape[-2:]) != required:
            encoder_input = F.interpolate(
                x, size=required, mode="bilinear", align_corners=False
            )

        features = self.encoder(encoder_input)
        fused = self.decoder(features)
        prediction = self.head(fused)
        if prediction.shape[-2:] != (target_size, target_size):
            prediction = F.interpolate(
                prediction,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        return prediction


def build_model(cfg: ModelConfig, *, output_size: int = 256) -> RelativeHeightModel:
    """Build the Phase 3 model and log what it actually is."""
    model = RelativeHeightModel(cfg, output_size=output_size)
    log.info("built relative-height model", extra=model.describe())
    return model
