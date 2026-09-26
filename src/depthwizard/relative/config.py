"""Phase 3 configuration: the relative-height baseline.

Phase 3 has its own config root rather than extending :class:`AppConfig`,
because it configures *training* rather than a scene: a dataset on disk, an
encoder, a loss, an optimiser and a GPU budget. The two roots are loaded by the
same strict machinery (:func:`depthwizard.config.check_keys` /
:func:`depthwizard.config.build_dataclass`), so a typo in either file still
fails loudly instead of being silently dropped.

What this file deliberately does **not** contain
------------------------------------------------
There is no metre scale, no sun geometry, no DEM source and no calibration
block anywhere in Phase 3. The model predicts a **relative** height field;
turning that into metres is Phase 4's job and is configured there.

Nothing here hard-codes a path. :class:`DatasetConfig` states the expected
DFC2019 layout explicitly (root, sub-directories, filename suffixes, the regex
that recovers a scene id) so that pointing DepthWizard at a real DFC2019
download is a config edit, not a code edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from depthwizard.config import ConfigError, build_dataclass, check_keys

__all__ = [
    "SplitConfig",
    "DatasetConfig",
    "ModelConfig",
    "LossConfig",
    "CheckpointConfig",
    "TrainingConfig",
    "InferenceConfig",
    "RelativeConfig",
    "load_relative_config",
    "DEFAULT_RELATIVE_CONFIG_PATH",
    "ENCODERS",
    "SPLIT_MODES",
    "PRECISIONS",
    "SIZE_MISMATCH_POLICIES",
]

DEFAULT_RELATIVE_CONFIG_PATH = Path("configs/phase3.yaml")

#: The one encoder is chosen here. DINOv2-Small is the preferred backbone;
#: ConvNeXt-Tiny is the documented fallback when the DINOv2 checkpoint cannot
#: be fetched. Exactly one is instantiated per run -- never both.
ENCODERS = ("dinov2_small", "convnext_tiny")

SPLIT_MODES = ("per_city_scene", "scene_prefix", "explicit", "random_scene", "random_tile")
PRECISIONS = ("bf16", "fp16", "fp32")
SIZE_MISMATCH_POLICIES = ("error", "resample_height_to_image")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """How scenes are divided into train and validation.

    ``mode`` is the important field, and the default is chosen to be honest
    rather than convenient:

    ``per_city_scene``
        Within each city (the ``city`` group of ``dataset.scene_id_regex``),
        roughly ``val_fraction`` of the scene ids are held out, chosen by a
        shuffle seeded from ``seed`` and the city name. Every view of a held-out
        tile goes to validation, so no ground pixel of a validation tile is
        ever seen in training, and every city contributes to both sides. This
        is the shipped DFC2019 default. It is tile-disjoint, not city-disjoint:
        a validation tile may border a training tile of the same city, so it
        measures generalisation to unseen tiles of the cities trained on.
    ``scene_prefix``
        Validation is every scene whose id starts with one of
        ``val_scene_prefixes``. This is the only mode that gives a genuinely
        **spatially separated** split, because whole geographic areas are held
        out. It needs the real dataset's scene/city naming to be present.
    ``explicit``
        Validation is exactly the scene ids listed in ``val_scene_ids``.
        Also spatially separated, if the caller chose the list that way.
    ``random_scene``
        Scenes (not tiles) are shuffled and split by ``val_fraction``. Weaker:
        two adjacent scenes of the same city can land on opposite sides.
    ``random_tile``
        Tiles are shuffled and split. **This is not a spatial split.** Training
        and validation tiles can be immediate neighbours sharing the same
        buildings, so a validation score from this mode measures interpolation,
        not generalisation to new ground. Provided because it is useful for
        debugging the training loop; :attr:`is_spatially_separated` reports
        False for it, and the trainer logs a warning.
    """

    mode: str = "scene_prefix"
    #: Scene-id prefixes held out for validation, e.g. ``["OMA"]``.
    val_scene_prefixes: Sequence[str] = field(default_factory=tuple)
    #: Exact scene ids held out for validation, used by ``mode: explicit``.
    val_scene_ids: Sequence[str] = field(default_factory=tuple)
    #: Fraction held out by ``per_city_scene`` (per city) and the two random modes.
    val_fraction: float = 0.2
    seed: int = 20260918

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).lower())
        object.__setattr__(self, "val_scene_prefixes", tuple(self.val_scene_prefixes or ()))
        object.__setattr__(self, "val_scene_ids", tuple(self.val_scene_ids or ()))
        object.__setattr__(self, "val_fraction", float(self.val_fraction))
        object.__setattr__(self, "seed", int(self.seed))
        if self.mode not in SPLIT_MODES:
            raise ConfigError(f"split.mode must be one of {SPLIT_MODES}, got {self.mode!r}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ConfigError(f"split.val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.mode == "scene_prefix" and not self.val_scene_prefixes:
            raise ConfigError(
                "split.mode='scene_prefix' needs a non-empty split.val_scene_prefixes. "
                "Set it to the held-out city/scene prefixes of your DFC2019 download "
                "(the prefixes must come from the real data -- do not invent them)."
            )
        if self.mode == "explicit" and not self.val_scene_ids:
            raise ConfigError("split.mode='explicit' needs a non-empty split.val_scene_ids")
        if self.mode == "per_city_scene" and self.val_fraction <= 0.0:
            raise ConfigError(
                "split.mode='per_city_scene' needs split.val_fraction > 0, otherwise "
                "it holds nothing out"
            )

    @property
    def is_spatially_separated(self) -> bool:
        """True only for splits that hold out whole tiles or areas, never parts of one.

        ``per_city_scene`` counts: every view of a held-out tile is held out
        with it. See the class docstring for what it does not separate.
        """
        return self.mode in ("per_city_scene", "scene_prefix", "explicit")


@dataclass(frozen=True)
class DatasetConfig:
    """Where DFC2019 lives on disk and how a pair of files is recognised.

    DepthWizard ships **no** DFC2019 data and invents none. Every field below
    describes a layout the user supplies; :func:`depthwizard.relative.data.discover_pairs`
    raises :class:`~depthwizard.relative.data.DatasetError` naming this config
    when ``root`` does not exist or contains no pairs.

    The default sub-directories and suffixes match the IEEE GRSS DFC2019
    Track-1 (single-view semantic 3D) training release as verified on a real
    download: ``Training-RGB/Track1-RGB/<stem>_RGB.tif`` paired with
    ``Training-Truth/Track1-Truth/<stem>_AGL.tif``. They remain configuration:
    a differently packaged copy only needs a config edit.
    """

    #: Root of the DFC2019 download. Required -- there is no default path.
    root: Path
    #: Relative to ``root``; may contain ``/`` for a nested layout.
    image_subdir: str = "Training-RGB/Track1-RGB"
    height_subdir: str = "Training-Truth/Track1-Truth"
    image_suffix: str = "_RGB.tif"
    height_suffix: str = "_AGL.tif"

    #: Recovers the scene id used for spatial splitting from a file stem.
    #: The default takes a leading ``<CITY>_<NNN>`` tile id, e.g. ``JAX_004``
    #: from ``JAX_004_007_RGB.tif`` -- the trailing ``_007`` is the view, and
    #: every view of one tile must land on the same side of the split. The
    #: optional ``city`` group (``JAX``) is what ``split.mode: per_city_scene``
    #: stratifies by.
    scene_id_regex: str = r"^(?P<scene>(?P<city>[A-Za-z]+)_\d+)"

    #: An extra "no data" value to mask in the height raster, on top of the
    #: raster's own nodata tag and of every non-finite value (which is always
    #: masked). The verified DFC2019 Track-1 AGL files carry NO nodata tag and
    #: NO sentinel: their few invalid pixels are non-finite. -9999 is kept only
    #: as a guard for other packagings; it matches no real AGL height. ``null``
    #: disables it. Whatever is masked is excluded from the loss, never imputed.
    #: There is deliberately no lower bound: small negative AGL values (a DSM
    #: minus a terrain model, both with error) are real measurements.
    height_nodata: float | None = -9999.0
    #: Heights above this are treated as invalid rather than clipped, so a
    #: sentinel that slipped past ``height_nodata`` cannot dominate the loss.
    height_valid_max: float | None = 1000.0

    tile_size: int = 256
    #: Random crops drawn per scene per training epoch.
    train_tiles_per_scene: int = 8
    #: A training crop is redrawn up to this many times to find one with enough
    #: valid target pixels; the last draw is kept regardless, with its mask.
    max_crop_attempts: int = 8
    min_valid_fraction: float = 0.25

    #: What to do when the RGB and height rasters disagree in size. ``error``
    #: (the default) refuses; ``resample_height_to_image`` performs a single
    #: explicit nearest-neighbour resample of the HEIGHT raster onto the image
    #: grid, logs it at WARNING, and records it in the sample metadata. The
    #: image is never resampled independently of the target.
    on_size_mismatch: str = "error"

    #: Scenes smaller than ``tile_size`` are zero-padded and the padding is
    #: marked invalid. False refuses them instead.
    allow_padding: bool = True

    #: ImageNet statistics, matching the pretrained encoders' training.
    normalize_mean: Sequence[float] = (0.485, 0.456, 0.406)
    normalize_std: Sequence[float] = (0.229, 0.224, 0.225)

    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 20260918

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for name in ("image_subdir", "height_subdir", "image_suffix", "height_suffix",
                     "scene_id_regex", "on_size_mismatch"):
            object.__setattr__(self, name, str(getattr(self, name)))
        object.__setattr__(self, "on_size_mismatch", self.on_size_mismatch.lower())
        for name in ("tile_size", "train_tiles_per_scene", "max_crop_attempts", "seed"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "min_valid_fraction", float(self.min_valid_fraction))
        object.__setattr__(self, "allow_padding", bool(self.allow_padding))
        object.__setattr__(self, "normalize_mean", tuple(float(v) for v in self.normalize_mean))
        object.__setattr__(self, "normalize_std", tuple(float(v) for v in self.normalize_std))
        if self.height_nodata is not None:
            object.__setattr__(self, "height_nodata", float(self.height_nodata))
        if self.height_valid_max is not None:
            object.__setattr__(self, "height_valid_max", float(self.height_valid_max))

        if self.tile_size <= 0:
            raise ConfigError(f"dataset.tile_size must be positive, got {self.tile_size}")
        if self.train_tiles_per_scene <= 0:
            raise ConfigError("dataset.train_tiles_per_scene must be positive")
        if self.max_crop_attempts <= 0:
            raise ConfigError("dataset.max_crop_attempts must be positive")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ConfigError("dataset.min_valid_fraction must be in [0, 1]")
        if self.on_size_mismatch not in SIZE_MISMATCH_POLICIES:
            raise ConfigError(
                f"dataset.on_size_mismatch must be one of {SIZE_MISMATCH_POLICIES}, "
                f"got {self.on_size_mismatch!r}"
            )
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ConfigError("dataset.normalize_mean/std must each have 3 entries (RGB)")
        if any(s <= 0 for s in self.normalize_std):
            raise ConfigError("dataset.normalize_std entries must be positive")
        try:
            compiled = re.compile(self.scene_id_regex)
        except re.error as exc:
            raise ConfigError(f"dataset.scene_id_regex is not a valid regex: {exc}") from exc
        if "scene" not in compiled.groupindex:
            raise ConfigError(
                "dataset.scene_id_regex must contain a named group '(?P<scene>...)' "
                "identifying the scene id used for the spatial split"
            )
        if self.split.mode == "per_city_scene" and "city" not in compiled.groupindex:
            raise ConfigError(
                "split.mode='per_city_scene' needs dataset.scene_id_regex to contain a "
                "named group '(?P<city>...)' identifying the city to stratify by"
            )
        if not self.image_suffix or not self.height_suffix:
            raise ConfigError("dataset.image_suffix and dataset.height_suffix must be non-empty")

    @property
    def image_dir(self) -> Path:
        return self.root / self.image_subdir

    @property
    def height_dir(self) -> Path:
        return self.root / self.height_subdir

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "dataset") -> "DatasetConfig":
        data = dict(data)
        check_keys(cls, data, where)
        if "split" in data:
            data["split"] = build_dataclass(SplitConfig, data["split"], f"{where}.split")
        return cls(**data)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Encoder choice, weight provenance and decoder size.

    ``pretrained`` and ``weights`` together decide where the encoder's
    parameters come from, and the answer is always recorded on the built model
    as ``weights_source``. There is no path through this config that produces
    randomly initialised weights while reporting them as pretrained: asking for
    random initialisation requires setting ``allow_random_init: true``, which
    stamps the model ``random_init`` and logs a warning.
    """

    encoder: str = "dinov2_small"
    #: Fetch the published pretrained checkpoint (via timm / HF hub).
    pretrained: bool = True
    #: ``null`` -> download by the encoder's canonical name. A path -> load that
    #: local state dict instead, for offline machines.
    weights: str | None = None
    #: Explicit opt-in to an untrained encoder. Only ever used by unit tests and
    #: offline experiments; never silently.
    allow_random_init: bool = False

    freeze_encoder: bool = True
    decoder_channels: int = 64
    #: Encoder blocks tapped for the DPT pyramid (ViT encoders only; the
    #: ConvNeXt encoder uses its four native stages).
    vit_intermediate_layers: Sequence[int] = (2, 5, 8, 11)
    #: Channels in the 3x3 conv before the 1-channel output.
    head_channels: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder", str(self.encoder).lower())
        object.__setattr__(self, "pretrained", bool(self.pretrained))
        object.__setattr__(self, "allow_random_init", bool(self.allow_random_init))
        object.__setattr__(self, "freeze_encoder", bool(self.freeze_encoder))
        object.__setattr__(self, "decoder_channels", int(self.decoder_channels))
        object.__setattr__(self, "head_channels", int(self.head_channels))
        object.__setattr__(
            self, "vit_intermediate_layers", tuple(int(i) for i in self.vit_intermediate_layers)
        )
        if self.weights is not None:
            object.__setattr__(self, "weights", str(self.weights))

        if self.encoder not in ENCODERS:
            raise ConfigError(f"model.encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if self.decoder_channels <= 0 or self.head_channels <= 0:
            raise ConfigError("model.decoder_channels and model.head_channels must be positive")
        if len(self.vit_intermediate_layers) != 4:
            raise ConfigError(
                "model.vit_intermediate_layers must list exactly 4 blocks, one per "
                f"DPT pyramid level, got {self.vit_intermediate_layers}"
            )
        if not self.pretrained and not self.allow_random_init:
            raise ConfigError(
                "model.pretrained is false but model.allow_random_init is false. "
                "Refusing to build an encoder with random weights by default -- a "
                "random encoder is not a pretrained one and must never be reported "
                "as such. Set model.allow_random_init: true if that is genuinely "
                "what you want (tests do)."
            )
        if not self.freeze_encoder:
            raise ConfigError(
                "model.freeze_encoder must be true in Phase 3. The frozen encoder is "
                "what makes the baseline fit 6 GB of VRAM and stay stable; unfreezing "
                "it is a later experiment, not this phase."
            )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossConfig:
    """Scale-invariant loss parameters. See :mod:`depthwizard.relative.loss`."""

    #: Weight on the mean-error term. 0 -> plain log-space MSE (scale
    #: sensitive), 1 -> fully scale invariant. 0.5 is the usual choice.
    lambda_si: float = 0.5
    #: Added to the target before the logarithm, so log(0) can never occur and
    #: a flat-ground pixel (height exactly 0) stays a usable training signal.
    epsilon: float = 1.0
    #: When true the head's output IS log-height, so no log is taken of the
    #: prediction. See the module docstring for why this is the safe default.
    predict_log: bool = True
    #: Samples with fewer valid target pixels than this contribute nothing.
    #: Capped at the sample's pixel count, so a fully valid sample is never
    #: dropped; it guards against masked-out data, not small tiles.
    min_valid_pixels: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_si", float(self.lambda_si))
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "predict_log", bool(self.predict_log))
        object.__setattr__(self, "min_valid_pixels", int(self.min_valid_pixels))
        if not 0.0 <= self.lambda_si <= 1.0:
            raise ConfigError(f"loss.lambda_si must be in [0, 1], got {self.lambda_si}")
        if self.epsilon <= 0.0:
            raise ConfigError(f"loss.epsilon must be positive, got {self.epsilon}")
        if self.min_valid_pixels < 1:
            raise ConfigError("loss.min_valid_pixels must be >= 1")


# ---------------------------------------------------------------------------
# Checkpointing and training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointConfig:
    """Where checkpoints go and what they carry."""

    dir: Path = Path("data/outputs/phase3/checkpoints")
    every_epoch: bool = True
    #: Also keep ``best.pt``, the lowest validation loss seen.
    keep_best: bool = True
    #: Embed the frozen encoder's weights in every checkpoint. Off by default:
    #: the encoder never changes, and the checkpoint records its name and
    #: weight source, so the full model is exactly reconstructible from the
    #: trainable state plus the config. Turn it on for a self-contained
    #: artefact at the cost of ~90 MB per epoch.
    include_encoder: bool = False
    #: ``null`` -> start from scratch. A path -> resume from that checkpoint.
    resume: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dir", Path(self.dir))
        object.__setattr__(self, "every_epoch", bool(self.every_epoch))
        object.__setattr__(self, "keep_best", bool(self.keep_best))
        object.__setattr__(self, "include_encoder", bool(self.include_encoder))
        if self.resume is not None:
            object.__setattr__(self, "resume", str(self.resume))


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, loader and precision settings, sized for a 6 GB laptop GPU."""

    image_size: int = 256
    batch_size: int = 8
    #: Micro-batches accumulated before each optimiser step. The product with
    #: ``batch_size`` is :attr:`effective_batch_size`.
    gradient_accumulation_steps: int = 2
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float | None = 1.0

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    #: "bf16" | "fp16" | "fp32".
    precision: str = "bf16"
    #: When the requested precision is unsupported: false -> raise, true ->
    #: fall back to fp32 and record the substitution everywhere it is reported.
    allow_precision_fallback: bool = False

    #: "auto" | "cuda" | "cpu".
    device: str = "auto"
    seed: int = 20260918
    log_every: int = 10
    #: Cap on optimiser steps per epoch. ``null`` -> the whole epoch. Used by
    #: the smoke test.
    max_steps_per_epoch: int | None = None
    #: Validation crops scored per epoch. The full validation grid (16 crops per
    #: 1024 px validation pair) is thousands of crops on DFC2019; this caps it
    #: to a fixed subset drawn once from ``dataset.seed`` -- the SAME crops
    #: every epoch, so epochs stay comparable. ``null`` -> the full grid, for a
    #: final validation run. The split itself is unaffected.
    max_val_crops: int | None = 512
    #: Cosine decay of the learning rate across epochs.
    use_scheduler: bool = True

    def __post_init__(self) -> None:
        for name in ("image_size", "batch_size", "gradient_accumulation_steps", "epochs",
                     "num_workers", "seed", "log_every"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("learning_rate", "weight_decay"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("pin_memory", "persistent_workers", "allow_precision_fallback",
                     "use_scheduler"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "precision", str(self.precision).lower())
        object.__setattr__(self, "device", str(self.device).lower())
        if self.grad_clip_norm is not None:
            object.__setattr__(self, "grad_clip_norm", float(self.grad_clip_norm))
        if self.max_steps_per_epoch is not None:
            object.__setattr__(self, "max_steps_per_epoch", int(self.max_steps_per_epoch))
        if self.max_val_crops is not None:
            object.__setattr__(self, "max_val_crops", int(self.max_val_crops))

        if self.image_size <= 0 or self.image_size % 32 != 0:
            raise ConfigError(
                f"training.image_size must be a positive multiple of 32, got {self.image_size}"
            )
        if self.batch_size <= 0:
            raise ConfigError("training.batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("training.gradient_accumulation_steps must be positive")
        if self.epochs <= 0:
            raise ConfigError("training.epochs must be positive")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be positive")
        if self.num_workers < 0:
            raise ConfigError("training.num_workers must be >= 0")
        if self.log_every <= 0:
            raise ConfigError("training.log_every must be positive")
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch <= 0:
            raise ConfigError("training.max_steps_per_epoch must be positive or null")
        if self.max_val_crops is not None and self.max_val_crops <= 0:
            raise ConfigError(
                "training.max_val_crops must be positive, or null for the full "
                "validation grid"
            )
        if self.precision not in PRECISIONS:
            raise ConfigError(
                f"training.precision must be one of {PRECISIONS}, got {self.precision!r}"
            )
        if self.device not in ("auto", "cuda", "cpu"):
            raise ConfigError(
                f"training.device must be 'auto', 'cuda' or 'cpu', got {self.device!r}"
            )
        if self.persistent_workers and self.num_workers == 0:
            raise ConfigError(
                "training.persistent_workers requires training.num_workers > 0; "
                "PyTorch rejects the combination rather than ignoring it"
            )

    @property
    def effective_batch_size(self) -> int:
        """Samples contributing to one optimiser step."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class InferenceConfig:
    """How a trained model is run on new imagery.

    Inference is deliberately allowed to use a larger tile than training. The
    decoder and head are fully convolutional and the encoder input scales with
    the tile (see :meth:`depthwizard.relative.model.EncoderSpec.encoder_size_for`),
    so a 512 px tile is a genuine 512 px forward pass at the same ground sample
    distance -- not a 256 px prediction upsampled. A larger tile also means
    fewer tiles, and each tile's free scale is independent, so fewer seams.
    """

    #: Edge length of an inference tile, in pixels. ``null`` -> the training
    #: tile size. The shipped Phase 3 config sets 512.
    tile_size: int | None = None

    def __post_init__(self) -> None:
        if self.tile_size is not None:
            object.__setattr__(self, "tile_size", int(self.tile_size))
            if self.tile_size <= 0 or self.tile_size % 32 != 0:
                raise ConfigError(
                    "inference.tile_size must be a positive multiple of 32 (the "
                    f"ConvNeXt encoder's total stride) or null, got {self.tile_size}"
                )

    def resolved_tile_size(self, training_tile_size: int) -> int:
        return self.tile_size if self.tile_size is not None else int(training_tile_size)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelativeConfig:
    """Root of ``configs/phase3.yaml``."""

    dataset: DatasetConfig
    run_id: str = "phase3"
    output_dir: Path = Path("data/outputs/phase3")
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.dataset.tile_size != self.training.image_size:
            raise ConfigError(
                f"dataset.tile_size ({self.dataset.tile_size}) must equal "
                f"training.image_size ({self.training.image_size}); the crop taken "
                "from disk is the tensor the model sees, and silently resizing "
                "between them would break image/target correspondence"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelativeConfig":
        data = dict(data)
        check_keys(cls, data, "phase3 config root")
        blocks = {
            "model": ModelConfig,
            "loss": LossConfig,
            "training": TrainingConfig,
            "checkpoint": CheckpointConfig,
            "inference": InferenceConfig,
        }
        for key, block in blocks.items():
            if key in data:
                data[key] = build_dataclass(block, data[key], key)
        if "dataset" not in data:
            raise ConfigError("phase3 config root: missing required key 'dataset'")
        data["dataset"] = DatasetConfig.from_dict(data["dataset"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RelativeConfig":
        """Load ``path``, first applying any ``extends:`` chain it declares.

        A file may start with ``extends: <other.yaml>`` (relative to its own
        directory). The other file is loaded first and this file's keys are
        deep-merged over it: nested mappings merge, everything else replaces.
        This is how the git-ignored ``configs/phase3.local.yaml`` supplies a
        machine-specific ``dataset.root`` without copying -- and drifting from
        -- the shared template. The merged result is validated as strictly as
        a single file, so a typo in an override still fails.
        """
        return cls.from_dict(_load_yaml_chain(Path(path), ()))

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON view, embedded verbatim in every checkpoint."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            if hasattr(value, "__dataclass_fields__"):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            return value

        return {f.name: convert(getattr(self, f.name)) for f in fields(self)}


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"phase 3 config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"phase 3 config file is empty: {path}")
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"config root of {path} must be a mapping, got {type(raw).__name__}"
        )
    return dict(raw)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_chain(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    """Read ``path`` and resolve its ``extends:`` chain, refusing cycles."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"phase 3 config 'extends' chain is circular: {chain}")
    raw = _read_yaml_mapping(path)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw
    if not isinstance(base_name, str) or not base_name:
        raise ConfigError(f"{path}: 'extends' must be a path to another config file")
    base = _load_yaml_chain(path.parent / base_name, (*seen, resolved))
    return _deep_merge(base, raw)


def load_relative_config(
    path: str | Path = DEFAULT_RELATIVE_CONFIG_PATH,
) -> RelativeConfig:
    """Load and validate a Phase 3 config file."""
    return RelativeConfig.from_yaml(path)
