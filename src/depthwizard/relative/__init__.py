"""Phase 3 -- the learned relative-height baseline.

What this package is
--------------------
A monocular height predictor trained on DFC2019: a frozen pretrained encoder, a
DPT-style decoder, one regression head, and a scale-invariant loss.

::

    RGB tile ──▶ frozen encoder ──▶ DPT decoder ──▶ height head ──▶ relative height
                 (DINOv2-S or          4 levels        1 channel      UNITLESS
                  ConvNeXt-T)          fused

* :mod:`depthwizard.relative.config`   -- the Phase 3 config root, loaded with
  the same strict machinery as :mod:`depthwizard.config`.
* :mod:`depthwizard.relative.data`     -- pair discovery, spatial splitting,
  crop geometry, paired reading. **numpy only, no torch.**
* :mod:`depthwizard.relative.dataset`  -- the ``torch.utils.data`` layer on top.
* :mod:`depthwizard.relative.model`    -- frozen encoder + DPT decoder + head.
* :mod:`depthwizard.relative.loss`     -- masked scale-invariant log loss.
* :mod:`depthwizard.relative.fallback` -- the Phase 3 relative-field fallback
  (raw output of a frozen pretrained Depth Anything V2-Small, for when training
  stalls), and the separately labelled random-init architecture stub used only
  for offline testing.
* :mod:`depthwizard.relative.train`    -- the training loop.
* :mod:`depthwizard.relative.inference` -- checkpoint loading, per-tile
  relative-height prediction and the RGB / ground truth / "Relative Height"
  figure. No metric conversion.

The output is relative and unitless
-----------------------------------
A scale-invariant loss cannot determine an absolute scale, and a single image
does not contain one. Every prediction from this package is defined only up to a
multiplicative constant. It is **not metres**. Converting it to metres is
Phase 4's job and no code here does it.

No data, no accuracy
--------------------
DFC2019 is not included in this repository and is not downloaded or simulated by
it. Nothing here reports a real-world accuracy figure, because none has been
measured.

Imports
-------
:mod:`~depthwizard.relative.config` and :mod:`~depthwizard.relative.data` are
imported eagerly and need only numpy, rasterio and PyYAML. Everything that needs
torch is resolved lazily on first attribute access, so::

    from depthwizard.relative import load_relative_config, discover_pairs

works in an environment without torch installed, while::

    from depthwizard.relative import build_model

raises ``ImportError`` naming torch only when it is actually needed.
"""

from __future__ import annotations

from typing import Any

from depthwizard.relative.config import (
    DEFAULT_RELATIVE_CONFIG_PATH,
    ENCODERS,
    PRECISIONS,
    SIZE_MISMATCH_POLICIES,
    SPLIT_MODES,
    CheckpointConfig,
    DatasetConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    RelativeConfig,
    SplitConfig,
    TrainingConfig,
    load_relative_config,
)
from depthwizard.relative.data import (
    CropWindow,
    DatasetError,
    ScenePair,
    TileSample,
    check_pair_registration,
    denormalize_image,
    discover_pairs,
    expected_steps,
    normalize_image,
    plan_train_crops,
    plan_validation_crops,
    raster_shape,
    read_pair_tile,
    sample_train_tile,
    split_pairs,
    summarize_pairs,
)

#: Torch-dependent names, mapped to the module that defines them. Resolved on
#: first access by :func:`__getattr__` so that importing this package does not
#: drag in torch for callers that only want the config or the data layer.
_LAZY: dict[str, str] = {
    # model
    "ENCODER_SPECS": "model",
    "DPTDecoder": "model",
    "EncoderSpec": "model",
    "FrozenEncoder": "model",
    "PretrainedWeightsError": "model",
    "RelativeHeightModel": "model",
    "build_model": "model",
    # loss
    "LossOutput": "loss",
    "MIN_LINEAR_PREDICTION": "loss",
    "ScaleInvariantLoss": "loss",
    "align_log_shift": "loss",
    "relative_metrics": "loss",
    "scale_invariant_loss": "loss",
    "to_log_space": "loss",
    "log_height": "loss",
    # dataset
    "RelativeHeightDataset": "dataset",
    "build_dataloaders": "dataset",
    "build_datasets": "dataset",
    "collate_samples": "dataset",
    "sample_to_tensors": "dataset",
    # fallback
    "FALLBACK_WEIGHTS_SOURCE": "fallback",
    "FallbackModel": "fallback",
    "assert_not_fallback": "fallback",
    "build_fallback_model": "fallback",
    "is_fallback": "fallback",
    "DEFAULT_PRETRAINED_FALLBACK": "fallback",
    "PRETRAINED_FALLBACK_KIND": "fallback",
    "RANDOM_INIT_STUB_KIND": "fallback",
    "PretrainedRelativeFallback": "fallback",
    "build_pretrained_fallback": "fallback",
    "fallback_kind": "fallback",
    "is_pretrained_fallback": "fallback",
    # train
    "EpochResult": "train",
    "PrecisionError": "train",
    "ResolvedPrecision": "train",
    "Trainer": "train",
    "TrainingReport": "train",
    "resolve_device": "train",
    "resolve_precision": "train",
    "set_seed": "train",
    # inference
    "CheckpointError": "inference",
    "OUTPUT_UNITS": "inference",
    "RelativeHeightPrediction": "inference",
    "infer_tile": "inference",
    "load_model_from_checkpoint": "inference",
    "plot_prediction": "inference",
    "predict_relative_height": "inference",
    "read_image_tile": "inference",
    "save_prediction": "inference",
}


def __getattr__(name: str) -> Any:
    """Import a torch-dependent submodule only when one of its names is used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache, so this costs nothing the second time
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # config
    "RelativeConfig",
    "DatasetConfig",
    "SplitConfig",
    "ModelConfig",
    "LossConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "InferenceConfig",
    "load_relative_config",
    "DEFAULT_RELATIVE_CONFIG_PATH",
    "ENCODERS",
    "SPLIT_MODES",
    "PRECISIONS",
    "SIZE_MISMATCH_POLICIES",
    # data
    "DatasetError",
    "ScenePair",
    "CropWindow",
    "TileSample",
    "discover_pairs",
    "split_pairs",
    "raster_shape",
    "plan_train_crops",
    "plan_validation_crops",
    "read_pair_tile",
    "check_pair_registration",
    "sample_train_tile",
    "normalize_image",
    "denormalize_image",
    "summarize_pairs",
    "expected_steps",
    *sorted(_LAZY),
]
