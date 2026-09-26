"""Phase 3 inference: one RGB tile in, one RELATIVE height field out.

What this produces
------------------
The network's raw output for a tile: a ``(tile, tile)`` float32 field that is
**relative and unitless**. With the default ``loss.predict_log`` it is a
log-space relative height, meaningful only up to an additive constant (a
multiplicative one in linear space). Nothing in this module converts it to
metres, and nothing here consults shadows, SRTM, CartoDEM or any other absolute
reference: turning a relative field into metres is Phase 4's job.

Per-tile, not per-scene
-----------------------
Each tile's free offset is determined independently, so two neighbouring tiles'
predictions are not on a common scale. Stitching them into one scene-sized
raster would produce seams that look like height steps but are artefacts of the
loss. This module therefore works one tile at a time and does not stitch.

Honesty rules
-------------
* A model whose encoder is randomly initialised is labelled as such in the
  returned object, in the JSON sidecar and in the figure title. Its output is
  written only because shape/pipeline smoke tests need something to look at.
* A checkpoint trained on a random encoder that did *not* save that encoder is
  refused: the random encoder cannot be rebuilt bit-for-bit, and a decoder fed
  a different random encoder would produce output that corresponds to nothing.
* The ground-truth panel is shown in the dataset's own units, because it *is*
  a measurement; the prediction panel is labelled "Relative Height" and carries
  no unit.

* The pretrained fallback (:class:`~depthwizard.relative.fallback.PretrainedRelativeFallback`)
  is labelled as such everywhere, and its output is described as what it is:
  raw relative inverse depth from a model not trained on DFC2019.

Tiles are ``inference.tile_size`` pixels (512 in the shipped config), run as a
genuine 512 px forward pass even though training uses 256 px tiles.

Usage::

    python -m depthwizard.relative.inference \\
        --checkpoint data/outputs/phase3/checkpoints/best.pt \\
        --image path/to/JAX_004_007_RGB.tif \\
        [--height path/to/JAX_004_007_AGL.tif] [--row 0 --col 0] [--tile-size 512]

    # If training stalled: raw output of the frozen pretrained fallback.
    python -m depthwizard.relative.inference --fallback \\
        --fallback-reason "val loss flat for 5 epochs" \\
        --image path/to/JAX_004_007_RGB.tif
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from rasterio.windows import Window

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import DatasetConfig, RelativeConfig
from depthwizard.relative.data import (
    CropWindow,
    ScenePair,
    _read_rgb_window,
    denormalize_image,
    normalize_image,
    open_raster,
    read_pair_tile,
)
from depthwizard.relative.fallback import (
    DEFAULT_PRETRAINED_FALLBACK,
    FALLBACK_WEIGHTS_SOURCE,
    PRETRAINED_FALLBACK_KIND,
    build_pretrained_fallback,
    fallback_kind,
    is_fallback,
)
from depthwizard.relative.model import RelativeHeightModel, build_model
from depthwizard.relative.train import resolve_device, resolve_precision

__all__ = [
    "OUTPUT_UNITS",
    "CheckpointError",
    "RelativeHeightPrediction",
    "load_model_from_checkpoint",
    "predict_relative_height",
    "read_image_tile",
    "infer_tile",
    "plot_prediction",
    "save_prediction",
    "main",
]

log = get_logger(__name__)

OUTPUT_UNITS = "relative (unitless) - NOT metres"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be turned back into the model it came from."""


@dataclass
class RelativeHeightPrediction:
    """One tile's prediction, plus everything needed to display it honestly."""

    #: ``(H, W)`` float32 raw network output. Relative, unitless, NOT metres.
    relative_height: np.ndarray
    #: True when ``relative_height`` is a log-space field (``loss.predict_log``).
    log_space: bool
    #: ``(H, W, 3)`` float32 in [0, 1], the input tile de-normalised for display.
    image_rgb: np.ndarray
    #: ``(H, W)`` float32 ground truth in the dataset's own units, NaN where
    #: invalid. None when no height raster was supplied.
    ground_truth: np.ndarray | None = None
    #: ``(H, W)`` bool validity of ``ground_truth``; None with it.
    valid: np.ndarray | None = None
    #: Tile bookkeeping: source paths, crop origin, transform, CRS.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: :meth:`RelativeHeightModel.describe` of the model that produced this.
    model_description: dict[str, Any] = field(default_factory=dict)
    #: True when the encoder's weights were random. Output is then meaningless.
    is_fallback: bool = False
    #: ``""`` for the trained model, ``"pretrained_raw_output"`` for the Phase 3
    #: pretrained fallback, ``"random_init_stub"`` for a random encoder.
    fallback_kind: str = ""

    @property
    def units(self) -> str:
        return OUTPUT_UNITS

    def _fallback_note(self) -> str:
        if self.is_fallback:
            return (
                "encoder weights are RANDOM; this output describes random weights, "
                "not the imagery, and must not be reported as a result"
            )
        if self.fallback_kind == PRETRAINED_FALLBACK_KIND:
            return (
                "PRETRAINED FALLBACK: raw output of a frozen pretrained depth model "
                f"({self.model_description.get('weights_source', '?')}), not the "
                "trained Phase 3 model. It is relative inverse depth (larger = "
                "nearer the sensor), unitless, NOT metres, and the model was not "
                "trained on DFC2019."
            )
        return ""

    def to_json(self) -> dict[str, Any]:
        """Sidecar record. Contains no pixel data and no accuracy claim."""
        pretrained_fallback = self.fallback_kind == PRETRAINED_FALLBACK_KIND
        return {
            "output": (
                "relative inverse depth (pretrained fallback)"
                if pretrained_fallback
                else "relative height"
            ),
            "output_units": OUTPUT_UNITS,
            "log_space": self.log_space,
            "is_fallback_model": self.is_fallback or pretrained_fallback,
            "fallback_kind": self.fallback_kind,
            "fallback_note": self._fallback_note(),
            "has_ground_truth": self.ground_truth is not None,
            "shape": list(self.relative_height.shape),
            "metadata": self.metadata,
            "model": self.model_description,
            "note": (
                "Per-tile relative field. Not metres, not calibrated, not "
                "comparable in absolute value across tiles. No accuracy is claimed."
            ),
        }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RelativeHeightModel, RelativeConfig]:
    """Rebuild the trained model and its config from a Phase 3 checkpoint.

    * Checkpoints that saved the encoder (``include_encoder``) are restored in
      full, without any download; the encoder's recorded weight source is
      carried over verbatim, so a random-init encoder stays labelled random.
    * Checkpoints without the encoder rebuild it from ``model.pretrained`` /
      ``model.weights`` exactly as training did, then load decoder and head.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        CheckpointError: if the checkpoint does not match the model its own
            config describes, or it omits a random encoder that cannot be
            reconstructed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model_state", "config", "model_description"):
        if key not in payload:
            raise CheckpointError(f"{path}: not a Phase 3 checkpoint (no {key!r})")

    cfg = RelativeConfig.from_dict(payload["config"])
    recorded = payload["model_description"]
    includes_encoder = bool(payload.get("includes_encoder", False))
    recorded_source = recorded.get("weights_source")

    if recorded.get("encoder") != cfg.model.encoder:
        raise CheckpointError(
            f"{path}: checkpoint records encoder {recorded.get('encoder')!r} but its "
            f"config says {cfg.model.encoder!r}"
        )

    if includes_encoder:
        # Every parameter comes from the checkpoint, so build the skeleton
        # without downloading anything, then restore the recorded provenance.
        skeleton_cfg = replace(cfg.model, pretrained=False, weights=None, allow_random_init=True)
        model = build_model(skeleton_cfg, output_size=cfg.training.image_size)
        _load_state(model, payload["model_state"], includes_encoder=True, source=path)
        model.encoder.weights_source = recorded_source
        log.info(
            "encoder restored from checkpoint",
            extra={"path": str(path), "weights_source": recorded_source},
        )
    else:
        if recorded_source == FALLBACK_WEIGHTS_SOURCE:
            raise CheckpointError(
                f"{path}: this checkpoint was trained on a RANDOMLY INITIALISED "
                "encoder and did not save it (checkpoint.include_encoder was false). "
                "That encoder cannot be reconstructed, and pairing the decoder with "
                "a different random encoder would produce output that corresponds "
                "to nothing. Re-run with checkpoint.include_encoder: true."
            )
        model = build_model(cfg.model, output_size=cfg.training.image_size)
        _load_state(model, payload["model_state"], includes_encoder=False, source=path)
        if model.weights_source != recorded_source:
            log.warning(
                "encoder weight source differs from the one recorded at training time",
                extra={"recorded": recorded_source, "now": model.weights_source},
            )

    model.eval()
    return model, cfg


def _load_state(
    model: RelativeHeightModel,
    state: dict[str, Any],
    *,
    includes_encoder: bool,
    source: Path,
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not includes_encoder:
        missing = [key for key in missing if not key.startswith("encoder.")]
    if missing or unexpected:
        raise CheckpointError(
            f"{source}: checkpoint does not match this model. "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_relative_height(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    autocast_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Run the model on one normalised ``(3, H, W)`` tile. Returns ``(H, W)`` float32.

    The result is the raw network output: relative, unitless, NOT metres. A
    model that takes ``output_size`` (the Phase 3 model does) is asked for the
    tile's own size, so a 512 px tile yields a 512 px field from a 512 px
    forward pass rather than the model's 256 px training default.
    """
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) tile, got shape {image_chw.shape}")
    device = torch.device(device)
    model.eval()
    batch = torch.from_numpy(np.ascontiguousarray(image_chw, dtype=np.float32))[None].to(device)
    kwargs: dict[str, Any] = {}
    if "output_size" in inspect.signature(model.forward).parameters:
        if image_chw.shape[1] != image_chw.shape[2]:
            raise ValueError(f"expected a square tile, got shape {image_chw.shape}")
        kwargs["output_size"] = int(image_chw.shape[-1])
    if autocast_dtype is None:
        output = model(batch, **kwargs)
    else:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            output = model(batch, **kwargs)
    if tuple(output.shape[-2:]) != tuple(image_chw.shape[-2:]):
        raise ValueError(
            f"model returned a {tuple(output.shape[-2:])} field for a "
            f"{tuple(image_chw.shape[-2:])} tile; refusing to resample it silently"
        )
    field_ = output.float()[0, 0].cpu().numpy()
    if not np.isfinite(field_).all():
        raise FloatingPointError(
            "model produced non-finite relative height values; refusing to return them"
        )
    return field_.astype(np.float32)


def read_image_tile(
    image_path: str | Path, window: CropWindow, cfg: DatasetConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one RGB tile without a height raster. Returns ``(image_chw, metadata)``."""
    image_path = Path(image_path)
    rio_window = Window(window.col_off, window.row_off, window.size, window.size)
    with open_raster(image_path) as src:
        rgb = _read_rgb_window(src, rio_window)
        transform = src.window_transform(rio_window)
        metadata = {
            "georeferenced": bool(src.crs),
            "image_path": str(image_path),
            "row_off": int(window.row_off),
            "col_off": int(window.col_off),
            "tile_size": int(window.size),
            "source_height": int(src.height),
            "source_width": int(src.width),
            "transform": [float(v) for v in tuple(transform)[:6]],
            "crs": src.crs.to_string() if src.crs else "",
        }
    return normalize_image(rgb, cfg.normalize_mean, cfg.normalize_std), metadata


def infer_tile(
    model: RelativeHeightModel,
    cfg: RelativeConfig,
    image_path: str | Path,
    *,
    height_path: str | Path | None = None,
    row_off: int = 0,
    col_off: int = 0,
    device: torch.device | str | None = None,
    tile_size: int | None = None,
) -> RelativeHeightPrediction:
    """Predict the relative height of one tile of ``image_path``.

    The tile is ``tile_size`` pixels square at ``(row_off, col_off)``;
    ``tile_size`` defaults to ``cfg.inference.tile_size`` (512 in the shipped
    config), falling back to the training tile size when that is null. When
    ``height_path`` is given, the same window is read from it through the
    training reader, so the ground truth shown is masked exactly as training
    masked it.
    """
    device = resolve_device(cfg.training.device) if device is None else torch.device(device)
    precision = resolve_precision(cfg.training, device)
    model.to(device)

    size = (
        int(tile_size)
        if tile_size is not None
        else cfg.inference.resolved_tile_size(cfg.dataset.tile_size)
    )
    window = CropWindow(row_off=int(row_off), col_off=int(col_off), size=size)
    ground_truth = valid = None
    if height_path is not None:
        pair = ScenePair(
            stem=Path(image_path).stem,
            scene_id=Path(image_path).stem,
            image_path=Path(image_path),
            height_path=Path(height_path),
        )
        sample = read_pair_tile(pair, window, cfg.dataset)
        image_chw, metadata = sample.image, dict(sample.metadata)
        valid = sample.valid
        ground_truth = np.where(valid, sample.height, np.nan).astype(np.float32)
    else:
        image_chw, metadata = read_image_tile(image_path, window, cfg.dataset)

    relative = predict_relative_height(
        model, image_chw, device=device, autocast_dtype=precision.autocast_dtype
    )
    metadata["device"] = str(device)
    metadata["precision"] = precision.to_dict()

    fallback = is_fallback(model)
    kind = fallback_kind(model)
    if fallback:
        log.warning(
            "inference ran on a model with a RANDOM encoder; the output is a "
            "pipeline smoke test, not a prediction of anything",
            extra={"image": str(image_path)},
        )
    return RelativeHeightPrediction(
        relative_height=relative,
        # The pretrained fallback's raw output is linear inverse depth; only
        # the trained Phase 3 head predicts in log space.
        log_space=cfg.loss.predict_log and kind != PRETRAINED_FALLBACK_KIND,
        image_rgb=denormalize_image(image_chw, cfg.dataset.normalize_mean, cfg.dataset.normalize_std),
        ground_truth=ground_truth,
        valid=valid,
        metadata=metadata,
        model_description=model.describe(),
        is_fallback=fallback,
        fallback_kind=kind,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _display_range(values: np.ndarray) -> tuple[float, float]:
    """2nd-98th percentile of the finite values, so one outlier cannot wash out the map."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [2.0, 98.0])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def plot_prediction(prediction: RelativeHeightPrediction, path: str | Path) -> Path:
    """Write a figure: input RGB | ground truth (if any) | predicted Relative Height."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_gt = prediction.ground_truth is not None
    panels = 3 if has_gt else 2
    figure, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.8), constrained_layout=True)

    axes[0].imshow(prediction.image_rgb)
    axes[0].set_title("Input RGB")

    if has_gt:
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("0.6")  # invalid / nodata pixels shown grey, never as a height
        low, high = _display_range(prediction.ground_truth)
        shown = axes[1].imshow(
            np.ma.masked_invalid(prediction.ground_truth), cmap=cmap, vmin=low, vmax=high
        )
        axes[1].set_title("Ground truth height\n(dataset units; grey = invalid)")
        figure.colorbar(shown, ax=axes[1], shrink=0.8)

    low, high = _display_range(prediction.relative_height)
    shown = axes[-1].imshow(prediction.relative_height, cmap="viridis", vmin=low, vmax=high)
    space = "log-space, " if prediction.log_space else ""
    if prediction.fallback_kind == PRETRAINED_FALLBACK_KIND:
        axes[-1].set_title(
            "Relative field - pretrained fallback\n"
            "(inverse depth, unitless - NOT metres)"
        )
    else:
        axes[-1].set_title(f"Relative Height (predicted)\n({space}unitless - NOT metres)")
    figure.colorbar(shown, ax=axes[-1], shrink=0.8, label="relative (unitless)")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    title = f"{prediction.model_description.get('encoder', '?')} " \
        f"[{prediction.model_description.get('weights_source', '?')}]"
    if prediction.is_fallback:
        title = "RANDOM ENCODER - NOT A TRAINED MODEL - OUTPUT IS MEANINGLESS\n" + title
    elif prediction.fallback_kind == PRETRAINED_FALLBACK_KIND:
        title = (
            "PRETRAINED FALLBACK (raw output, not trained on DFC2019) - NOT the "
            "Phase 3 trained model\n" + title
        )
    figure.suptitle(
        title,
        color="red" if prediction.is_fallback else (
            "darkorange" if prediction.fallback_kind else "black"
        ),
    )

    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def save_prediction(
    prediction: RelativeHeightPrediction, out_dir: str | Path, name: str
) -> dict[str, Path]:
    """Write ``<name>_relative_height.npy``, ``<name>.png`` and ``<name>.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = out_dir / f"{name}_relative_height.npy"
    np.save(array_path, prediction.relative_height)
    figure_path = plot_prediction(prediction, out_dir / f"{name}.png")
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(prediction.to_json(), indent=2), encoding="utf-8")
    return {"array": array_path, "figure": figure_path, "json": json_path}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    """``python -m depthwizard.relative.inference --checkpoint ... --image ...``."""
    import argparse

    from depthwizard.logging_setup import setup_logging

    parser = argparse.ArgumentParser(
        description="Predict a RELATIVE (unitless, not metres) height field for one tile"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="a trained Phase 3 checkpoint")
    source.add_argument(
        "--fallback",
        action="store_true",
        help="use the PRETRAINED FALLBACK (raw Depth Anything V2-Small output) "
        "instead of a trained checkpoint, e.g. when training stalled",
    )
    parser.add_argument(
        "--fallback-reason",
        default=None,
        help="why the trained model is not used (required with --fallback)",
    )
    parser.add_argument(
        "--fallback-source",
        default=DEFAULT_PRETRAINED_FALLBACK,
        help="Hugging Face id or local directory of the pretrained fallback model",
    )
    parser.add_argument(
        "--config",
        default="configs/phase3.yaml",
        help="Phase 3 config for --fallback (tiling and normalisation)",
    )
    parser.add_argument("--image", required=True, help="RGB GeoTIFF")
    parser.add_argument("--height", default=None, help="optional ground-truth height GeoTIFF")
    parser.add_argument("--row", type=int, default=0, help="tile row offset (pixels)")
    parser.add_argument("--col", type=int, default=0, help="tile column offset (pixels)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="tile edge in pixels (default: inference.tile_size from the config, 512 shipped)",
    )
    parser.add_argument("--device", default=None, help="auto | cuda | cpu (default: config)")
    parser.add_argument("--out", default=None, help="output dir (default: <output_dir>/inference)")
    parser.add_argument("--log-format", choices=("text", "json"), default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging({"level": "INFO", "format": args.log_format})
    device = resolve_device(args.device) if args.device else None
    if args.fallback:
        if not args.fallback_reason:
            parser.error("--fallback requires --fallback-reason")
        from depthwizard.relative.config import load_relative_config

        cfg = load_relative_config(args.config)
        model = build_pretrained_fallback(
            reason=args.fallback_reason, source=args.fallback_source
        )
    else:
        model, cfg = load_model_from_checkpoint(args.checkpoint)
    prediction = infer_tile(
        model,
        cfg,
        args.image,
        height_path=args.height,
        row_off=args.row,
        col_off=args.col,
        device=device,
        tile_size=args.tile_size,
    )
    out_dir = Path(args.out) if args.out else cfg.output_dir / "inference"
    name = f"{Path(args.image).stem}_r{args.row}_c{args.col}"
    written = save_prediction(prediction, out_dir, name)

    print(f"output units : {OUTPUT_UNITS}")
    if prediction.is_fallback:
        print("WARNING      : encoder weights are RANDOM; this output is meaningless")
    elif prediction.fallback_kind == PRETRAINED_FALLBACK_KIND:
        print(
            "NOTE         : PRETRAINED FALLBACK - raw relative inverse depth from "
            f"{model.weights_source}, not the trained Phase 3 model"
        )
    for kind, written_path in written.items():
        print(f"{kind:<13}: {written_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
