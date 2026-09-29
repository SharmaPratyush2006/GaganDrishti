"""Process one uploaded image with the existing pipeline, and say exactly what came out.

What an arbitrary upload can honestly get, given what Phases 0-7 implement:

* Phase 1: routing to ABSOLUTE / RELATIVE, the metadata summary with provenance,
  and a full pixel decode (a corrupt file fails here).
* Phase 3: the RELATIVE height field (unitless, NOT metres) on one centred
  square tile, when the input has 3+ bands and a Phase 3 checkpoint is present.
  A single-band image is refused rather than copied into three fake RGB bands.
* Everything metric (shadow heights, calibrated AGL, DSM, DTM, nDSM,
  validation) is reported ``not available`` with its reason: Phase 2 and the
  Phase 4a calibration need SUPPLIED building footprints, Phase 4b is
  implemented for the SYNTHETIC fixture only, and an upload has no reference
  data to validate against. Nothing is invented to fill those slots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from depthwizard.demo.common import (
    NOT_MEASURED,
    UNITLESS,
    metadata_summary,
    mode_units,
    not_available,
    report_path,
    url_for,
)
from depthwizard.ingest.metadata import MetadataError, MissingMetadataError, UnsupportedInputError
from depthwizard.ingest.router import SUPPORTED_SUFFIXES, load_scene, route
from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = [
    "DEFAULT_CHECKPOINT",
    "MAX_TILE_PX",
    "InputError",
    "ProcessingError",
    "process_image",
]

log = get_logger(__name__)

#: The shipped Phase 3 checkpoint (git-ignored; trained on DFC2019 JAX + OMA).
#: Override with DEPTHWIZARD_CHECKPOINT.
DEFAULT_CHECKPOINT = Path("data/outputs/phase3/checkpoints/best.pt")
#: Largest Phase 3 tile (the shipped inference.tile_size).
MAX_TILE_PX = 512
#: Smallest tile worth predicting on (one DINOv2 patch is 14 px).
MIN_TILE_PX = 32

_MODEL_CACHE: dict[str, Any] = {}


class InputError(ValueError):
    """The upload itself is unusable. ``kind`` picks the HTTP status."""

    def __init__(self, kind: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kind = kind  # unsupported_format | corrupt_image | missing_metadata | invalid_mode
        self.detail = detail or {}


class ProcessingError(RuntimeError):
    """A pipeline stage failed on a valid input."""


def _checkpoint_path(checkpoint: str | Path | None) -> Path:
    return Path(checkpoint or os.environ.get("DEPTHWIZARD_CHECKPOINT") or DEFAULT_CHECKPOINT)


def _load_model(checkpoint: Path) -> tuple[Any, Any]:
    key = str(checkpoint.resolve())
    if key not in _MODEL_CACHE:
        from depthwizard.relative.inference import load_model_from_checkpoint

        _MODEL_CACHE[key] = load_model_from_checkpoint(checkpoint)
    return _MODEL_CACHE[key]


def _relative_height(image_path: Path, band_count: int, width: int, height: int, out: Path,
                     checkpoint: Path, device: str | None) -> dict[str, Any]:
    """Phase 3 on one centred square tile. RELATIVE, unitless, whatever the input's mode."""
    if band_count < 3:
        return not_available(
            f"the Phase 3 model takes a 3-band RGB image and this input has {band_count} band(s); "
            "copying one band into three would fabricate RGB, so it is not run")
    side = min(width, height, MAX_TILE_PX)
    if side < MIN_TILE_PX:
        return not_available(f"the image is too small for Phase 3 ({width} x {height} px; need >= {MIN_TILE_PX} px)")
    if not checkpoint.is_file():
        return not_available(f"no Phase 3 checkpoint at {checkpoint} (git-ignored; set DEPTHWIZARD_CHECKPOINT)")
    from depthwizard.relative.inference import OUTPUT_UNITS, infer_tile, save_prediction

    model, cfg = _load_model(checkpoint)
    row_off, col_off = (height - side) // 2, (width - side) // 2
    prediction = infer_tile(model, cfg, image_path, row_off=row_off, col_off=col_off, tile_size=side,
                            device=device or "cpu")
    written = save_prediction(prediction, out, image_path.stem)
    values = prediction.relative_height
    return {
        "status": "produced",
        "phase": "3",
        "units": UNITLESS,
        "output_units": OUTPUT_UNITS,
        "log_space": prediction.log_space,
        "tile": {"row_off": row_off, "col_off": col_off, "size_px": side,
                 "rule": f"centred square tile of min(width, height, {MAX_TILE_PX}) px"},
        "band_note": ("bands 1-3 read as R, G, B (the Phase 3 convention); not verified for this file"
                      if band_count > 3 else "3-band input read as R, G, B"),
        "checkpoint": report_path(checkpoint),
        "model_weights_source": prediction.model_description.get("weights_source"),
        "is_fallback_model": bool(prediction.is_fallback or prediction.fallback_kind),
        "value_range": [float(np.min(values)), float(np.max(values))],
        "files": {k: report_path(v) for k, v in written.items()},
        "urls": {k: url_for(v) for k, v in written.items()},
        "note": "Per-tile relative field: NOT metres, not calibrated, not comparable in absolute value across "
                "tiles. No accuracy is claimed for an uploaded image.",
    }


def process_image(
    image_path: str | Path,
    workspace: str | Path,
    *,
    require: str = "auto",
    run_relative: bool = True,
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the existing pipeline on one image and return the response record.

    Args:
        image_path: the input file (already inside ``workspace``/input for uploads).
        workspace: where outputs go; recorded repo-relative when inside the repo.
        require: ``"auto"`` (best available mode, downgrades recorded) or
            ``"absolute"`` (refuse unless the file supports metric heights).
        run_relative: run Phase 3 when the input allows it.

    Raises:
        InputError: unsupported format, corrupt image, missing metadata, bad ``require``.
        ProcessingError: a stage failed on an otherwise valid input.
    """
    if require not in ("auto", "absolute"):
        raise InputError("invalid_mode", f"require must be 'auto' or 'absolute', got {require!r}")
    image_path = Path(image_path)
    workspace = Path(workspace)
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InputError("unsupported_format",
                         f"unsupported file type {image_path.suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")

    # Phase 1: route (metadata only), then decode every pixel so a truncated file fails here.
    try:
        decision = route(image_path, require=Mode.ABSOLUTE if require == "absolute" else None)
    except MissingMetadataError as exc:
        raise InputError("missing_metadata", str(exc), {"missing": list(getattr(exc, "missing", []))}) from exc
    except UnsupportedInputError as exc:
        raise InputError("unsupported_format", str(exc)) from exc
    except MetadataError as exc:
        raise InputError("corrupt_image", f"metadata could not be read: {exc}") from exc
    except Exception as exc:  # rasterio / OpenCV refusing the bytes
        raise InputError("corrupt_image", f"the file could not be opened as an image: {exc}") from exc
    try:
        scene = load_scene(image_path)
    except Exception as exc:
        raise InputError("corrupt_image", f"the pixels could not be decoded: {exc}") from exc
    if scene.array.size == 0:
        raise InputError("corrupt_image", "the image has no pixels")

    mode = decision.mode
    metadata = decision.metadata
    try:
        relative = (_relative_height(image_path, scene.band_count, scene.width, scene.height,
                                     workspace / "relative", _checkpoint_path(checkpoint), device)
                    if run_relative else not_available("Phase 3 was switched off for this request"))
    except Exception as exc:
        raise ProcessingError(f"Phase 3 relative-height inference failed: {exc}") from exc

    footprint_reason = ("SUPPLIED building footprints (DepthWizard does not detect buildings, "
                        "and none are included with an uploaded image)")
    if mode is Mode.ABSOLUTE:
        metric_reason = ("the input is ABSOLUTE-capable (CRS, GSD and sun angles present), but metric heights "
                         f"still require {footprint_reason}. Phase 4b is SYNTHETIC by design (its relative field "
                         "is constructed from the fixture's truth) and is not a metric DSM generator for arbitrary "
                         "uploads; see POST /demo/synthetic for the full metric chain on the synthetic fixture")
    else:
        metric_reason = (f"RELATIVE mode: {decision.reason}. Without a CRS/GSD and both sun angles there is no "
                         "metric scale, so nothing can be expressed in metres")
    products = {
        "relative_height": relative,
        "shadow_heights": not_available(f"Phase 2 requires {footprint_reason}" if mode is Mode.ABSOLUTE
                                        else metric_reason),
        "calibrated_agl": not_available(metric_reason),
        "dsm": not_available(metric_reason),
        "dtm": not_available(metric_reason),
        "ndsm": not_available(metric_reason),
        "mesh": not_available("export the displayed raster from the viewer (Export panel); the relative field "
                              "exports as unitless heights" if relative.get("status") == "produced"
                              else "no height raster was produced for this input"),
    }
    produced = [k for k, v in products.items() if v.get("status") == "produced"]
    # Units of the height values in this response: metres only if a metric product was
    # produced (never on this upload path), unitless for the relative field, else None.
    metric_heights_produced = any(k in produced for k in ("shadow_heights", "calibrated_agl", "dsm", "dtm", "ndsm"))
    if metric_heights_produced or "relative_height" in produced:
        units = mode_units(mode, metric_product=metric_heights_produced)
    else:
        units = None
    viewer_url = None
    if relative.get("status") == "produced" and relative["urls"].get("array"):
        viewer_url = f"http://localhost:5173/?npy={relative['files']['array'].replace(os.sep, '/')}"
    return {
        "mode": mode.value.upper(),
        "units": units,
        "metric_capable_input": mode is Mode.ABSOLUTE,
        "routing": {"loader": decision.loader, "reason": decision.reason, "downgraded": decision.was_downgraded},
        "metadata": metadata_summary(metadata),
        "available_products": produced,
        "products": products,
        "validation": not_available("an uploaded image has no reference heights to validate against"),
        "metrics": {"accuracy": NOT_MEASURED, "uncertainty": NOT_MEASURED},
        "uncertainty": NOT_MEASURED,
        "viewer_url": viewer_url,
        "workspace": report_path(workspace),
    }


def write_response(record: dict[str, Any], workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "process_report.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
