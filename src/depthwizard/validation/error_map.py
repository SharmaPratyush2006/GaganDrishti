"""Phase 5: the spatial error map -- ``prediction - reference`` per pixel.

Raster
------
:func:`write_error_raster` writes ``error = prediction - reference`` on the
prediction's own grid, as float32, with invalid pixels as the declared nodata
(never 0, which is a real error). Positive = over-prediction.

* **Georeferenced grid** -- written through the Phase 4b
  :func:`~depthwizard.ingest.geotiff.write_cog` and re-checked by
  :func:`~depthwizard.ingest.geotiff.validate_cog`: CRS, transform, width,
  height, bounds, nodata, dtype and every pixel against memory. It opens in QGIS
  as-is.
* **Grid with no CRS** (DFC2019 Track 1 tiles have none) -- written as a plain
  GeoTIFF that preserves exactly what the source had: no CRS, and the source
  (pixel-space) transform. Nothing is invented to make it look georeferenced.
  It is re-read and checked the same way.

Figure
------
:func:`plot_error_map` draws the raster with a diverging colour map centred on
zero (blue = under-prediction, red = over-prediction), invalid pixels in grey,
and a histogram of **every** valid error beside it. The colour scale spans
+/- the configured percentile of |error|; pixels beyond it are drawn in the
end colours (not hidden), counted, and the true min/max are printed. The
figure is a diagnostic; the raster is the product.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from depthwizard.ingest.geotiff import validate_cog, write_cog, write_single_band
from depthwizard.logging_setup import get_logger

__all__ = ["ErrorMapError", "error_array", "write_error_raster", "plot_error_map"]

log = get_logger(__name__)


class ErrorMapError(RuntimeError):
    """The error map could not be written, or did not survive the round trip."""


def error_array(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``prediction - reference`` as float64 where ``valid``, NaN elsewhere."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (prediction.shape == reference.shape == valid.shape):
        raise ErrorMapError(
            f"shape mismatch: prediction {prediction.shape}, reference {reference.shape}, valid {valid.shape}"
        )
    with np.errstate(invalid="ignore"):
        error = np.where(valid, prediction - reference, np.nan)
    if not np.isfinite(error[valid]).all():
        raise ErrorMapError("valid pixels produced a non-finite error; the validity mask is wrong")
    return error


def _verify_plain(path: Path, *, crs: Any, transform: Affine, width: int, height: int,
                  nodata: float, expected: np.ndarray) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read(1)
            record = {
                "path": str(path), "driver": src.driver, "layout": "GeoTIFF (not COG: no CRS to carry)",
                "crs": src.crs.to_string() if src.crs else None,
                "transform": [float(v) for v in tuple(src.transform)[:6]],
                "width": src.width, "height": src.height, "nodata": src.nodata, "dtype": src.dtypes[0],
            }
    failures = []
    if (src_crs := record["crs"]) != (CRS.from_user_input(crs).to_string() if crs else None):
        failures.append(f"CRS {src_crs} != expected {crs}")
    if not np.allclose(record["transform"], tuple(transform)[:6], rtol=0.0, atol=1e-9):
        failures.append(f"transform {record['transform']} != expected {tuple(transform)[:6]}")
    if (record["width"], record["height"]) != (width, height):
        failures.append(f"dimensions {record['width']}x{record['height']} != {width}x{height}")
    if record["nodata"] != nodata:
        failures.append(f"nodata {record['nodata']} != {nodata}")
    written_invalid = data == nodata
    expected_invalid = ~np.isfinite(expected)
    if not np.array_equal(written_invalid, expected_invalid):
        failures.append("nodata mask differs from memory")
    both = ~written_invalid & ~expected_invalid
    max_diff = float(np.abs(data[both].astype(np.float64) - expected[both]).max()) if both.any() else 0.0
    if max_diff > 0.0:
        failures.append(f"pixel values differ from memory by up to {max_diff}")
    record.update(max_abs_diff_vs_memory=max_diff, valid_pixels=int(both.sum()),
                  nodata_pixels=int(written_invalid.sum()), checks_failed=failures, ok=not failures)
    if failures:
        raise ErrorMapError(f"error map {path} failed validation: " + "; ".join(failures))
    return record


def write_error_raster(
    path: str | Path,
    error: np.ndarray,
    *,
    crs: Any,
    transform: Affine | None,
    nodata: float = -9999.0,
    tags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the error array (NaN = invalid) and verify it by reading it back.

    Args:
        path: Destination ``.tif``.
        error: ``(H, W)`` error, NaN where invalid (see :func:`error_array`).
        crs: The source grid's CRS, or None when it has none.
        transform: The source grid's affine transform (identity when it has none).
        nodata: Value written for invalid pixels; must not collide with an error.
        tags: Extra GeoTIFF tags (units, content, SYNTHETIC flag, ...).

    Returns:
        The verification record (CRS, transform, size, nodata, pixel check).

    Raises:
        ErrorMapError: on a nodata collision, a write failure or a failed check.
    """
    path = Path(path)
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 2:
        raise ErrorMapError(f"error must be 2-D, got shape {error.shape}")
    stored = error.astype(np.float32)
    finite = np.isfinite(stored)
    if np.any(stored[finite] == np.float32(nodata)):
        raise ErrorMapError(f"nodata {nodata} collides with a real error value")
    transform = transform if transform is not None else Affine.identity()
    height, width = error.shape
    all_tags = {"CONTENT": "error = prediction - reference (positive = over-prediction)", **dict(tags or {})}

    if crs:
        write_cog(path, stored, crs=crs, transform=transform, nodata=nodata, tags=all_tags)
        record = validate_cog(path, crs=crs, transform=transform, width=width, height=height, nodata=nodata,
                              dtype="float32", expected=stored.astype(np.float64), atol=0.0)
    else:
        data = np.where(finite, stored, np.float32(nodata)).astype(np.float32)
        with warnings.catch_warnings():
            # Deliberate: the source grid has no georeferencing, and neither does this.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            write_single_band(path, data, crs=None, transform=transform, tags=all_tags, nodata=nodata)
        record = _verify_plain(path, crs=None, transform=transform, width=width, height=height,
                               nodata=nodata, expected=stored.astype(np.float64))
    log.info("wrote error map raster", extra={"path": str(path), "crs": record["crs"],
                                              "valid_pixels": record["valid_pixels"]})
    return record


def plot_error_map(
    path: str | Path,
    error: np.ndarray,
    *,
    units: str,
    title: str,
    crs: Any = None,
    transform: Affine | None = None,
    display_percentile: float = 99.0,
    dpi: int = 120,
) -> dict[str, Any] | None:
    """Draw the error map and its histogram. Returns what was drawn, or None.

    None (and no file) when there is no valid pixel to draw.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    path = Path(path)
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(error)
    if not finite.any():
        log.warning("no error map figure drawn: no valid pixel", extra={"path": str(path)})
        return None

    values = error[finite]
    abs_values = np.abs(values)
    max_abs = float(abs_values.max())
    limit = float(np.percentile(abs_values, display_percentile))
    if limit == 0.0:
        limit = max_abs
    all_zero = limit == 0.0
    if all_zero:
        limit = 1.0  # every error is exactly 0; any symmetric scale shows that
    beyond = int((abs_values > limit).sum())

    cmap = colormaps["RdBu_r"].with_extremes(bad="0.72", over="#3b0000", under="#00103b")
    norm = Normalize(vmin=-limit, vmax=limit)

    figure = Figure(figsize=(12.0, 5.4), dpi=dpi, layout="constrained")
    FigureCanvasAgg(figure)
    map_axes, hist_axes = figure.subplots(1, 2, width_ratios=[1.35, 1.0])

    height, width = error.shape
    georeferenced = bool(crs) and transform is not None
    if georeferenced:
        left, top = transform.c, transform.f
        right, bottom = left + transform.a * width, top + transform.e * height
        extent = (left, right, bottom, top)
        crs_label = CRS.from_user_input(crs).to_string()
        map_axes.set_xlabel(f"easting ({crs_label})")
        map_axes.set_ylabel(f"northing ({crs_label})")
        map_axes.ticklabel_format(useOffset=False, style="plain")
    else:
        extent = None
        map_axes.set_xlabel("column (px) - no CRS in source")
        map_axes.set_ylabel("row (px)")
    image = map_axes.imshow(np.ma.masked_invalid(error), cmap=cmap, norm=norm, extent=extent,
                            interpolation="nearest")
    colorbar = figure.colorbar(image, ax=map_axes, extend="both", shrink=0.9)
    colorbar.set_label(f"prediction - reference ({units})\n+ over-prediction / - under-prediction")
    invalid = int((~finite).sum())
    map_axes.legend(handles=[Patch(color="0.72", label=f"nodata / invalid ({invalid:,} px)")],
                    loc="lower right", fontsize=8, framealpha=0.85)
    map_axes.set_title(title, fontsize=10)

    bins = min(200, max(10, int(np.sqrt(values.size))))
    hist_axes.hist(values, bins=bins, color="0.35")
    hist_axes.set_yscale("log")
    hist_axes.axvline(0.0, color="black", linewidth=1.0, label="zero error")
    if not all_zero:
        hist_axes.axvline(-limit, color="tab:blue", linestyle="--", linewidth=0.8, label="colour-scale limits")
        hist_axes.axvline(limit, color="tab:red", linestyle="--", linewidth=0.8)
    hist_axes.set_xlabel(f"prediction - reference ({units})")
    hist_axes.set_ylabel("valid pixels (log scale)")
    hist_axes.set_title("all valid errors (unclipped)", fontsize=10)
    hist_axes.legend(fontsize=8)
    scale_note = (
        "every error is exactly 0" if all_zero
        else f"colour scale: +/-{limit:.4g} {units} (p{display_percentile:g} of |error|)"
    )
    hist_axes.text(
        0.02, 0.98,
        f"valid pixels: {values.size:,}\nmin / max: {values.min():.4g} / {values.max():.4g} {units}\n"
        f"mean (bias): {values.mean():.4g} {units}\n{scale_note}\n"
        f"beyond scale (end colours): {beyond:,} px",
        transform=hist_axes.transAxes, va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "linewidth": 0.4},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    drawn = {
        "path": str(path),
        "display_range": [-limit, limit],
        "display_range_rule": scale_note,
        "pixels_beyond_display_range": beyond,
        "min_error": float(values.min()),
        "max_error": float(values.max()),
        "invalid_pixels": invalid,
        "axes": "map coordinates" if georeferenced else "pixel indices (source has no CRS)",
    }
    log.info("wrote error map figure", extra={"path": str(path)})
    return drawn
