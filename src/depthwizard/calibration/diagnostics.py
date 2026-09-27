"""Error accounting and report writing for Phase 4a.

Every number here is computed from arrays the run actually produced. An
accumulator that saw no data reports :data:`~depthwizard.validation.evaluation.NOT_MEASURED`
for its metrics, never zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from depthwizard.validation.evaluation import NOT_MEASURED

__all__ = [
    "ErrorAccumulator",
    "HEIGHT_BINS_M",
    "error_sums",
    "dense_error_sums",
    "distribution",
    "height_distribution",
    "binned_errors",
    "count_reasons",
    "to_jsonable",
    "write_json",
]

#: Bins for "error versus true building height", metres (last bin open).
HEIGHT_BINS_M: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 40.0)


@dataclass
class ErrorAccumulator:
    """Pooled MAE / RMSE / bias over any number of ``predicted - true`` arrays."""

    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_err: float = 0.0
    tiles: set[str] = field(default_factory=set)

    def add(self, predicted: np.ndarray, true: np.ndarray, tile_id: str | None = None) -> None:
        errors = np.asarray(predicted, dtype=np.float64) - np.asarray(true, dtype=np.float64)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            return
        self.count += int(errors.size)
        self.sum_abs += float(np.abs(errors).sum())
        self.sum_sq += float((errors**2).sum())
        self.sum_err += float(errors.sum())
        if tile_id is not None:
            self.tiles.add(tile_id)

    def add_sums(self, sums: dict[str, Any], tile_id: str | None = None) -> None:
        """Merge a :func:`error_sums` record (additive, so tiles pool exactly)."""
        if not sums or not sums.get("n"):
            return
        self.count += int(sums["n"])
        self.sum_abs += float(sums["sum_abs"])
        self.sum_sq += float(sums["sum_sq"])
        self.sum_err += float(sums["sum_err"])
        if tile_id is not None:
            self.tiles.add(tile_id)

    @property
    def mae(self) -> float | None:
        return self.sum_abs / self.count if self.count else None

    @property
    def rmse(self) -> float | None:
        return math.sqrt(self.sum_sq / self.count) if self.count else None

    @property
    def bias(self) -> float | None:
        return self.sum_err / self.count if self.count else None

    def to_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"n": 0, "tiles": 0, "mae_m": NOT_MEASURED, "rmse_m": NOT_MEASURED,
                    "bias_m": NOT_MEASURED}
        record: dict[str, Any] = {"n": self.count}
        if self.tiles:
            record["tiles"] = len(self.tiles)
        record.update({"mae_m": self.mae, "rmse_m": self.rmse, "bias_m": self.bias})
        return record


def error_sums(predicted: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    """Additive error sufficient statistics: n, sum|e|, sum e^2, sum e."""
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    return {
        "n": int(errors.size),
        "sum_abs": float(np.abs(errors).sum()),
        "sum_sq": float((errors**2).sum()),
        "sum_err": float(errors.sum()),
    }


def dense_error_sums(
    predicted: np.ndarray, true: np.ndarray, eval_mask: np.ndarray, building_px: np.ndarray
) -> dict[str, dict[str, Any]]:
    """:func:`error_sums` for building, ground (non-building) and all evaluated pixels."""
    return {
        "building": error_sums(predicted[eval_mask & building_px], true[eval_mask & building_px]),
        "ground": error_sums(predicted[eval_mask & ~building_px], true[eval_mask & ~building_px]),
        "all": error_sums(predicted[eval_mask], true[eval_mask]),
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Count, median, mean, p10/p25/p75/p90/p95, min, max of finite values."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90, 95])
    return {
        "count": int(v.size), "median": float(q[2]), "mean": float(v.mean()),
        "p10": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p90": float(q[4]),
        "p95": float(q[5]), "min": float(v.min()), "max": float(v.max()),
    }


def height_distribution(heights: Sequence[float]) -> dict[str, Any]:
    """:func:`distribution` plus counts above 10 / 20 / 30 m."""
    v = np.asarray([h for h in heights if h is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    out = distribution(v)
    out.update({"count_gt_10m": int((v > 10).sum()), "count_gt_20m": int((v > 20).sum()),
                "count_gt_30m": int((v > 30).sum())})
    return out


def binned_errors(
    predicted: Sequence[float], true: Sequence[float], bins: Sequence[float] = HEIGHT_BINS_M
) -> list[dict[str, Any]]:
    """MAE / RMSE / bias of ``predicted - true`` per bin of the **true** value."""
    p = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    edges = list(bins) + [math.inf]
    rows: list[dict[str, Any]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        sel = (t >= low) & (t < high)
        acc = ErrorAccumulator()
        acc.add(p[sel], t[sel])
        rows.append({"true_height_m": f"[{low:g}, {high:g})", **acc.to_dict()})
    return rows


def count_reasons(reasons: Iterable[str | None]) -> dict[str, int]:
    """Histogram of rejection reasons, keyed by the part before any ':'."""
    counts: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        key = reason.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def plot_tile_diagnostic(
    path: str | Path,
    *,
    title: str,
    rgb: np.ndarray,
    building_mask: np.ndarray,
    shadow_mask: np.ndarray,
    reference_masks: Sequence[np.ndarray],
    rejected_masks: Sequence[np.ndarray],
    sun_azimuth_deg: float | None,
    ray_segments: Sequence[tuple[float, float, float, float]],
) -> Path:
    """Two panels: RGB with rays and sun arrow | CLS buildings, shadow, references.

    ``ray_segments`` are ``(row0, col0, row1, col1)`` from launch pixel to the
    last shadow pixel of each hit ray.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from depthwizard.physics.sun import shadow_direction_pixels

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)

    axes[0].imshow(rgb)
    for r0, c0, r1, c1 in ray_segments:
        axes[0].plot([c0, c1], [r0, r1], color="yellow", linewidth=0.5, alpha=0.7)
    if sun_azimuth_deg is not None:
        # Arrow points TOWARDS the sun (opposite to the shadow direction).
        d_row, d_col = shadow_direction_pixels(sun_azimuth_deg)
        n = rgb.shape[0]
        axes[0].annotate("", xy=(n * 0.12 - d_col * 40, n * 0.12 - d_row * 40), xytext=(n * 0.12, n * 0.12),
                         arrowprops=dict(color="orange", width=2.5, headwidth=9))
        axes[0].text(n * 0.02, n * 0.97, f"sun az {sun_azimuth_deg:.0f}° (arrow = towards sun)",
                     color="orange", fontsize=8, backgroundcolor="black")
    axes[0].set_title("RGB + measured shadow rays")

    overlay = np.zeros(building_mask.shape + (3,), dtype=np.float32)
    overlay[shadow_mask] = (0.25, 0.25, 0.6)
    overlay[building_mask] = (0.55, 0.55, 0.55)
    for mask in rejected_masks:
        overlay[mask] = (0.8, 0.3, 0.3)
    for mask in reference_masks:
        overlay[mask] = (0.2, 0.9, 0.2)
    axes[1].imshow(overlay)
    axes[1].set_title("CLS buildings (grey) | shadow (blue)\nreferences (green) | rejected candidates (red)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=9)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and Paths; non-finite floats -> None."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return path
