"""Radiometric normalisation.

Satellite imagery arrives in wildly different radiometric ranges: 11-bit
WorldView, 12-bit Pleiades, 16-bit Sentinel reflectance, 8-bit web imagery.
Downstream stages want one predictable range, so this module maps any input
onto 8-bit using a **percentile stretch**.

Why percentiles and not min/max: a single saturated pixel (specular glint off
a roof) or a single dead pixel drags a min/max stretch to uselessness. Clipping
at the 2nd and 98th percentiles throws those away and spends the full output
range on the values that actually occur.

The per-band cut values are returned in :class:`BandStretch`, not just applied,
because a stretch is a lossy decision: to reproduce a result later, or to map
an 8-bit value back towards physical units, you need to know exactly what was
clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from depthwizard.logging_setup import get_logger

__all__ = [
    "BandStretch",
    "StretchResult",
    "compute_band_stretches",
    "apply_stretch",
    "normalize_to_uint8",
]

log = get_logger(__name__)


@dataclass(frozen=True)
class BandStretch:
    """The clip window chosen for one band."""

    band: int
    low: float
    high: float
    lower_percentile: float
    upper_percentile: float
    #: True when the band was flat (low == high) and no stretch was possible.
    degenerate: bool = False

    @property
    def span(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class StretchResult:
    """Normalised pixels plus the exact parameters used to produce them."""

    array: np.ndarray
    bands: tuple[BandStretch, ...]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype


def _valid_mask(band: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean mask of pixels that should influence the percentiles."""
    mask = np.isfinite(band) if np.issubdtype(band.dtype, np.floating) else np.ones(band.shape, bool)
    if nodata is not None:
        mask &= band != nodata
    return mask


def compute_band_stretches(
    array: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    per_band: bool = True,
    nodata: float | None = None,
) -> tuple[BandStretch, ...]:
    """Choose a clip window per band (or one shared window for all bands).

    Args:
        array: ``(rows, cols, bands)`` input.
        lower_percentile: Percentile mapped to black. Must be < upper.
        upper_percentile: Percentile mapped to white.
        per_band: True stretches each band independently, which maximises
            contrast. False uses one window across all bands, which preserves
            relative colour between bands.
        nodata: Value excluded from the percentile calculation.

    Returns:
        One :class:`BandStretch` per band.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a (rows, cols, bands) array, got shape {array.shape}")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            f"need 0 <= lower < upper <= 100, got lower={lower_percentile}, "
            f"upper={upper_percentile}"
        )

    n_bands = array.shape[2]

    if per_band:
        windows = []
        for index in range(n_bands):
            band = array[:, :, index]
            valid = band[_valid_mask(band, nodata)]
            windows.append(_percentiles(valid, lower_percentile, upper_percentile))
    else:
        valid = array[_valid_mask(array, nodata)]
        shared = _percentiles(valid, lower_percentile, upper_percentile)
        windows = [shared] * n_bands

    return tuple(
        BandStretch(
            band=index,
            low=low,
            high=high,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            degenerate=degenerate,
        )
        for index, (low, high, degenerate) in enumerate(windows)
    )


def _percentiles(
    values: np.ndarray, lower: float, upper: float
) -> tuple[float, float, bool]:
    """Return ``(low, high, degenerate)`` for a flat array of valid samples."""
    if values.size == 0:
        # Nothing valid to measure. Say so rather than inventing a window.
        return 0.0, 1.0, True
    low = float(np.percentile(values, lower))
    high = float(np.percentile(values, upper))
    if not high > low:
        # Flat band, or percentiles collapsed onto one value. Widen to the full
        # range if we can; otherwise flag it and let apply_stretch emit zeros.
        full_low, full_high = float(values.min()), float(values.max())
        if full_high > full_low:
            return full_low, full_high, False
        return full_low, full_low, True
    return low, high, False


def apply_stretch(
    array: np.ndarray,
    stretches: Sequence[BandStretch],
    *,
    out_dtype: type = np.uint8,
) -> np.ndarray:
    """Clip each band to its window and rescale to the output dtype's range.

    A degenerate (flat) band becomes all zeros rather than dividing by zero.

    Non-finite pixels (NaN, +/-inf) are excluded from the percentile
    calculation but still have to be assigned an output value. They are mapped
    deterministically -- NaN and -inf to the output minimum, +inf to the
    maximum -- because casting NaN to an integer dtype is undefined behaviour
    and would otherwise produce platform-dependent garbage.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a (rows, cols, bands) array, got shape {array.shape}")
    if len(stretches) != array.shape[2]:
        raise ValueError(
            f"got {len(stretches)} stretch windows for {array.shape[2]} bands"
        )

    info = np.iinfo(out_dtype)
    scale = float(info.max - info.min)
    out = np.empty(array.shape, dtype=out_dtype)

    for stretch in stretches:
        band = array[:, :, stretch.band].astype(np.float64, copy=False)
        if stretch.degenerate:
            out[:, :, stretch.band] = info.min
            continue
        with np.errstate(invalid="ignore"):
            normalised = (band - stretch.low) / stretch.span
        np.nan_to_num(normalised, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(normalised, 0.0, 1.0, out=normalised)
        out[:, :, stretch.band] = np.rint(normalised * scale + info.min).astype(out_dtype)

    return out


def normalize_to_uint8(
    array: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    per_band: bool = True,
    nodata: float | None = None,
) -> StretchResult:
    """Percentile-stretch ``array`` and convert it to 8-bit.

    This is the normal entry point. It is a no-op in spirit for imagery that is
    already well-scaled 8-bit, and rescues 11/12/16-bit imagery that would
    otherwise render as near-black.
    """
    stretches = compute_band_stretches(
        array,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        per_band=per_band,
        nodata=nodata,
    )
    out = apply_stretch(array, stretches, out_dtype=np.uint8)

    degenerate = [s.band for s in stretches if s.degenerate]
    if degenerate:
        log.warning(
            "flat band(s) could not be stretched and were zeroed",
            extra={"bands": degenerate},
        )
    log.info(
        "radiometric normalisation applied",
        extra={
            "in_dtype": str(array.dtype),
            "bands": len(stretches),
            "percentiles": f"{lower_percentile}-{upper_percentile}",
            "windows": "; ".join(f"b{s.band}:[{s.low:g},{s.high:g}]" for s in stretches),
        },
    )
    return StretchResult(array=out, bands=stretches)
