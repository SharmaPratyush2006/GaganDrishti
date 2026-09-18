"""Shadow detection: a swappable interface, and one classical implementation.

Phase 2 needs a binary "is this pixel shadowed?" map. It gets one from a
deterministic classical rule. A later phase is expected to replace that rule
with a learned segmenter -- and the only thing that should have to change is
which :class:`ShadowDetector` is constructed.

The interface
-------------
:class:`ShadowDetector` is an abstract base class with a single method::

    detect(image, context) -> ShadowMask

``context`` carries what the detector is allowed to know about the scene (GSD,
sun angles, band order, processing mode); ``ShadowMask`` carries the binary
mask *plus* the method name, the threshold, the parameters and the diagnostics
that produced it. Nothing about how a mask was made is thrown away, because a
downstream height that looks wrong is usually a mask that was wrong.

The classical rule
------------------
:class:`ClassicalShadowDetector` implements exactly one method: **HSV
value-channel thresholding**.

An image is converted to HSV and the **V** (value) channel is thresholded; the
dark side of the threshold is shadow. V is used rather than a luminance average
because V is ``max(R, G, B)``, which is what "how brightly is this surface lit"
actually means: a surface in shadow is dim in *every* channel, whereas a dark
but sunlit surface -- a red roof, say -- can still be bright in one.

The threshold is chosen by **Otsu's method**, which picks the value maximising
between-class variance of the V histogram. Otsu is used because it is
deterministic, parameter-free and reproducible: the same image always gives the
same threshold, so a regression in the mask can never be blamed on a random
seed. A fixed threshold can be supplied instead when a scene needs pinning.

Deliberately **not** implemented here, because Phase 2 keeps to one method:
saturation/hue ratio indices, chromaticity-invariant illumination models,
region growing, morphological cleanup, and any learned model.

This module detects shadows only. It does **not** detect buildings: footprints
are supplied to the measurement API by the caller.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = [
    "DetectionContext",
    "ShadowMask",
    "ShadowDetector",
    "ClassicalShadowDetector",
]

log = get_logger(__name__)

#: Channel orders this module knows how to pick R, G and B out of.
BAND_ORDERS: dict[str, tuple[int, int, int]] = {
    # OpenCV (and therefore ImageLoader) hands back blue, green, red.
    "bgr": (2, 1, 0),
    # rasterio hands back the bands in the order the file stores them.
    "rgb": (0, 1, 2),
}


# ---------------------------------------------------------------------------
# Context and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionContext:
    """What a detector is told about the scene it is looking at.

    Every field is optional because a detector must also work on a plain image
    in :attr:`~depthwizard.mode.Mode.RELATIVE` mode, where none of the
    geometry is known. The classical detector uses none of them; they exist so
    that a future learned detector -- which may well want the sun angles -- can
    be dropped in without changing a single call site.
    """

    gsd_m: float | None = None
    sun_elevation_deg: float | None = None
    sun_azimuth_deg: float | None = None
    mode: Mode = Mode.RELATIVE
    scene_name: str | None = None
    #: How to read colour out of a 3+ band array: ``"bgr"`` or ``"rgb"``.
    band_order: str = "bgr"

    def __post_init__(self) -> None:
        if self.band_order not in BAND_ORDERS:
            raise ValueError(
                f"band_order must be one of {sorted(BAND_ORDERS)}, got {self.band_order!r}"
            )

    @classmethod
    def from_scene_metadata(cls, metadata: Any, *, band_order: str = "rgb") -> "DetectionContext":
        """Build a context from a Phase 1
        :class:`~depthwizard.ingest.metadata.SceneMetadata`.

        Defaults to ``"rgb"`` because that path comes from rasterio, which
        hands back bands in file order rather than OpenCV's BGR.
        """
        return cls(
            gsd_m=getattr(metadata, "gsd_m", None) if metadata.georeference else None,
            sun_elevation_deg=metadata.sun_elevation_deg,
            sun_azimuth_deg=metadata.sun_azimuth_deg,
            mode=metadata.mode,
            scene_name=str(getattr(metadata, "path", "")) or None,
            band_order=band_order,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gsd_m": self.gsd_m,
            "sun_elevation_deg": self.sun_elevation_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "mode": self.mode.value,
            "scene_name": self.scene_name,
            "band_order": self.band_order,
        }


@dataclass(frozen=True)
class ShadowMask:
    """A binary shadow mask and the complete record of how it was made."""

    #: Boolean array, ``(rows, cols)``. True means "this pixel is shadowed".
    mask: np.ndarray
    #: Name of the detection method, e.g. ``"hsv_value_otsu"``.
    method: str
    #: The threshold actually applied, on the same 0-1 scale as ``score``.
    threshold: float
    #: Continuous per-pixel darkness in [0, 1] (``1 - V``); the mask is a cut
    #: through this. Kept so a borderline pixel can be inspected rather than
    #: just accepted or rejected.
    score: np.ndarray
    #: Every configuration value the detector ran with.
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Measured properties of this particular run.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise TypeError(f"mask must be a boolean array, got dtype {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2-D (rows, cols), got shape {self.mask.shape}")
        if self.score.shape != self.mask.shape:
            raise ValueError(
                f"score shape {self.score.shape} does not match mask shape {self.mask.shape}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))

    @property
    def shadow_pixel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def shadow_fraction(self) -> float:
        """Fraction of the image flagged as shadow, in [0, 1]."""
        return float(self.mask.mean())

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics only -- the arrays are left out deliberately."""
        return {
            "method": self.method,
            "threshold": self.threshold,
            "shape": list(self.shape),
            "shadow_pixel_count": self.shadow_pixel_count,
            "shadow_fraction": self.shadow_fraction,
            "parameters": dict(self.parameters),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class ShadowDetector(abc.ABC):
    """Turn an image into a binary shadow mask.

    Subclasses implement :meth:`detect`. They must:

    * return a :class:`ShadowMask` whose ``mask`` has the image's ``(rows,
      cols)`` shape,
    * name their method and report the threshold and parameters they used, and
    * not detect buildings -- footprints come from the caller.

    Swapping implementations is the point of this class. Phase 2 ships
    :class:`ClassicalShadowDetector`; a later phase can add a learned detector
    with the same signature and every downstream stage keeps working.
    """

    #: Stable identifier for the method, recorded on every mask.
    name: str = "shadow_detector"

    @abc.abstractmethod
    def detect(self, image: np.ndarray, context: DetectionContext) -> ShadowMask:
        """Detect shadows in ``image``.

        Args:
            image: 2-D ``(rows, cols)`` or 3-D ``(rows, cols, bands)`` array.
            context: Scene geometry and band order. See :class:`DetectionContext`.

        Returns:
            A :class:`ShadowMask`.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Array preparation
# ---------------------------------------------------------------------------


def _to_unit_float(image: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Scale any supported dtype to float32 in [0, 1].

    Returns ``(array, scale, note)``. The scale is reported rather than hidden,
    because it determines what the threshold *means*.
    """
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0, 255.0, "uint8 / 255"
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0, 65535.0, "uint16 / 65535"
    if np.issubdtype(array.dtype, np.integer):
        peak = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / peak, peak, f"{array.dtype} / {peak:g}"
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        peak = float(finite.max()) if finite.size else 0.0
        if peak <= 1.0:
            return np.nan_to_num(array.astype(np.float32), nan=0.0), 1.0, "float already in [0, 1]"
        return (
            np.nan_to_num(array.astype(np.float32), nan=0.0) / peak,
            peak,
            f"float / observed max {peak:g}",
        )
    raise TypeError(f"unsupported image dtype {array.dtype}")


def _hsv_value_and_saturation(
    image: np.ndarray, band_order: str
) -> tuple[np.ndarray, np.ndarray, float, str, int]:
    """Compute the HSV V and S channels, both in [0, 1].

    The HSV definitions are written out rather than delegated, so the physics
    reviewer can see exactly what "value" and "saturation" mean here::

        V = max(R, G, B)
        S = (V - min(R, G, B)) / V,  and 0 where V == 0

    A single-band image has no colour: V is the band itself and S is zero
    everywhere. That is the correct HSV of a grey pixel, not a special case.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {array.shape}")

    unit, scale, scale_note = _to_unit_float(array)
    band_count = int(unit.shape[2])

    if band_count == 1:
        value = unit[:, :, 0]
        saturation = np.zeros_like(value)
        return value, saturation, scale, scale_note, band_count

    if band_count == 2:
        raise ValueError(
            "a 2-band image has no usable colour interpretation; supply 1 band "
            "(panchromatic) or 3+ bands (colour)"
        )

    r_idx, g_idx, b_idx = BAND_ORDERS[band_order]
    rgb = np.stack([unit[:, :, r_idx], unit[:, :, g_idx], unit[:, :, b_idx]], axis=2)
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(value > 0.0, (value - minimum) / np.where(value > 0.0, value, 1.0), 0.0)
    return value.astype(np.float32), saturation.astype(np.float32), scale, scale_note, band_count


def _otsu_threshold_u8(value_u8: np.ndarray) -> tuple[int, float]:
    """Otsu's threshold over a uint8 array, plus its separability.

    Returns ``(threshold, separability)``. The threshold ``t`` is the value that
    maximises between-class variance; pixels ``<= t`` are the dark class.
    Separability is ``sigma_between^2 / sigma_total^2`` in [0, 1] -- Otsu's own
    goodness measure, reported as a diagnostic so a bimodal scene (high
    separability, trustworthy split) can be told apart from a flat one.

    Implemented here rather than called out to, so the threshold that decides
    every shadow pixel is readable in this file.
    """
    histogram = np.bincount(value_u8.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:  # pragma: no cover - an empty image cannot reach here
        return 0, 0.0

    levels = np.arange(256, dtype=np.float64)
    probability = histogram / total
    weight_low = np.cumsum(probability)
    mean_low_sum = np.cumsum(probability * levels)
    mean_total = mean_low_sum[-1]

    # Between-class variance for every candidate threshold, in one vectorised go.
    denominator = weight_low * (1.0 - weight_low)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(
            denominator > 0.0,
            (mean_total * weight_low - mean_low_sum) ** 2 / np.where(denominator > 0.0, denominator, 1.0),
            0.0,
        )

    threshold = int(np.argmax(between))
    variance_total = float((probability * (levels - mean_total) ** 2).sum())
    separability = float(between[threshold] / variance_total) if variance_total > 0.0 else 0.0
    return threshold, separability


# ---------------------------------------------------------------------------
# The classical detector
# ---------------------------------------------------------------------------


class ClassicalShadowDetector(ShadowDetector):
    """HSV value-channel thresholding. One rule, deterministic, no training.

    Args:
        threshold: Fixed threshold on the HSV V channel, in [0, 1]. Pixels with
            ``V <= threshold`` are shadow. Leave as None (the default) to pick
            the threshold with Otsu's method on this image's V histogram.
        min_shadow_fraction: If the resulting mask covers less than this
            fraction of the image, the run is flagged in diagnostics as
            ``suspicious_shadow_fraction``. It is a warning, never a rejection:
            the mask is still returned, and the measurement stage is what
            decides whether the evidence is sufficient.
        max_shadow_fraction: The same, at the other end. A mask covering most
            of the image usually means the scene had no bright surface for Otsu
            to split against.

    Example::

        detector = ClassicalShadowDetector()
        mask = detector.detect(image, DetectionContext(gsd_m=0.5))
        print(mask.method, mask.threshold, mask.shadow_fraction)
    """

    name = "classical_hsv_value"

    def __init__(
        self,
        *,
        threshold: float | None = None,
        min_shadow_fraction: float = 0.001,
        max_shadow_fraction: float = 0.60,
    ) -> None:
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if not 0.0 <= min_shadow_fraction < max_shadow_fraction <= 1.0:
            raise ValueError(
                "need 0 <= min_shadow_fraction < max_shadow_fraction <= 1, got "
                f"{min_shadow_fraction} and {max_shadow_fraction}"
            )
        self.threshold = None if threshold is None else float(threshold)
        self.min_shadow_fraction = float(min_shadow_fraction)
        self.max_shadow_fraction = float(max_shadow_fraction)

    @property
    def method(self) -> str:
        """``"hsv_value_otsu"`` or ``"hsv_value_fixed"``, per the threshold source."""
        return "hsv_value_fixed" if self.threshold is not None else "hsv_value_otsu"

    def detect(self, image: np.ndarray, context: DetectionContext) -> ShadowMask:
        """Threshold the HSV value channel. See the class docstring.

        Args:
            image: 2-D ``(rows, cols)`` or 3-D ``(rows, cols, bands)`` array of
                any integer or float dtype.
            context: Supplies ``band_order``. No other field is read -- the
                classical rule uses no scene geometry at all.

        Returns:
            A :class:`ShadowMask` whose ``score`` is ``1 - V``.
        """
        value, saturation, scale, scale_note, band_count = _hsv_value_and_saturation(
            image, context.band_order
        )

        value_u8 = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
        if self.threshold is None:
            threshold_u8, separability = _otsu_threshold_u8(value_u8)
            threshold = threshold_u8 / 255.0
            threshold_source = "otsu"
        else:
            threshold = self.threshold
            threshold_u8 = int(round(threshold * 255.0))
            _, separability = _otsu_threshold_u8(value_u8)
            threshold_source = "fixed"

        # Compare in uint8 space, so the mask matches the threshold Otsu chose
        # exactly rather than drifting by a float rounding step.
        mask = value_u8 <= threshold_u8

        shadow_fraction = float(mask.mean())
        suspicious = (
            shadow_fraction < self.min_shadow_fraction or shadow_fraction > self.max_shadow_fraction
        )

        shadow_values = value[mask]
        lit_values = value[~mask]
        diagnostics: dict[str, Any] = {
            "threshold_u8": int(threshold_u8),
            "threshold_source": threshold_source,
            "otsu_separability": separability,
            "input_dtype": str(np.asarray(image).dtype),
            "input_band_count": band_count,
            "input_scale": scale,
            "input_scale_note": scale_note,
            "value_mean": float(value.mean()),
            "value_min": float(value.min()),
            "value_max": float(value.max()),
            "saturation_mean": float(saturation.mean()),
            "shadow_value_mean": float(shadow_values.mean()) if shadow_values.size else None,
            "lit_value_mean": float(lit_values.mean()) if lit_values.size else None,
            "shadow_fraction": shadow_fraction,
            "suspicious_shadow_fraction": bool(suspicious),
        }

        log.info(
            "detected shadows",
            extra={
                "method": self.method,
                "threshold": threshold,
                "shadow_fraction": shadow_fraction,
                "otsu_separability": separability,
                "scene": context.scene_name,
            },
        )

        return ShadowMask(
            mask=mask,
            method=self.method,
            threshold=threshold,
            score=(1.0 - value).astype(np.float32),
            parameters={
                "detector": self.name,
                "threshold": self.threshold,
                "band_order": context.band_order,
                "min_shadow_fraction": self.min_shadow_fraction,
                "max_shadow_fraction": self.max_shadow_fraction,
            },
            diagnostics=diagnostics,
        )
