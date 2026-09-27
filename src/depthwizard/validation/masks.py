"""Phase 5: which pixels are evaluated, and which category each belongs to.

Nothing is filtered silently. :func:`build_valid_mask` returns the boolean mask
**and** a count of every excluded pixel under the single reason that excluded
it, so a report can say exactly how many pixels were scored and why the rest
were not.

Exclusion reasons, in precedence order
--------------------------------------
A pixel that fails several tests is counted once, under the **first** reason
that applies. The order is fixed and documented because it decides what the
report says::

    1. prediction_nodata     prediction equals its declared nodata value
    2. prediction_nonfinite  prediction is NaN or +/-inf
    3. reference_nodata      reference equals its declared nodata value
    4. reference_nonfinite   reference is NaN or +/-inf
    5. reference_invalid     the reference's own validity mask says invalid
    6. outside_region        outside the supplied evaluation region
    7. excluded              explicitly excluded by the caller (e.g. the
                             reference buildings a calibration was fitted on)

Nodata is checked before NaN so that a sentinel such as -9999 is always
reported as nodata, never as a value, and never enters a metric.

Categories
----------
:class:`CategoryMasks` holds the OVERALL / TERRAIN / BUILDING split. The masks
come from labels the data actually carries -- DFC2019 CLS classes, or the
synthetic fixture's specified footprints -- and are never inferred from the
prediction or from the reference height. A category the data cannot supply is
``None`` and is reported as not yet measured, never as an empty (zero) set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "EXCLUSION_ORDER",
    "OVERALL",
    "TERRAIN",
    "BUILDING",
    "CATEGORIES",
    "ValidityMask",
    "build_valid_mask",
    "CategoryMasks",
    "category_masks_from_labels",
]

EXCLUSION_ORDER: tuple[str, ...] = (
    "prediction_nodata",
    "prediction_nonfinite",
    "reference_nodata",
    "reference_nonfinite",
    "reference_invalid",
    "outside_region",
    "excluded",
)

OVERALL = "overall"
TERRAIN = "terrain"
BUILDING = "building"
CATEGORIES: tuple[str, ...] = (OVERALL, TERRAIN, BUILDING)


@dataclass(frozen=True)
class ValidityMask:
    """The evaluation mask plus a per-reason account of every excluded pixel.

    ``valid.sum() + sum(exclusions.values()) == total_pixels`` always holds.
    """

    valid: np.ndarray
    exclusions: dict[str, int]
    total_pixels: int

    @property
    def valid_pixels(self) -> int:
        return int(self.valid.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "excluded_pixels": dict(self.exclusions),
            "exclusion_precedence": list(EXCLUSION_ORDER),
        }


def _as_bool(mask: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} has shape {mask.shape}, expected {shape}")
    return mask


def _equals_nodata(values: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.zeros(values.shape, dtype=bool)
    if np.isnan(nodata):
        # A NaN nodata is the non-finite case; counted there, not here.
        return np.zeros(values.shape, dtype=bool)
    return values == nodata


def build_valid_mask(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    prediction_nodata: float | None = None,
    reference_nodata: float | None = None,
    reference_valid: np.ndarray | None = None,
    region: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> ValidityMask:
    """Decide which pixels are evaluated, and record why the others are not.

    Args:
        prediction: 2-D predicted values, any units.
        reference: 2-D reference values on the **same grid**.
        prediction_nodata: The prediction raster's declared nodata, or None.
        reference_nodata: The reference raster's declared nodata, or None.
        reference_valid: Optional validity mask the reference carries itself
            (e.g. the Phase 3 reader's mask for DFC2019 AGL).
        region: Optional mask of pixels inside the evaluation region.
        exclude: Optional mask of pixels to leave out on purpose.

    Returns:
        A :class:`ValidityMask`. Counts follow :data:`EXCLUSION_ORDER`.

    Raises:
        ValueError: if any array is not on the prediction's grid.
    """
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be 2-D, got shape {prediction.shape}")
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference {reference.shape} and prediction {prediction.shape} are not on the same grid"
        )
    shape = prediction.shape
    tests: list[tuple[str, np.ndarray | None]] = [
        ("prediction_nodata", _equals_nodata(prediction, prediction_nodata)),
        ("prediction_nonfinite", ~np.isfinite(prediction)),
        ("reference_nodata", _equals_nodata(reference, reference_nodata)),
        ("reference_nonfinite", ~np.isfinite(reference)),
        ("reference_invalid", None if reference_valid is None else ~_as_bool(reference_valid, shape, "reference_valid")),
        ("outside_region", None if region is None else ~_as_bool(region, shape, "region")),
        ("excluded", _as_bool(exclude, shape, "exclude")),
    ]
    remaining = np.ones(shape, dtype=bool)
    exclusions: dict[str, int] = {}
    for name, failed in tests:
        if failed is None:
            exclusions[name] = 0
            continue
        hit = remaining & failed
        exclusions[name] = int(hit.sum())
        remaining &= ~failed
    return ValidityMask(valid=remaining, exclusions=exclusions, total_pixels=int(prediction.size))


@dataclass(frozen=True)
class CategoryMasks:
    """OVERALL / TERRAIN / BUILDING membership, from labels the data carries.

    ``terrain`` or ``building`` is None when the data has no label for it; the
    report then says not yet measured rather than scoring an empty set.
    """

    shape: tuple[int, int]
    terrain: np.ndarray | None
    building: np.ndarray | None
    #: Where each mask came from, recorded in the report.
    source: str
    #: Free-text definition of each category, recorded in the report.
    definitions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (TERRAIN, BUILDING):
            mask = getattr(self, name)
            if mask is not None:
                object.__setattr__(self, name, _as_bool(mask, tuple(self.shape), name))
        if self.terrain is not None and self.building is not None and (self.terrain & self.building).any():
            raise ValueError("terrain and building masks overlap; a pixel cannot be both")

    def get(self, category: str) -> np.ndarray | None:
        """The category's mask; OVERALL is every pixel."""
        if category == OVERALL:
            return np.ones(self.shape, dtype=bool)
        if category == TERRAIN:
            return self.terrain
        if category == BUILDING:
            return self.building
        raise KeyError(f"unknown category {category!r}; expected one of {CATEGORIES}")


def category_masks_from_labels(
    labels: np.ndarray,
    *,
    terrain_classes: Sequence[int],
    building_classes: Sequence[int],
    source: str,
) -> CategoryMasks:
    """TERRAIN / BUILDING masks from a class-label raster (e.g. DFC2019 CLS).

    Pixels of any other class (vegetation, water, bridges, unlabelled) belong to
    neither category, but still count towards OVERALL.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    terrain_classes = tuple(int(c) for c in terrain_classes)
    building_classes = tuple(int(c) for c in building_classes)
    if set(terrain_classes) & set(building_classes):
        raise ValueError(f"a class cannot be both terrain and building: {set(terrain_classes) & set(building_classes)}")
    return CategoryMasks(
        shape=labels.shape,
        terrain=np.isin(labels, terrain_classes) if terrain_classes else None,
        building=np.isin(labels, building_classes) if building_classes else None,
        source=source,
        definitions={
            TERRAIN: f"label in {list(terrain_classes)}",
            BUILDING: f"label in {list(building_classes)}",
            OVERALL: "every evaluated pixel, whatever its label",
        },
    )
