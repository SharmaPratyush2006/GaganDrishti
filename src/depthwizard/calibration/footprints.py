"""Building footprints from the DFC2019 CLS semantic labels.

No building detector is trained or run here. DFC2019 Track 1 ships a CLS raster
beside every RGB/AGL pair, and its building class *is* the footprint source::

    CLS == building class  ->  connected components  ->  minimum-area filter

The building class value
------------------------
The CLS rasters use ASPRS LAS classification codes. The values present in the
real download are exactly {2, 5, 6, 9, 17, 65}, and the official DFC2019
baseline code maps them as (``pubgeo/dfc2019``, ``track1/track1-metrics.py``,
``las_to_sequential_labels``)::

    2 ground | 5 trees | 6 building roof | 9 water | 17 bridge / elevated road
    (anything else, i.e. 65, is unlabeled)

and ``track1/unets/params.py`` names the constant ``LAS_LABEL_ROOF = 6``. The
class is therefore 6, and :data:`DFC2019_BUILDING_CLASS_SOURCE` records where
that came from.

Connectivity
------------
4-connectivity by default: two roofs whose labels touch only at a corner stay
two buildings. Components are never merged afterwards.

Building ids
------------
``"<tile_id>:b<label>"``, with ``label`` the component number from
:func:`scipy.ndimage.label` (raster order), so the id is stable for a given
tile and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from depthwizard.shadows.measure import BuildingFootprint

__all__ = [
    "DFC2019_LAS_CLASSES",
    "DFC2019_BUILDING_CLASS",
    "DFC2019_BUILDING_CLASS_SOURCE",
    "RejectedComponent",
    "FootprintExtraction",
    "extract_building_footprints",
]

#: ASPRS LAS codes used by the DFC2019 CLS rasters.
DFC2019_LAS_CLASSES: dict[int, str] = {
    2: "ground",
    5: "trees",
    6: "building roof",
    9: "water",
    17: "bridge / elevated road",
    65: "unlabeled",
}

DFC2019_BUILDING_CLASS = 6

DFC2019_BUILDING_CLASS_SOURCE = (
    "pubgeo/dfc2019 (official DFC2019 baseline): track1/track1-metrics.py "
    "las_to_sequential_labels 'labels[las_labels == 6] = 2  # building roof' and "
    "track1/unets/params.py 'LAS_LABEL_ROOF = 6'; the real CLS rasters contain "
    "exactly the values {2, 5, 6, 9, 17, 65}"
)

REJECT_MIN_AREA = "below_min_area"
REJECT_EDGE = "touches_tile_edge"


@dataclass(frozen=True)
class RejectedComponent:
    building_id: str
    area_px: int
    reason: str


@dataclass(frozen=True)
class FootprintExtraction:
    """Every building-class component in a tile, kept or rejected."""

    tile_id: str
    building_class: int
    min_area_px: int
    connectivity: int
    footprints: tuple[BuildingFootprint, ...]
    rejected: tuple[RejectedComponent, ...]
    #: ``(rows, cols)`` bool, every building-class pixel (kept or not).
    building_mask: np.ndarray = field(repr=False)
    #: ``(rows, cols)`` int32 component labels; 0 = not building.
    labels: np.ndarray = field(repr=False)

    @property
    def n_components(self) -> int:
        return len(self.footprints) + len(self.rejected)

    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.rejected:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "building_class": self.building_class,
            "min_area_px": self.min_area_px,
            "connectivity": self.connectivity,
            "components_found": self.n_components,
            "components_surviving": len(self.footprints),
            "components_rejected": len(self.rejected),
            "rejection_reasons": self.rejection_counts(),
        }


def extract_building_footprints(
    cls: np.ndarray,
    *,
    tile_id: str,
    building_class: int = DFC2019_BUILDING_CLASS,
    min_area_px: int = 50,
    connectivity: int = 4,
    reject_edge_touching: bool = True,
) -> FootprintExtraction:
    """Label the building class of ``cls`` into filtered footprints.

    Args:
        cls: 2-D integer semantic label raster.
        tile_id: Prefix for the stable building ids.
        building_class: The verified CLS value for buildings.
        min_area_px: Components with fewer pixels are rejected (``below_min_area``).
        connectivity: 4 or 8.
        reject_edge_touching: Reject components touching the raster border
            (``touches_tile_edge``); the crop truncates them.

    Returns:
        A :class:`FootprintExtraction`. A component rejected for both reasons is
        reported once, under ``below_min_area``.
    """
    labels_in = np.asarray(cls)
    if labels_in.ndim != 2:
        raise ValueError(f"CLS raster must be 2-D, got shape {labels_in.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    building_mask = labels_in == building_class
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, n = ndimage.label(building_mask, structure=structure)

    footprints: list[BuildingFootprint] = []
    rejected: list[RejectedComponent] = []
    if n:
        areas = np.bincount(labels.ravel(), minlength=n + 1)
        border = np.zeros(n + 1, dtype=bool)
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            border[np.unique(edge)] = True
        slices = ndimage.find_objects(labels)
        for label in range(1, n + 1):
            building_id = f"{tile_id}:b{label:04d}"
            area = int(areas[label])
            if area < min_area_px:
                rejected.append(RejectedComponent(building_id, area, REJECT_MIN_AREA))
                continue
            if reject_edge_touching and border[label]:
                rejected.append(RejectedComponent(building_id, area, REJECT_EDGE))
                continue
            mask = np.zeros_like(building_mask)
            window = slices[label - 1]
            mask[window] = labels[window] == label
            footprints.append(BuildingFootprint(building_id=building_id, mask=mask))

    return FootprintExtraction(
        tile_id=tile_id,
        building_class=int(building_class),
        min_area_px=int(min_area_px),
        connectivity=int(connectivity),
        footprints=tuple(footprints),
        rejected=tuple(rejected),
        building_mask=building_mask,
        labels=labels.astype(np.int32),
    )
