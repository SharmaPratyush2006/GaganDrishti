"""Shadow anchors: metric heights from shadows, with the sun fixed first.

The degeneracy, and how it is avoided
-------------------------------------
The shadow constraint for building ``i`` is::

    a * r_bar_i + b = L_i * tan(theta)

Scaling ``tan(theta)``, ``a`` and ``b`` together leaves every equation true, so
solving for all three at once is meaningless. They are therefore never solved
jointly. In order:

1. **Sun calibration.** A few automatically selected *reference* buildings with
   known DFC2019 AGL fix the scale: ``tan(theta)_i = h_ref_i / L_ref_i``,
   aggregated (median) into one value per tile.
2. **Shadow heights.** With that value fixed, every other building gets
   ``h_i = L_i * tan(theta)``.
3. **Scale and offset.** Only then are ``a, b`` fitted
   (:mod:`depthwizard.calibration.fusion`) -- two unknowns, many constraints.

GSD invariance
--------------
``L`` in metres is ``L_px * GSD``. With the reference calibration the GSD
cancels exactly::

    h_i = (L_i,px * GSD) * h_ref / (L_ref,px * GSD) = h_ref * L_i,px / L_ref,px

and likewise ``dh_i = tan(theta) * dL_i``. So the calibration is carried out as
one number, :attr:`ShadowScale.metres_per_shadow_px` (``= GSD * tan(theta)``),
and every height, weight and fitted ``a, b`` is independent of the GSD. The
DFC2019 Track 1 tile GSD could not be verified from an authoritative source, so
it is not assumed: ``tan(theta)`` and ``theta`` in degrees are only reported
when a verified GSD is configured, and the Phase 2 25-45 degree sun-band gate
is then recorded as *not evaluable* rather than applied.

Because the tiles are unrectified off-nadir views, whatever shifts the roof
edge relative to the shadow tip (relief displacement) is also proportional to
building height, so the calibrated scale is an *effective* one that absorbs
it. Even with a GSD the derived ``theta`` is an effective elevation.

Reuse of Phase 2
----------------
Shadow lengths come from :func:`depthwizard.shadows.measure.measure_shadow_lengths`
unchanged. Where both the GSD and the sun elevation are known (the synthetic
control), confidence and the band gate come from
:func:`depthwizard.physics.height.height_from_shadow`.

Uncertainty
-----------
Two forms, compared in the run report:

* ``explicit`` -- a ``dL`` supplied with a real justification (the synthetic
  control's grid-quantisation bound). None exists for DFC2019.
* ``ray_spread_proxy`` -- ``dL_px = ray_length_spread_px`` (Phase 2's p84 - p16
  over the rays). **This is an EMPIRICAL PROXY, not a measurement
  uncertainty**: the spread mixes real roofline variation, shadow geometry,
  occlusion and measurement error.

Without a valid ``dh`` a constraint is **dropped**; no default is invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image

from depthwizard.physics.height import (
    USABLE_SUN_ELEVATION_MAX_DEG,
    USABLE_SUN_ELEVATION_MIN_DEG,
    Confidence,
    height_from_shadow,
)
from depthwizard.physics.sun import shadow_direction_pixels
from depthwizard.shadows.measure import (
    BuildingFootprint,
    MeasurementParams,
    ShadowLengthMeasurement,
    measure_shadow_lengths,
)

__all__ = [
    "UNCERTAINTY_EXPLICIT",
    "UNCERTAINTY_RAY_SPREAD_PROXY",
    "PROXY_LABEL",
    "BuildingShadow",
    "measure_building_shadows",
    "merged_ray_fraction",
    "K_ESTIMATORS",
    "estimate_k",
    "ReferenceCandidate",
    "assess_reference_candidates",
    "ShadowScale",
    "scale_from_references",
    "scale_from_known_geometry",
    "HeightConstraint",
    "shadow_height_constraints",
    "explicit_grid_dl_px",
]

UNCERTAINTY_EXPLICIT = "explicit"
UNCERTAINTY_RAY_SPREAD_PROXY = "ray_spread_proxy"
PROXY_LABEL = (
    "EMPIRICAL PROXY: dh = tan(theta) * ray_length_spread; the ray spread mixes roofline "
    "variation, shadow geometry, occlusion and measurement error. It is NOT a rigorous "
    "measurement uncertainty."
)


# ---------------------------------------------------------------------------
# Measurement (Phase 2, reused)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildingShadow:
    """One footprint's Phase 2 shadow measurement plus Phase 4a validity."""

    footprint: BuildingFootprint = field(repr=False)
    measurement: ShadowLengthMeasurement = field(repr=False)
    confidence: Confidence
    reduced_confidence_reasons: tuple[str, ...]
    #: Share of the hit rays that stopped at the tile edge.
    edge_terminated_fraction: float
    #: None when usable; otherwise why the shadow cannot anchor a height.
    rejection_reason: str | None
    #: Share of hit rays crossing another building footprint; None if no hits.
    merged_fraction: float | None = None

    @property
    def building_id(self) -> str:
        return self.footprint.building_id

    @property
    def shadow_length_px(self) -> float | None:
        return self.measurement.shadow_length_px

    @property
    def spread_px(self) -> float | None:
        return self.measurement.ray_length_spread_px

    @property
    def usable(self) -> bool:
        return self.rejection_reason is None


def measure_building_shadows(
    footprints: Sequence[BuildingFootprint],
    shadow_mask: np.ndarray,
    *,
    sun_azimuth_deg: float,
    params: MeasurementParams,
    max_edge_terminated_fraction: float = 0.5,
    max_merged_fraction: float | None = 0.5,
    occluder_mask: np.ndarray | None = None,
    gsd_m: float | None = None,
    sun_elevation_deg: float | None = None,
    sun_band_min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
    sun_band_max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
) -> list[BuildingShadow]:
    """Measure every footprint's shadow with Phase 2 and apply the validity rules.

    Without a GSD the measurement runs in pixel units (``gsd_m=1``): only
    ``shadow_length_px`` and ``ray_length_spread_px`` are read afterwards, never
    the metre fields. Without a sun elevation, NaN is passed; Phase 2 only
    records it.

    Rejections (``rejection_reason``):

    * ``failed_confidence: <Phase 2 reason>`` -- the measurement failed;
    * ``shadow_truncated_by_tile_edge`` -- most hit rays ran off the tile;
    * ``merged_shadow`` -- most hit rays crossed another building's footprint
      (:func:`merged_ray_fraction`), the same majority rule as edge truncation;
      ``max_merged_fraction=None`` disables it;
    * ``outside_usable_sun_band`` -- only when the elevation is known.
    """
    both_known = gsd_m is not None and sun_elevation_deg is not None
    all_buildings = np.zeros(np.asarray(shadow_mask).shape, dtype=bool)
    for footprint in footprints:
        all_buildings |= footprint.mask
    if occluder_mask is not None:
        all_buildings |= np.asarray(occluder_mask, dtype=bool)
    measurements = measure_shadow_lengths(
        footprints,
        shadow_mask,
        gsd_m=float(gsd_m) if gsd_m is not None else 1.0,
        sun_azimuth_deg=float(sun_azimuth_deg),
        sun_elevation_deg=float(sun_elevation_deg) if sun_elevation_deg is not None else math.nan,
        params=params,
        occluder_mask=occluder_mask,
    )

    results: list[BuildingShadow] = []
    for footprint, measurement in zip(footprints, measurements):
        hits = [ray for ray in measurement.rays if ray.hit]
        edge_fraction = (
            sum(ray.terminated_by == "image_edge" for ray in hits) / len(hits) if hits else 0.0
        )
        reasons = tuple(measurement.reduced_confidence_reasons)
        rejection: str | None = None
        if not measurement.ok or measurement.shadow_length_px is None:
            confidence = Confidence.FAILED
            rejection = f"failed_confidence: {measurement.failure_reason}"
        elif both_known:
            estimate = height_from_shadow(
                measurement.shadow_length_m,
                float(sun_elevation_deg),
                extra_reduced_confidence_reasons=reasons,
                sun_band_min_deg=sun_band_min_deg,
                sun_band_max_deg=sun_band_max_deg,
            )
            confidence = estimate.confidence
            reasons = estimate.reduced_confidence_reasons
            if not estimate.sun_band.in_band:
                rejection = "outside_usable_sun_band"
        else:
            confidence = Confidence.REDUCED if reasons else Confidence.NOMINAL
        merged = merged_ray_fraction(measurement, all_buildings & ~footprint.mask)
        if rejection is None and edge_fraction > max_edge_terminated_fraction:
            rejection = "shadow_truncated_by_tile_edge"
        if (
            rejection is None
            and max_merged_fraction is not None
            and merged is not None
            and merged > max_merged_fraction
        ):
            rejection = "merged_shadow"
        results.append(
            BuildingShadow(
                footprint=footprint,
                measurement=measurement,
                confidence=confidence,
                reduced_confidence_reasons=reasons,
                edge_terminated_fraction=float(edge_fraction),
                rejection_reason=rejection,
                merged_fraction=merged,
            )
        )
    return results


def merged_ray_fraction(measurement: ShadowLengthMeasurement, other_buildings: np.ndarray) -> float | None:
    """Share of hit rays whose measured segment crosses another building's footprint.

    Phase 2 skips other buildings while walking a ray, so a ray can pass a
    neighbour and carry on through *that* building's shadow: the measured
    length then belongs partly to the neighbour. A ray whose segment from its
    launch pixel to its last shadow pixel crosses another footprint is counted
    as merged. None when no ray hit.
    """
    hits = [ray for ray in measurement.rays if ray.hit and ray.length_px]
    if not hits:
        return None
    d_row, d_col = measurement.direction_px
    n_rows, n_cols = other_buildings.shape
    crossed = 0
    for ray in hits:
        t = np.arange(0.5, float(ray.length_px) + 0.5, 0.5)
        rows = np.clip(np.rint(ray.start_row + d_row * t).astype(int), 0, n_rows - 1)
        cols = np.clip(np.rint(ray.start_col + d_col * t).astype(int), 0, n_cols - 1)
        crossed += bool(other_buildings[rows, cols].any())
    return crossed / len(hits)


# ---------------------------------------------------------------------------
# Reference-building selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCandidate:
    building_id: str
    area_px: int
    shadow_length_px: float | None
    spread_px: float | None
    #: Median DFC2019 AGL over the footprint; None when not computed.
    h_ref_m: float | None
    solidity: float | None
    shadow_zone_ground_fraction: float | None
    rejection_reason: str | None

    @property
    def eligible(self) -> bool:
        return self.rejection_reason is None

    @property
    def relative_spread(self) -> float | None:
        if self.spread_px is None or not self.shadow_length_px:
            return None
        return self.spread_px / self.shadow_length_px

    @property
    def ratio(self) -> float | None:
        """``h_ref / L_px`` -- this building's metres-per-shadow-pixel."""
        if self.h_ref_m is None or not self.shadow_length_px:
            return None
        return self.h_ref_m / self.shadow_length_px

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "area_px": self.area_px,
            "shadow_length_px": self.shadow_length_px,
            "ray_length_spread_px": self.spread_px,
            "relative_spread": self.relative_spread,
            "h_ref_m": self.h_ref_m,
            "metres_per_shadow_px": self.ratio,
            "solidity": self.solidity,
            "shadow_zone_ground_fraction": self.shadow_zone_ground_fraction,
            "rejection_reason": self.rejection_reason,
        }


def _window(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    return (
        slice(max(0, rows.min() - margin), min(mask.shape[0], rows.max() + margin + 1)),
        slice(max(0, cols.min() - margin), min(mask.shape[1], cols.max() + margin + 1)),
    )


def _shadow_zone(mask: np.ndarray, sun_azimuth_deg: float, length_px: float) -> np.ndarray:
    """Pixels swept by the footprint moved 1..ceil(L) px along the shadow, minus itself."""
    d_row, d_col = shadow_direction_pixels(sun_azimuth_deg)
    zone = np.zeros_like(mask)
    n_rows, n_cols = mask.shape
    rows, cols = np.nonzero(mask)
    for t in range(1, int(math.ceil(length_px)) + 1):
        r = rows + int(round(d_row * t))
        c = cols + int(round(d_col * t))
        keep = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
        zone[r[keep], c[keep]] = True
    return zone & ~mask


def assess_reference_candidates(
    shadows: Iterable[BuildingShadow],
    *,
    agl: np.ndarray,
    agl_valid: np.ndarray,
    cls: np.ndarray,
    building_mask: np.ndarray,
    sun_azimuth_deg: float,
    surface_classes: Sequence[int],
    min_area_px: int,
    isolation_px: int,
    min_solidity: float,
    min_height_m: float,
    min_valid_fraction: float,
    min_shadow_zone_ground_fraction: float,
    max_relative_spread: float,
) -> list[ReferenceCandidate]:
    """Judge every building as a reference, and rank the eligible ones.

    Preference order among eligible buildings: smallest relative ray spread
    (most reliable shadow), then largest area, then building id. The ranking
    uses no ground truth beyond the eligibility gates, so it cannot select
    buildings for agreeing with the answer.

    The first failed gate is the recorded reason. Gates, in order: usable
    NOMINAL measurement, area, solidity (clean segmentation), isolation (no
    other building-class pixel within ``isolation_px``; 0 disables it, leaving
    contamination to the per-measurement ``merged_shadow`` check), valid AGL coverage,
    minimum height, relative ray spread, and a shadow zone lying on open,
    flat-labelled ground (``surface_classes``). AGL cannot show terrain slope --
    DFC2019 AGL has the terrain removed -- so "flat ground" is judged from the
    labels of the ground the shadow falls on.
    """
    candidates: list[ReferenceCandidate] = []
    surface = np.isin(cls, np.asarray(surface_classes))
    for shadow in shadows:
        mask = shadow.footprint.mask
        area = int(mask.sum())
        common = dict(
            building_id=shadow.building_id,
            area_px=area,
            shadow_length_px=shadow.shadow_length_px,
            spread_px=shadow.spread_px,
        )

        def reject(reason: str, **extra: Any) -> None:
            fields_ = {"h_ref_m": None, "solidity": None, "shadow_zone_ground_fraction": None}
            fields_.update(extra)
            candidates.append(ReferenceCandidate(**common, **fields_, rejection_reason=reason))

        if not shadow.usable:
            reject(f"shadow_not_usable: {shadow.rejection_reason}")
            continue
        if shadow.confidence is not Confidence.NOMINAL:
            reject("confidence_not_nominal")
            continue
        if area < min_area_px:
            reject("below_reference_min_area")
            continue
        rows, cols = _window(mask, isolation_px + 1)
        local = mask[rows, cols]
        solidity = float(local.sum() / convex_hull_image(local).sum())
        if solidity < min_solidity:
            reject("low_solidity", solidity=solidity)
            continue
        if isolation_px > 0:
            grown = ndimage.binary_dilation(local, iterations=int(isolation_px))
            if (grown & building_mask[rows, cols] & ~local).any():
                reject("not_isolated", solidity=solidity)
                continue
        valid = agl_valid & mask
        if valid.sum() < min_valid_fraction * area:
            reject("insufficient_valid_agl", solidity=solidity)
            continue
        h_ref = float(np.median(agl[valid]))
        if h_ref < min_height_m:
            reject("below_reference_min_height", solidity=solidity, h_ref_m=h_ref)
            continue
        relative = shadow.spread_px / shadow.shadow_length_px if shadow.spread_px is not None else None
        if relative is None or not math.isfinite(relative) or relative > max_relative_spread:
            reject("unreliable_shadow_spread", solidity=solidity, h_ref_m=h_ref)
            continue
        zone = _shadow_zone(mask, sun_azimuth_deg, float(shadow.shadow_length_px))
        ground_fraction = float(surface[zone].mean()) if zone.any() else 0.0
        if ground_fraction < min_shadow_zone_ground_fraction:
            reject(
                "shadow_zone_not_open_ground",
                solidity=solidity,
                h_ref_m=h_ref,
                shadow_zone_ground_fraction=ground_fraction,
            )
            continue
        candidates.append(
            ReferenceCandidate(
                **common,
                h_ref_m=h_ref,
                solidity=solidity,
                shadow_zone_ground_fraction=ground_fraction,
                rejection_reason=None,
            )
        )

    eligible = sorted(
        (c for c in candidates if c.eligible),
        key=lambda c: (c.relative_spread, -c.area_px, c.building_id),
    )
    return eligible + [c for c in candidates if not c.eligible]


# ---------------------------------------------------------------------------
# The sun scale
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowScale:
    """``metres_per_shadow_px = GSD * tan(theta)``: the only number heights need."""

    metres_per_shadow_px: float
    source: str
    reference_ids: tuple[str, ...] = ()
    reference_heights_m: tuple[float, ...] = ()
    reference_lengths_px: tuple[float, ...] = ()
    #: Per-reference ``h_ref / L_px``.
    reference_ratios: tuple[float, ...] = ()
    gsd_m: float | None = None
    note: str = ""

    @property
    def tan_theta(self) -> float | None:
        return self.metres_per_shadow_px / self.gsd_m if self.gsd_m else None

    @property
    def sun_elevation_deg(self) -> float | None:
        tan = self.tan_theta
        return math.degrees(math.atan(tan)) if tan is not None else None

    def to_dict(self) -> dict[str, Any]:
        per_reference_tan = (
            [r / self.gsd_m for r in self.reference_ratios] if self.gsd_m else None
        )
        return {
            "source": self.source,
            "metres_per_shadow_px": self.metres_per_shadow_px,
            "reference_ids": list(self.reference_ids),
            "reference_agl_m": list(self.reference_heights_m),
            "reference_shadow_lengths_px": list(self.reference_lengths_px),
            "reference_metres_per_shadow_px": list(self.reference_ratios),
            "gsd_m": self.gsd_m,
            "reference_tan_theta": per_reference_tan,
            "tan_theta": self.tan_theta,
            "sun_elevation_deg": self.sun_elevation_deg,
            "note": self.note,
        }


GSD_UNVERIFIED_NOTE = (
    "tan(theta) and theta in degrees are not determinable: the tile GSD is unverified. "
    "Shadow lengths are in pixels and every height uses metres_per_shadow_px = GSD*tan(theta), "
    "which the references determine without a GSD. theta would be an effective elevation "
    "(it absorbs off-nadir relief displacement)."
)


K_ESTIMATORS = ("median", "mean", "inverse_variance")


def estimate_k(ratios: Sequence[float], rel_spreads: Sequence[float] | None = None, how: str = "median") -> float:
    """Aggregate per-reference ``k_i = h_ref / L_px`` into one tile value.

    * ``median`` (the production choice) -- robust to one bad reference and
      always defined.
    * ``mean`` -- for comparison.
    * ``inverse_variance`` -- weights ``1 / (k_i * spread_i / L_i)^2``, the same
      proxy as ``dh``. Undefined when any reference has zero ray spread, and
      then raises rather than giving that reference infinite weight.
    """
    k = np.asarray(ratios, dtype=np.float64)
    if k.size == 0:
        raise ValueError("no reference ratios")
    if how == "median":
        return float(np.median(k))
    if how == "mean":
        return float(k.mean())
    if how == "inverse_variance":
        rel = np.asarray(rel_spreads, dtype=np.float64)
        sigma = k * rel
        if not (np.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError("inverse_variance k is undefined: a reference has zero or invalid ray spread")
        w = 1.0 / sigma**2
        return float((w * k).sum() / w.sum())
    raise ValueError(f"unknown k estimator {how!r}; choose from {K_ESTIMATORS}")


def scale_from_references(
    references: Sequence[ReferenceCandidate], *, gsd_m: float | None = None, estimator: str = "median"
) -> ShadowScale:
    """Aggregate reference buildings into one tile scale (:func:`estimate_k`)."""
    if not references:
        raise ValueError("no reference buildings")
    ratios = [c.ratio for c in references]
    if any(r is None or not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("every reference needs a positive h_ref and shadow length")
    return ShadowScale(
        metres_per_shadow_px=estimate_k(ratios, [c.relative_spread for c in references], estimator),
        source=f"{estimator} over {len(references)} DFC2019 reference building(s)",
        reference_ids=tuple(c.building_id for c in references),
        reference_heights_m=tuple(float(c.h_ref_m) for c in references),
        reference_lengths_px=tuple(float(c.shadow_length_px) for c in references),
        reference_ratios=tuple(float(r) for r in ratios),
        gsd_m=gsd_m,
        note="" if gsd_m else GSD_UNVERIFIED_NOTE,
    )


def scale_from_known_geometry(*, gsd_m: float, sun_elevation_deg: float) -> ShadowScale:
    """Scale from known metadata (synthetic control): ``GSD * tan(theta)``."""
    return ShadowScale(
        metres_per_shadow_px=float(gsd_m) * math.tan(math.radians(float(sun_elevation_deg))),
        source="known metadata (true sun elevation and GSD)",
        gsd_m=float(gsd_m),
    )


def explicit_grid_dl_px(sun_azimuth_deg: float) -> float:
    """Half the pixel-centre pitch along the shadow direction, in pixels.

    Rays read lengths between pixel centres, whose spacing along direction
    ``(d_row, d_col)`` is ``1 / max(|d_row|, |d_col|)`` px; half of it bounds
    the rounding of a length rendered on that grid. For the synthetic fixture
    this is derived from its construction (1/sqrt(2) px at azimuth 135 deg); it
    is not a property of real imagery.
    """
    d_row, d_col = shadow_direction_pixels(sun_azimuth_deg)
    return 0.5 / max(abs(d_row), abs(d_col))


# ---------------------------------------------------------------------------
# Shadow-derived height constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeightConstraint:
    building_id: str
    confidence: Confidence
    shadow_length_px: float | None
    spread_px: float | None
    height_m: float | None
    dh_m: float | None
    dh_source: str | None
    weight: float | None
    rejection_reason: str | None
    mask: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def usable(self) -> bool:
        return self.rejection_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "confidence": self.confidence.value,
            "shadow_length_px": self.shadow_length_px,
            "ray_length_spread_px": self.spread_px,
            "height_m": self.height_m,
            "dh_m": self.dh_m,
            "dh_source": self.dh_source,
            "weight": self.weight,
            "rejection_reason": self.rejection_reason,
        }


def shadow_height_constraints(
    shadows: Iterable[BuildingShadow],
    scale: ShadowScale,
    *,
    exclude_ids: Iterable[str] = (),
    uncertainty: str = UNCERTAINTY_RAY_SPREAD_PROXY,
    explicit_dl_px: float | None = None,
    reduced_weight_factor: float = 0.5,
) -> list[HeightConstraint]:
    """``h_i = metres_per_shadow_px * L_i,px`` for every non-calibration building.

    ``exclude_ids`` (the calibration buildings) are left out entirely.
    Weights are ``factor / dh^2`` with factor 1 for NOMINAL and
    ``reduced_weight_factor`` for REDUCED; FAILED never gets this far.
    """
    if uncertainty not in (UNCERTAINTY_EXPLICIT, UNCERTAINTY_RAY_SPREAD_PROXY):
        raise ValueError(f"unknown uncertainty mode {uncertainty!r}")
    excluded = set(exclude_ids)
    k = scale.metres_per_shadow_px
    constraints: list[HeightConstraint] = []
    for shadow in shadows:
        if shadow.building_id in excluded:
            continue
        base = dict(
            building_id=shadow.building_id,
            confidence=shadow.confidence,
            shadow_length_px=shadow.shadow_length_px,
            spread_px=shadow.spread_px,
            mask=shadow.footprint.mask,
        )
        if not shadow.usable:
            constraints.append(
                HeightConstraint(**base, height_m=None, dh_m=None, dh_source=None, weight=None,
                                 rejection_reason=shadow.rejection_reason)
            )
            continue
        if not (math.isfinite(k) and k > 0):
            constraints.append(
                HeightConstraint(**base, height_m=None, dh_m=None, dh_source=None, weight=None,
                                 rejection_reason="cannot_convert_to_metres")
            )
            continue
        height = k * float(shadow.shadow_length_px)
        if uncertainty == UNCERTAINTY_EXPLICIT:
            dl_px = explicit_dl_px
            missing = "no_explicit_dh"
        else:
            dl_px = shadow.spread_px
            missing = "no_valid_ray_spread_proxy"
        if dl_px is None or not math.isfinite(dl_px) or dl_px <= 0.0:
            constraints.append(
                HeightConstraint(**base, height_m=height, dh_m=None, dh_source=None, weight=None,
                                 rejection_reason=missing)
            )
            continue
        dh = k * float(dl_px)
        factor = 1.0 if shadow.confidence is Confidence.NOMINAL else reduced_weight_factor
        constraints.append(
            HeightConstraint(**base, height_m=height, dh_m=dh, dh_source=uncertainty,
                             weight=factor / dh**2, rejection_reason=None)
        )
    return constraints
