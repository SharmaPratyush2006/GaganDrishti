"""Shadow length measurement along the anti-sun direction.

Given a **supplied** building footprint and a shadow mask, measure how far the
building's shadow reaches, in pixels and then in metres.

No building detection happens here. Footprints come from the caller -- drawn by
hand, taken from a cadastral layer, or (for the synthetic fixture) read out of
the ground truth sidecar. Automatic footprint extraction is out of scope for
Phase 2.

Why rays, and why many of them
------------------------------
A real shadow is not a clean line. It is broken by cars and trees, blended into
dark asphalt, cut off by a neighbouring building, and ragged at its tip. Taking
"the longest run of dark pixels" from one point would pick up whichever single
artefact happened to be longest.

So the measurement casts **many rays**:

1. The shadow direction is the anti-sun bearing,
   :func:`~depthwizard.physics.sun.shadow_azimuth_deg`, converted to a unit
   step in ``(row, col)`` index space by
   :func:`~depthwizard.physics.sun.shadow_direction_pixels`.
2. Every footprint pixel that is **shadow-facing** -- one whose next step along
   that direction leaves the footprint -- launches a ray. For a sun in the
   south-east that is the building's north and west edges, picked out by
   geometry rather than hard-coded.
3. Each ray is sampled at sub-pixel steps. Samples that land back inside the
   footprint are skipped (a roof is not its own shadow, and is not a gap
   either). Shadow samples extend the ray; non-shadow samples open a gap, and
   the ray only stops once that gap exceeds ``gap_tolerance_px``. Small holes
   in the mask are therefore tolerated, large ones terminate the ray.
4. A ray's length is the distance from its launch pixel's centre to the centre
   of its **last** shadow pixel, projected onto the shadow direction. Both ends
   are pixel centres, so the half-pixel conventions cancel and the result does
   not depend on the sampling step.
5. The per-building length is a **robust percentile** (the median by default)
   over the rays that found shadow. One clipped ray, or one that ran down a
   dark alley, cannot move a median.

Failure is a result, not a fallback
-----------------------------------
If too few rays find shadow -- an occluded building, a wrong sun azimuth, a
mask that missed -- the measurement returns ``ok=False`` with a stated reason
and ``shadow_length_px=None``. It never falls back to a nominal value, and it
never substitutes a prior. A missing measurement is recoverable information; a
fabricated one is not.

Uncertainty
-----------
``measurement_uncertainty_px`` is supplied by the caller and converted with the
same GSD as the length itself. Nothing here estimates an uncertainty from the
data. The observed spread between rays *is* reported, as
``ray_length_spread_px``, but it is a diagnostic describing this measurement --
not a calibrated error bar, and it is never used as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterable, Sequence

import numpy as np

from depthwizard.logging_setup import get_logger
from depthwizard.physics.height import pixels_to_metres
from depthwizard.physics.sun import shadow_azimuth_deg, shadow_direction_pixels

__all__ = [
    "BuildingFootprint",
    "MeasurementParams",
    "RayResult",
    "ShadowLengthMeasurement",
    "measure_shadow_length",
    "measure_shadow_lengths",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildingFootprint:
    """One supplied building footprint, as a boolean mask over the scene.

    A mask rather than a polygon, because a mask composes directly with the
    shadow mask and needs no rasterisation step that could disagree with the
    one that made the image.

    Args:
        building_id: Stable identifier, carried through to every diagnostic and
            evaluation record.
        mask: Boolean array, ``(rows, cols)``, True inside the footprint.
    """

    building_id: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
            object.__setattr__(self, "mask", mask)
        if mask.ndim != 2:
            raise ValueError(f"footprint mask must be 2-D, got shape {mask.shape}")
        if not mask.any():
            raise ValueError(f"footprint {self.building_id!r} is empty")

    @classmethod
    def from_bbox(
        cls,
        building_id: str,
        *,
        shape: tuple[int, int],
        row_min: int,
        row_max: int,
        col_min: int,
        col_max: int,
    ) -> "BuildingFootprint":
        """Build a rectangular footprint from half-open pixel bounds.

        The bounds are half-open like a slice, matching
        :class:`~depthwizard.ingest.synthetic.FootprintPixels`, so a footprint
        from the synthetic fixture's ground truth converts without an off-by-one.
        """
        mask = np.zeros(shape, dtype=bool)
        mask[int(row_min) : int(row_max), int(col_min) : int(col_max)] = True
        return cls(building_id=building_id, mask=mask)

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """``(row_min, row_max, col_min, col_max)``, half-open."""
        rows, cols = np.nonzero(self.mask)
        return (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)

    @property
    def centroid_rc(self) -> tuple[float, float]:
        rows, cols = np.nonzero(self.mask)
        return (float(rows.mean()), float(cols.mean()))


@dataclass(frozen=True)
class MeasurementParams:
    """Everything that shapes a measurement, in one inspectable object.

    Args:
        step_px: Sampling step along a ray, in pixels. Must be <= 0.5 so that no
            pixel on a diagonal ray is stepped over.
        gap_tolerance_px: How long a run of non-shadow a ray may cross before it
            gives up. This is what makes a broken shadow measurable; it is also
            what stops a ray from walking across the whole scene, so it is kept
            small.
        max_length_px: Hard cap on how far a ray travels. None means the image
            diagonal, i.e. no effective cap.
        min_rays: Minimum number of launch pixels. Below this the footprint is
            too small to average over and the measurement fails.
        min_hit_fraction: Minimum share of rays that must find shadow for the
            measurement to succeed at all.
        nominal_hit_fraction: Share of rays that must find shadow for the result
            to keep :attr:`~depthwizard.physics.height.Confidence.NOMINAL`.
            Between this and ``min_hit_fraction`` the measurement succeeds with
            a recorded reduced-confidence reason.
        percentile: Robust aggregator over per-ray lengths. 50 is the median.
        max_rays: Cap on launch pixels, subsampled with an even stride, so a
            city-block footprint does not cast ten thousand rays.
    """

    step_px: float = 0.5
    gap_tolerance_px: float = 2.0
    max_length_px: float | None = None
    min_rays: int = 3
    min_hit_fraction: float = 0.5
    nominal_hit_fraction: float = 0.8
    percentile: float = 50.0
    max_rays: int = 512

    def __post_init__(self) -> None:
        if not 0.0 < self.step_px <= 0.5:
            raise ValueError(
                f"step_px must be in (0, 0.5] so diagonal rays skip no pixel, got {self.step_px}"
            )
        if self.gap_tolerance_px < 0.0:
            raise ValueError(f"gap_tolerance_px must be >= 0, got {self.gap_tolerance_px}")
        if self.max_length_px is not None and self.max_length_px <= 0.0:
            raise ValueError(f"max_length_px must be positive or None, got {self.max_length_px}")
        if self.min_rays < 1:
            raise ValueError(f"min_rays must be >= 1, got {self.min_rays}")
        if not 0.0 < self.min_hit_fraction <= 1.0:
            raise ValueError(f"min_hit_fraction must be in (0, 1], got {self.min_hit_fraction}")
        if not self.min_hit_fraction <= self.nominal_hit_fraction <= 1.0:
            raise ValueError(
                "need min_hit_fraction <= nominal_hit_fraction <= 1, got "
                f"{self.min_hit_fraction} and {self.nominal_hit_fraction}"
            )
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {self.percentile}")
        if self.max_rays < 1:
            raise ValueError(f"max_rays must be >= 1, got {self.max_rays}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_px": self.step_px,
            "gap_tolerance_px": self.gap_tolerance_px,
            "max_length_px": self.max_length_px,
            "min_rays": self.min_rays,
            "min_hit_fraction": self.min_hit_fraction,
            "nominal_hit_fraction": self.nominal_hit_fraction,
            "percentile": self.percentile,
            "max_rays": self.max_rays,
        }


DEFAULT_MEASUREMENT_PARAMS = MeasurementParams()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RayResult:
    """One ray cast from the footprint boundary along the shadow direction."""

    start_row: int
    start_col: int
    hit: bool
    #: Distance from the launch pixel centre to the last shadow pixel centre,
    #: projected onto the shadow direction. None when the ray found no shadow.
    length_px: float | None
    #: Number of samples that landed on a shadow pixel.
    shadow_samples: int
    #: Why the walk stopped: ``"gap"``, ``"image_edge"`` or ``"max_length"``.
    terminated_by: str


@dataclass(frozen=True)
class ShadowLengthMeasurement:
    """A shadow length, in pixels and metres, with its full derivation.

    On failure ``ok`` is False, the lengths are None and ``failure_reason``
    says what was insufficient. There is never a substituted value.
    """

    building_id: str
    ok: bool
    shadow_length_px: float | None
    shadow_length_m: float | None
    gsd_m: float
    sun_elevation_deg: float
    sun_azimuth_deg: float
    shadow_azimuth_deg: float
    direction_px: tuple[float, float]
    n_rays: int
    n_rays_hit: int
    hit_fraction: float
    aggregation: str
    params: MeasurementParams
    #: ``dL`` in pixels exactly as supplied by the caller, or None.
    measurement_uncertainty_px: float | None = None
    #: ``dL_m = dL_px * GSD_m``, or None when no uncertainty was supplied.
    shadow_length_uncertainty_m: float | None = None
    #: Observed inter-ray dispersion (p84 - p16), a diagnostic only.
    ray_length_spread_px: float | None = None
    failure_reason: str | None = None
    reduced_confidence_reasons: tuple[str, ...] = ()
    rays: tuple[RayResult, ...] = field(default=(), repr=False)

    @property
    def hit_lengths_px(self) -> tuple[float, ...]:
        return tuple(r.length_px for r in self.rays if r.hit and r.length_px is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "measurement_ok": self.ok,
            "gsd_m": self.gsd_m,
            "sun_elevation_deg": self.sun_elevation_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "shadow_azimuth_deg": self.shadow_azimuth_deg,
            "shadow_direction_d_row": self.direction_px[0],
            "shadow_direction_d_col": self.direction_px[1],
            "shadow_length_px": self.shadow_length_px,
            "shadow_length_m": self.shadow_length_m,
            "measurement_uncertainty_px": self.measurement_uncertainty_px,
            "shadow_length_uncertainty_m": self.shadow_length_uncertainty_m,
            "n_rays": self.n_rays,
            "n_rays_hit": self.n_rays_hit,
            "hit_fraction": self.hit_fraction,
            "aggregation": self.aggregation,
            "ray_length_spread_px": self.ray_length_spread_px,
            "measurement_failure_reason": self.failure_reason,
            "measurement_reduced_confidence_reasons": list(self.reduced_confidence_reasons),
            "params": self.params.to_dict(),
        }


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------


def _launch_pixels(
    footprint_mask: np.ndarray, direction: tuple[float, float], max_rays: int
) -> np.ndarray:
    """Footprint pixels whose next step along the shadow direction leaves it.

    Shifting the mask one pixel *backwards* along the shadow direction and
    keeping what no longer overlaps is the same test done for the whole
    footprint at once: a pixel survives exactly when the pixel one step ahead of
    it is outside the footprint. That is the shadow-facing boundary, derived
    from the sun azimuth alone -- no edge is ever hard-coded.

    Returns an ``(n, 2)`` array of ``(row, col)``, stride-subsampled to at most
    ``max_rays`` entries.
    """
    d_row, d_col = direction
    step_row = int(round(d_row))
    step_col = int(round(d_col))

    n_rows, n_cols = footprint_mask.shape
    # ahead[r, c] == footprint[r + step_row, c + step_col], padded with False so
    # that stepping off the image counts as leaving the footprint.
    ahead = np.zeros_like(footprint_mask)
    src_r0, src_r1 = max(0, step_row), min(n_rows, n_rows + step_row)
    dst_r0, dst_r1 = max(0, -step_row), min(n_rows, n_rows - step_row)
    src_c0, src_c1 = max(0, step_col), min(n_cols, n_cols + step_col)
    dst_c0, dst_c1 = max(0, -step_col), min(n_cols, n_cols - step_col)
    if src_r1 > src_r0 and src_c1 > src_c0:
        ahead[dst_r0:dst_r1, dst_c0:dst_c1] = footprint_mask[src_r0:src_r1, src_c0:src_c1]

    boundary = footprint_mask & ~ahead
    rows, cols = np.nonzero(boundary)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    points = np.stack([rows, cols], axis=1)
    if points.shape[0] > max_rays:
        stride = int(math.ceil(points.shape[0] / max_rays))
        points = points[::stride][:max_rays]
    return points


def _cast_ray(
    shadow_mask: np.ndarray,
    skip_mask: np.ndarray,
    start: tuple[int, int],
    direction: tuple[float, float],
    params: MeasurementParams,
    max_length_px: float,
) -> RayResult:
    """Walk one ray and report how far the shadow reached along it."""
    d_row, d_col = direction
    start_row, start_col = start
    n_rows, n_cols = shadow_mask.shape

    last_hit: tuple[int, int] | None = None
    shadow_samples = 0
    gap_px = 0.0
    terminated_by = "max_length"

    t = params.step_px
    while t <= max_length_px:
        row = int(round(start_row + d_row * t))
        col = int(round(start_col + d_col * t))
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            terminated_by = "image_edge"
            break
        if skip_mask[row, col]:
            # Inside the building that cast this shadow: neither evidence nor gap.
            t += params.step_px
            continue
        if shadow_mask[row, col]:
            last_hit = (row, col)
            shadow_samples += 1
            gap_px = 0.0
        else:
            gap_px += params.step_px
            if gap_px > params.gap_tolerance_px:
                terminated_by = "gap"
                break
        t += params.step_px

    if last_hit is None:
        return RayResult(
            start_row=start_row,
            start_col=start_col,
            hit=False,
            length_px=None,
            shadow_samples=0,
            terminated_by=terminated_by,
        )

    # Project centre-to-centre onto the shadow direction. Both endpoints are
    # pixel centres, so the half-pixel at each end cancels.
    length_px = (last_hit[0] - start_row) * d_row + (last_hit[1] - start_col) * d_col
    return RayResult(
        start_row=start_row,
        start_col=start_col,
        hit=True,
        length_px=float(length_px),
        shadow_samples=shadow_samples,
        terminated_by=terminated_by,
    )


# ---------------------------------------------------------------------------
# The measurement API
# ---------------------------------------------------------------------------


def _failed(
    footprint: BuildingFootprint,
    reason: str,
    *,
    gsd_m: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    shadow_bearing: float,
    direction: tuple[float, float],
    params: MeasurementParams,
    measurement_uncertainty_px: float | None,
    n_rays: int = 0,
    n_rays_hit: int = 0,
    rays: Sequence[RayResult] = (),
) -> ShadowLengthMeasurement:
    return ShadowLengthMeasurement(
        building_id=footprint.building_id,
        ok=False,
        shadow_length_px=None,
        shadow_length_m=None,
        gsd_m=gsd_m,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        shadow_azimuth_deg=shadow_bearing,
        direction_px=direction,
        n_rays=n_rays,
        n_rays_hit=n_rays_hit,
        hit_fraction=(n_rays_hit / n_rays) if n_rays else 0.0,
        aggregation=f"p{params.percentile:g}",
        params=params,
        measurement_uncertainty_px=measurement_uncertainty_px,
        shadow_length_uncertainty_m=None,
        failure_reason=reason,
        rays=tuple(rays),
    )


def measure_shadow_length(
    footprint: BuildingFootprint,
    shadow_mask: np.ndarray,
    *,
    gsd_m: float,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    measurement_uncertainty_px: float | None = None,
    params: MeasurementParams = DEFAULT_MEASUREMENT_PARAMS,
    occluder_mask: np.ndarray | None = None,
) -> ShadowLengthMeasurement:
    """Measure one building's shadow length along the anti-sun direction.

    Args:
        footprint: The **supplied** building footprint. Not detected here.
        shadow_mask: Boolean shadow mask over the same grid as the footprint,
            typically ``ShadowMask.mask`` from a
            :class:`~depthwizard.shadows.detector.ShadowDetector`.
        gsd_m: Ground sample distance in metres per pixel, used for the one
            pixel-to-metre conversion: ``L_m = L_px * GSD_m``.
        sun_azimuth_deg: Compass bearing of the **sun**, degrees clockwise from
            North. The shadow runs along ``(sun_azimuth + 180) % 360``.
        sun_elevation_deg: Sun elevation above the horizon, in (0, 90]. Recorded
            for diagnostics; the height equation that consumes it lives in
            :mod:`depthwizard.physics.height`.
        measurement_uncertainty_px: ``dL`` in pixels, if the caller has one.
            Converted to metres with the same GSD. Omit it and both uncertainty
            fields stay None -- nothing is invented.
        params: See :class:`MeasurementParams`.
        occluder_mask: Optional mask of pixels that are neither ground nor this
            building's shadow -- most usefully, the other buildings in the
            scene. Rays skip these pixels instead of treating them as a gap, so
            a shadow that runs behind a neighbour is not cut short by it.

    Returns:
        A :class:`ShadowLengthMeasurement`. Check ``ok`` before using the
        lengths; on failure they are None and ``failure_reason`` explains why.

    Raises:
        ValueError: on a non-positive GSD, a shape mismatch between the masks,
            or an invalid ``measurement_uncertainty_px``.
    """
    shadow_mask = np.asarray(shadow_mask)
    if shadow_mask.dtype != np.bool_:
        shadow_mask = shadow_mask.astype(bool)
    if shadow_mask.shape != footprint.mask.shape:
        raise ValueError(
            f"shadow mask shape {shadow_mask.shape} does not match footprint "
            f"{footprint.building_id!r} shape {footprint.mask.shape}"
        )
    gsd = float(gsd_m)
    if not math.isfinite(gsd) or gsd <= 0.0:
        raise ValueError(f"gsd_m must be positive and finite, got {gsd_m}")
    if measurement_uncertainty_px is not None:
        uncertainty_px = float(measurement_uncertainty_px)
        if not math.isfinite(uncertainty_px) or uncertainty_px < 0.0:
            raise ValueError(
                f"measurement_uncertainty_px must be finite and >= 0, got {measurement_uncertainty_px}"
            )
    else:
        uncertainty_px = None

    # Direction: anti-sun bearing, converted to a unit step in (row, col).
    shadow_bearing = shadow_azimuth_deg(sun_azimuth_deg)
    direction = shadow_direction_pixels(sun_azimuth_deg)

    n_rows, n_cols = shadow_mask.shape
    max_length_px = (
        params.max_length_px if params.max_length_px is not None else math.hypot(n_rows, n_cols)
    )

    fail = partial(
        _failed,
        footprint,
        gsd_m=gsd,
        sun_elevation_deg=float(sun_elevation_deg),
        sun_azimuth_deg=float(sun_azimuth_deg) % 360.0,
        shadow_bearing=shadow_bearing,
        direction=direction,
        params=params,
        measurement_uncertainty_px=uncertainty_px,
    )

    launch = _launch_pixels(footprint.mask, direction, params.max_rays)
    if launch.shape[0] < params.min_rays:
        return fail(
            f"only {launch.shape[0]} shadow-facing boundary pixel(s) available, "
            f"need at least {params.min_rays}; the footprint is too small at this GSD",
            n_rays=int(launch.shape[0]),
        )

    skip_mask = footprint.mask
    if occluder_mask is not None:
        occluder = np.asarray(occluder_mask).astype(bool)
        if occluder.shape != footprint.mask.shape:
            raise ValueError(
                f"occluder_mask shape {occluder.shape} does not match "
                f"footprint shape {footprint.mask.shape}"
            )
        skip_mask = footprint.mask | occluder

    rays = [
        _cast_ray(shadow_mask, skip_mask, (int(r), int(c)), direction, params, max_length_px)
        for r, c in launch
    ]
    hit_lengths = [ray.length_px for ray in rays if ray.hit and ray.length_px is not None]
    n_rays = len(rays)
    n_hit = len(hit_lengths)
    hit_fraction = n_hit / n_rays

    if hit_fraction < params.min_hit_fraction:
        return fail(
            f"only {n_hit}/{n_rays} rays ({hit_fraction:.1%}) found shadow, below the "
            f"{params.min_hit_fraction:.0%} minimum; shadow evidence is insufficient",
            n_rays=n_rays,
            n_rays_hit=n_hit,
            rays=rays,
        )

    lengths = np.asarray(hit_lengths, dtype=np.float64)
    length_px = float(np.percentile(lengths, params.percentile))
    spread_px = float(np.percentile(lengths, 84.0) - np.percentile(lengths, 16.0))

    if length_px <= 0.0:
        return fail(
            f"aggregated shadow length is {length_px:.3f} px, which is not positive; "
            "the shadow-facing boundary found no shadow beyond itself",
            n_rays=n_rays,
            n_rays_hit=n_hit,
            rays=rays,
        )

    length_m = pixels_to_metres(length_px, gsd)
    uncertainty_m = pixels_to_metres(uncertainty_px, gsd) if uncertainty_px is not None else None

    reasons: list[str] = []
    if hit_fraction < params.nominal_hit_fraction:
        reasons.append(
            f"only {n_hit}/{n_rays} rays ({hit_fraction:.1%}) found shadow, below the "
            f"{params.nominal_hit_fraction:.0%} nominal threshold; the shadow is "
            "partially occluded or partially missed by the mask"
        )

    log.info(
        "measured shadow length",
        extra={
            "building_id": footprint.building_id,
            "shadow_azimuth_deg": shadow_bearing,
            "shadow_length_px": length_px,
            "shadow_length_m": length_m,
            "rays": n_rays,
            "rays_hit": n_hit,
        },
    )

    return ShadowLengthMeasurement(
        building_id=footprint.building_id,
        ok=True,
        shadow_length_px=length_px,
        shadow_length_m=length_m,
        gsd_m=gsd,
        sun_elevation_deg=float(sun_elevation_deg),
        sun_azimuth_deg=float(sun_azimuth_deg) % 360.0,
        shadow_azimuth_deg=shadow_bearing,
        direction_px=direction,
        n_rays=n_rays,
        n_rays_hit=n_hit,
        hit_fraction=hit_fraction,
        aggregation=f"p{params.percentile:g}",
        params=params,
        measurement_uncertainty_px=uncertainty_px,
        shadow_length_uncertainty_m=uncertainty_m,
        ray_length_spread_px=spread_px,
        failure_reason=None,
        reduced_confidence_reasons=tuple(reasons),
        rays=tuple(rays),
    )


def measure_shadow_lengths(
    footprints: Iterable[BuildingFootprint],
    shadow_mask: np.ndarray,
    *,
    gsd_m: float,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    measurement_uncertainty_px: float | None = None,
    params: MeasurementParams = DEFAULT_MEASUREMENT_PARAMS,
    occluder_mask: np.ndarray | None = None,
    exclude_other_footprints: bool = True,
) -> list[ShadowLengthMeasurement]:
    """Measure a whole scene's footprints.

    Args:
        exclude_other_footprints: When True (the default), every other supplied
            footprint is added to each building's occluder mask, so a shadow
            passing behind a neighbour is skipped over rather than cut short.

    Other arguments are as :func:`measure_shadow_length`.
    """
    footprints = list(footprints)
    combined: np.ndarray | None = None
    if exclude_other_footprints and footprints:
        combined = np.zeros_like(footprints[0].mask, dtype=bool)
        for footprint in footprints:
            combined |= footprint.mask

    results: list[ShadowLengthMeasurement] = []
    for footprint in footprints:
        others = None
        if combined is not None:
            others = combined & ~footprint.mask
        if occluder_mask is not None:
            extra = np.asarray(occluder_mask).astype(bool)
            others = extra if others is None else (others | extra)
        results.append(
            measure_shadow_length(
                footprint,
                shadow_mask,
                gsd_m=gsd_m,
                sun_azimuth_deg=sun_azimuth_deg,
                sun_elevation_deg=sun_elevation_deg,
                measurement_uncertainty_px=measurement_uncertainty_px,
                params=params,
                occluder_mask=others,
            )
        )
    return results
