"""The Phase 2 pipeline: image + footprints -> heights, with full diagnostics.

Four stages, each of which can be run and inspected on its own:

1. **Detect** -- a :class:`~depthwizard.shadows.detector.ShadowDetector` turns
   the image into a binary shadow mask.
2. **Direct** -- the shadow bearing is the anti-sun azimuth,
   ``(sun_azimuth + 180) % 360``.
3. **Measure** -- :func:`~depthwizard.shadows.measure.measure_shadow_length`
   casts rays from each **supplied** footprint along that bearing and returns
   ``L`` in pixels, then in metres via ``L_m = L_px * GSD_m``.
4. **Convert** -- :func:`~depthwizard.physics.height.height_from_shadow`
   applies ``h = L * tan(theta)``, propagates ``dh = tan(theta) * dL``, and
   runs the 25-45 degree sun-band gate.

Confidence is assembled, never assumed:

* the measurement failed          -> ``FAILED``, height is None
* the sun sat outside 25-45 deg   -> ``REDUCED``, with the band's reason
* too few rays found shadow       -> ``REDUCED``, with the measurement's reason
* neither                         -> ``NOMINAL``

A scene whose sun is outside the band is **not** rejected and processing is
**not** stopped. Every building still gets a height; the flag and the reason
travel with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from depthwizard.logging_setup import get_logger
from depthwizard.physics.height import (
    USABLE_SUN_ELEVATION_MAX_DEG,
    USABLE_SUN_ELEVATION_MIN_DEG,
    Confidence,
    HeightEstimate,
    height_from_shadow,
)
from depthwizard.physics.sun import shadow_azimuth_deg
from depthwizard.shadows.detector import (
    ClassicalShadowDetector,
    DetectionContext,
    ShadowDetector,
    ShadowMask,
)
from depthwizard.shadows.measure import (
    DEFAULT_MEASUREMENT_PARAMS,
    BuildingFootprint,
    MeasurementParams,
    ShadowLengthMeasurement,
    measure_shadow_lengths,
)

__all__ = [
    "BuildingHeightEstimate",
    "SceneHeightResult",
    "ShadowHeightPipeline",
    "estimate_heights_from_mask",
]

log = get_logger(__name__)


@dataclass(frozen=True)
class BuildingHeightEstimate:
    """One building's height, or one building's stated failure to produce one.

    ``height`` is None exactly when ``confidence`` is
    :attr:`~depthwizard.physics.height.Confidence.FAILED`.
    """

    building_id: str
    measurement: ShadowLengthMeasurement
    height: HeightEstimate | None
    confidence: Confidence
    reduced_confidence_reasons: tuple[str, ...] = ()
    failure_reason: str | None = None

    @property
    def height_m(self) -> float | None:
        return self.height.height_m if self.height else None

    @property
    def height_uncertainty_m(self) -> float | None:
        return self.height.height_uncertainty_m if self.height else None

    @property
    def shadow_length_px(self) -> float | None:
        return self.measurement.shadow_length_px

    @property
    def shadow_length_m(self) -> float | None:
        return self.measurement.shadow_length_m

    def to_dict(self) -> dict[str, Any]:
        """Every number behind this estimate, flat, for logs and reports."""
        record: dict[str, Any] = {
            "building_id": self.building_id,
            "confidence_flag": self.confidence.value,
            "reduced_confidence_reasons": list(self.reduced_confidence_reasons),
            "measurement_failure_reason": self.failure_reason,
        }
        record.update(self.measurement.to_dict())
        if self.height is not None:
            record.update(self.height.to_dict())
        else:
            record.update(
                {
                    "height_m": None,
                    "height_uncertainty_m": None,
                    "tan_theta": None,
                    "equation": None,
                }
            )
        # The outer flags win: measurement.to_dict() has no confidence, but
        # height.to_dict() does, and a FAILED estimate has no height at all.
        record["confidence_flag"] = self.confidence.value
        record["reduced_confidence_reasons"] = list(self.reduced_confidence_reasons)
        return record


@dataclass(frozen=True)
class SceneHeightResult:
    """Everything one run of the pipeline produced."""

    estimates: tuple[BuildingHeightEstimate, ...]
    shadow_mask: ShadowMask
    context: DetectionContext
    sun_elevation_deg: float
    sun_azimuth_deg: float
    shadow_azimuth_deg: float
    gsd_m: float

    def __iter__(self):
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    @property
    def successful(self) -> tuple[BuildingHeightEstimate, ...]:
        return tuple(e for e in self.estimates if e.height is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sun_elevation_deg": self.sun_elevation_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "shadow_azimuth_deg": self.shadow_azimuth_deg,
            "gsd_m": self.gsd_m,
            "detector": self.shadow_mask.to_dict(),
            "context": self.context.to_dict(),
            "buildings": [e.to_dict() for e in self.estimates],
        }


# ---------------------------------------------------------------------------
# Measurement -> height
# ---------------------------------------------------------------------------


def _estimate_from_measurement(
    measurement: ShadowLengthMeasurement,
    *,
    sun_band_min_deg: float,
    sun_band_max_deg: float,
) -> BuildingHeightEstimate:
    if not measurement.ok or measurement.shadow_length_m is None:
        return BuildingHeightEstimate(
            building_id=measurement.building_id,
            measurement=measurement,
            height=None,
            confidence=Confidence.FAILED,
            reduced_confidence_reasons=measurement.reduced_confidence_reasons,
            failure_reason=measurement.failure_reason,
        )

    estimate = height_from_shadow(
        measurement.shadow_length_m,
        measurement.sun_elevation_deg,
        shadow_length_uncertainty_m=measurement.shadow_length_uncertainty_m,
        extra_reduced_confidence_reasons=measurement.reduced_confidence_reasons,
        sun_band_min_deg=sun_band_min_deg,
        sun_band_max_deg=sun_band_max_deg,
    )
    return BuildingHeightEstimate(
        building_id=measurement.building_id,
        measurement=measurement,
        height=estimate,
        confidence=estimate.confidence,
        reduced_confidence_reasons=estimate.reduced_confidence_reasons,
        failure_reason=None,
    )


def estimate_heights_from_mask(
    footprints: Iterable[BuildingFootprint],
    shadow_mask: np.ndarray,
    *,
    gsd_m: float,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    measurement_uncertainty_px: float | None = None,
    params: MeasurementParams = DEFAULT_MEASUREMENT_PARAMS,
    occluder_mask: np.ndarray | None = None,
    sun_band_min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
    sun_band_max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
) -> list[BuildingHeightEstimate]:
    """Measure and convert, starting from a shadow mask you already have.

    Use this when the mask comes from somewhere other than a
    :class:`~depthwizard.shadows.detector.ShadowDetector` -- a hand-drawn mask,
    a fixture's ground truth, or a future learned segmenter run separately.
    """
    measurements = measure_shadow_lengths(
        footprints,
        shadow_mask,
        gsd_m=gsd_m,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        measurement_uncertainty_px=measurement_uncertainty_px,
        params=params,
        occluder_mask=occluder_mask,
    )
    return [
        _estimate_from_measurement(
            m, sun_band_min_deg=sun_band_min_deg, sun_band_max_deg=sun_band_max_deg
        )
        for m in measurements
    ]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class ShadowHeightPipeline:
    """Detect shadows, measure them, and convert them to heights.

    Args:
        detector: Any :class:`~depthwizard.shadows.detector.ShadowDetector`.
            Defaults to the Phase 2
            :class:`~depthwizard.shadows.detector.ClassicalShadowDetector`.
            Swap in a different one and nothing else changes.
        params: Measurement parameters. See
            :class:`~depthwizard.shadows.measure.MeasurementParams`.
        measurement_uncertainty_px: ``dL`` in pixels, applied to every building
            in the run. **Supply this only when you have a real number for it.**
            Left as None, the heights carry no uncertainty rather than a made-up
            one.
        sun_band_min_deg / sun_band_max_deg: Edges of the confidence band.

    Example::

        pipeline = ShadowHeightPipeline()
        result = pipeline.run(image, footprints, context)
        for estimate in result:
            print(estimate.building_id, estimate.height_m, estimate.confidence)
    """

    def __init__(
        self,
        detector: ShadowDetector | None = None,
        *,
        params: MeasurementParams = DEFAULT_MEASUREMENT_PARAMS,
        measurement_uncertainty_px: float | None = None,
        sun_band_min_deg: float = USABLE_SUN_ELEVATION_MIN_DEG,
        sun_band_max_deg: float = USABLE_SUN_ELEVATION_MAX_DEG,
    ) -> None:
        self.detector = detector if detector is not None else ClassicalShadowDetector()
        self.params = params
        self.measurement_uncertainty_px = measurement_uncertainty_px
        self.sun_band_min_deg = sun_band_min_deg
        self.sun_band_max_deg = sun_band_max_deg

    def run(
        self,
        image: np.ndarray,
        footprints: Sequence[BuildingFootprint],
        context: DetectionContext,
        *,
        gsd_m: float | None = None,
        sun_azimuth_deg: float | None = None,
        sun_elevation_deg: float | None = None,
    ) -> SceneHeightResult:
        """Run the full pipeline over one scene.

        Args:
            image: The scene pixels.
            footprints: **Supplied** building footprints. Nothing here detects
                buildings.
            context: Scene geometry, normally built from Phase 1 metadata with
                :meth:`~depthwizard.shadows.detector.DetectionContext.from_scene_metadata`.
            gsd_m, sun_azimuth_deg, sun_elevation_deg: Override the
                corresponding ``context`` fields. Useful when the geometry is
                known but the context was built for detection only.

        Returns:
            A :class:`SceneHeightResult`.

        Raises:
            ValueError: if the GSD or either sun angle is still unknown after
                the overrides. Metric heights need all three, and guessing one
                would silently invent a scale.
        """
        gsd = gsd_m if gsd_m is not None else context.gsd_m
        azimuth = sun_azimuth_deg if sun_azimuth_deg is not None else context.sun_azimuth_deg
        elevation = (
            sun_elevation_deg if sun_elevation_deg is not None else context.sun_elevation_deg
        )
        missing = [
            name
            for name, value in (
                ("gsd_m", gsd),
                ("sun_azimuth_deg", azimuth),
                ("sun_elevation_deg", elevation),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"cannot estimate metric heights without {', '.join(missing)}; "
                "supply them on the DetectionContext or as run() overrides "
                "(a scene missing them is Mode.RELATIVE and has no metric scale)"
            )

        mask = self.detector.detect(image, context)
        for footprint in footprints:
            if mask.shape != footprint.mask.shape:
                raise ValueError(
                    f"shadow mask shape {mask.shape} does not match footprint "
                    f"{footprint.building_id!r} shape {footprint.mask.shape}"
                )

        estimates = estimate_heights_from_mask(
            footprints,
            mask.mask,
            gsd_m=float(gsd),
            sun_azimuth_deg=float(azimuth),
            sun_elevation_deg=float(elevation),
            measurement_uncertainty_px=self.measurement_uncertainty_px,
            params=self.params,
            sun_band_min_deg=self.sun_band_min_deg,
            sun_band_max_deg=self.sun_band_max_deg,
        )

        result = SceneHeightResult(
            estimates=tuple(estimates),
            shadow_mask=mask,
            context=context,
            sun_elevation_deg=float(elevation),
            sun_azimuth_deg=float(azimuth) % 360.0,
            shadow_azimuth_deg=shadow_azimuth_deg(float(azimuth)),
            gsd_m=float(gsd),
        )
        log.info(
            "estimated building heights from shadows",
            extra={
                "buildings": len(result),
                "succeeded": len(result.successful),
                "detector": self.detector.name,
                "sun_elevation_deg": result.sun_elevation_deg,
                "shadow_azimuth_deg": result.shadow_azimuth_deg,
            },
        )
        return result
