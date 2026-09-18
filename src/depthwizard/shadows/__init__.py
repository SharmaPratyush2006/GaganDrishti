"""Shadow extraction and measurement -- Phase 2.

The chain this package implements::

    image ──▶ ShadowDetector ──▶ shadow mask
                                     │
    supplied footprints ─────────────┤
                                     ▼
                            measure_shadow_length      L_px, then L_m = L_px * GSD_m
                                     │
                                     ▼
                     depthwizard.physics.height        h = L * tan(theta)

* :mod:`depthwizard.shadows.detector` -- the swappable
  :class:`~depthwizard.shadows.detector.ShadowDetector` interface, plus the
  Phase 2 :class:`~depthwizard.shadows.detector.ClassicalShadowDetector`
  (HSV value-channel thresholding).
* :mod:`depthwizard.shadows.measure` -- ray-cast shadow length measurement
  along the anti-sun direction, with failure reported rather than papered over.
* :mod:`depthwizard.shadows.pipeline` -- the two joined to the height equation.

Building footprints are **supplied**, never detected. There is no building
detector anywhere in this package.

Typical use::

    from depthwizard.shadows import (
        BuildingFootprint, DetectionContext, ShadowHeightPipeline,
    )

    footprints = [BuildingFootprint.from_bbox("b1", shape=image.shape[:2], ...)]
    context = DetectionContext(gsd_m=0.5, sun_elevation_deg=45, sun_azimuth_deg=135)
    result = ShadowHeightPipeline().run(image, footprints, context)
"""

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
    RayResult,
    ShadowLengthMeasurement,
    measure_shadow_length,
    measure_shadow_lengths,
)
from depthwizard.shadows.pipeline import (
    BuildingHeightEstimate,
    SceneHeightResult,
    ShadowHeightPipeline,
    estimate_heights_from_mask,
)

__all__ = [
    # detection
    "ShadowDetector",
    "ClassicalShadowDetector",
    "DetectionContext",
    "ShadowMask",
    # measurement
    "BuildingFootprint",
    "MeasurementParams",
    "DEFAULT_MEASUREMENT_PARAMS",
    "RayResult",
    "ShadowLengthMeasurement",
    "measure_shadow_length",
    "measure_shadow_lengths",
    # pipeline
    "BuildingHeightEstimate",
    "SceneHeightResult",
    "ShadowHeightPipeline",
    "estimate_heights_from_mask",
]
