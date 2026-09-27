"""Phase 4a: sun azimuth recovered from the SYNTHETIC fixture's shadows.

The fixture is rendered at a known sun azimuth, so the recovered value can be
checked exactly. The tolerance is the estimator's stated quantisation bound,
``atan(0.5 / band_px)`` -- a band of ``band_px`` pixels cannot resolve a
smaller change of direction.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from depthwizard.calibration.azimuth import (
    angular_difference_deg,
    angular_resolution_deg,
    estimate_sun_azimuth,
)
from depthwizard.ingest.synthetic import render_scene
from depthwizard.physics.sun import shadow_azimuth_deg
from depthwizard.shadows.detector import ClassicalShadowDetector, DetectionContext

BAND_PX = 8


@pytest.mark.parametrize("true_azimuth", [45.0, 90.0, 135.0, 200.0, 290.0])
def test_recovers_synthetic_sun_azimuth(scene, true_azimuth: float) -> None:
    render = render_scene(replace(scene, sun=replace(scene.sun, azimuth_deg=true_azimuth)))
    image = np.clip(np.rint(render.reflectance * 255.0), 0, 255).astype(np.uint8)
    shadow = ClassicalShadowDetector().detect(image, DetectionContext()).mask

    estimate = estimate_sun_azimuth(shadow, render.building_mask, band_px=BAND_PX)

    assert estimate.ok, estimate.failure_reason
    error = angular_difference_deg(estimate.sun_azimuth_deg, true_azimuth)
    assert error <= angular_resolution_deg(BAND_PX)
    # A SUN azimuth, not a shadow bearing: the shadow runs the other way.
    assert estimate.shadow_azimuth_deg == pytest.approx(shadow_azimuth_deg(estimate.sun_azimuth_deg))
    assert angular_difference_deg(estimate.shadow_azimuth_deg, true_azimuth) >= 180.0 - error
