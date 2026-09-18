"""Physical models used by DepthWizard.

* :mod:`depthwizard.physics.sun` - sun and shadow geometry: ``L = h / tan(theta)``,
  the anti-sun shadow bearing, and the map/pixel conversions that go with them.
* :mod:`depthwizard.physics.height` - the inverse estimator ``h = L * tan(theta)``,
  its uncertainty ``dh = tan(theta) * dL``, and the 25-45 degree solar-elevation
  confidence band.

No terrain-slope, off-nadir or per-scene bias correction lives here; those
belong to the calibration stage, which is not implemented.
"""

from depthwizard.physics.height import (
    USABLE_SUN_ELEVATION_MAX_DEG,
    USABLE_SUN_ELEVATION_MIN_DEG,
    Confidence,
    HeightEstimate,
    SunBand,
    SunBandStatus,
    height_from_shadow,
    height_uncertainty_m,
    pixels_to_metres,
    sun_band_status,
)
from depthwizard.physics.sun import (
    height_from_shadow_length_m,
    shadow_azimuth_deg,
    shadow_direction_pixels,
    shadow_direction_unit,
    shadow_length_m,
    shadow_pixel_offset,
)

__all__ = [
    # sun geometry
    "shadow_length_m",
    "height_from_shadow_length_m",
    "shadow_azimuth_deg",
    "shadow_direction_unit",
    "shadow_direction_pixels",
    "shadow_pixel_offset",
    # height estimation
    "USABLE_SUN_ELEVATION_MIN_DEG",
    "USABLE_SUN_ELEVATION_MAX_DEG",
    "SunBand",
    "SunBandStatus",
    "Confidence",
    "HeightEstimate",
    "sun_band_status",
    "pixels_to_metres",
    "height_uncertainty_m",
    "height_from_shadow",
]
