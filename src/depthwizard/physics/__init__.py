"""Physical models used by DepthWizard.

Phase 0 implements only :mod:`depthwizard.physics.sun`, the sun/shadow
geometry that the whole project rests on.
"""

from depthwizard.physics.sun import (
    height_from_shadow_length_m,
    shadow_direction_unit,
    shadow_length_m,
    shadow_pixel_offset,
)

__all__ = [
    "shadow_length_m",
    "height_from_shadow_length_m",
    "shadow_direction_unit",
    "shadow_pixel_offset",
]
