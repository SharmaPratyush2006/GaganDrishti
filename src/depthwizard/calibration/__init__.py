"""Calibration: the relative Phase 3 field to metres (Phase 4a, pixel space).

Phase 3 predicts above-ground height (AGL) in log space, ``z_rel ~= log(AGL + 1)
+ c`` with an unknown per-tile ``c``. Phase 4a turns it into metres per tile::

    AGL(x) = a * exp(z_rel(x)) + b

with ``a, b`` fitted to shadow-derived building heights. The sun is fixed
*first*, from a few reference buildings, so scale is never solved jointly with
``a, b`` (that system is degenerate). Terrain plays no part: it is additive in
the future georeferenced DSM (Phase 4b, not implemented), ``DSM = T + AGL``.

Modules:

* :mod:`~depthwizard.calibration.footprints` -- DFC2019 CLS (class 6) -> footprints
* :mod:`~depthwizard.calibration.azimuth` -- sun azimuth from shadow placement
* :mod:`~depthwizard.calibration.shadow_anchor` -- references, sun scale, shadow heights
* :mod:`~depthwizard.calibration.fusion` -- linearisation and the WLS ``a, b`` fit
* :mod:`~depthwizard.calibration.ransac` -- seeded robust affine fitting
* :mod:`~depthwizard.calibration.diagnostics` -- error accounting and reports
* :mod:`~depthwizard.calibration.run` -- the DFC2019 run and the SYNTHETIC control
"""

from depthwizard.calibration.fusion import (
    AffineFit,
    CalibrationError,
    apply_affine,
    fit_affine_wls,
    linearise_relative,
)
from depthwizard.calibration.ransac import RansacResult, ransac_affine

__all__ = [
    "AffineFit",
    "CalibrationError",
    "RansacResult",
    "apply_affine",
    "fit_affine_wls",
    "linearise_relative",
    "ransac_affine",
]
