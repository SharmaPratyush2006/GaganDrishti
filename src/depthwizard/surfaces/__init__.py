"""Surface products (Phase 4b): the georeferenced DSM.

``DSM = T + AGL = T + a*exp(z_rel) + b`` -- terrain from a DEM reprojected onto
the image grid (:mod:`depthwizard.calibration.terrain`) plus the calibrated
above-ground height. There is no DTM / nDSM step: Phase 3 already predicts AGL,
which *is* the nDSM (Phase 6 is cancelled).

Validated on the SYNTHETIC georeferenced fixture only; the real-data
georeferenced path is NOT YET VERIFIED.
"""

from depthwizard.surfaces.dsm import DsmError, DsmResult, calibrated_agl, fuse_dsm

__all__ = ["DsmError", "DsmResult", "calibrated_agl", "fuse_dsm"]
