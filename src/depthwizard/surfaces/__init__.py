"""Surface products.

Phase 4b, the georeferenced DSM: ``DSM = T + AGL = T + a*exp(z_rel) + b`` --
terrain from a DEM reprojected onto the image grid
(:mod:`depthwizard.calibration.terrain`) plus the model-derived, calibrated
above-ground height. The DSM assembly itself subtracts no DTM.

Phase 6, an independent DSM-only product: ground extraction
(:mod:`depthwizard.surfaces.ground`), the DTM and ``nDSM = DSM - DTM``
(:mod:`depthwizard.surfaces.dtm`), and the export and acceptance run
(:mod:`depthwizard.surfaces.phase6`). Import those modules directly.

Both are validated on the SYNTHETIC georeferenced fixture only; the real-data
georeferenced path is NOT YET VERIFIED, and real-world DTM / nDSM accuracy is
not yet measured.
"""

from depthwizard.surfaces.dsm import DsmError, DsmResult, calibrated_agl, fuse_dsm

__all__ = ["DsmError", "DsmResult", "calibrated_agl", "fuse_dsm"]
