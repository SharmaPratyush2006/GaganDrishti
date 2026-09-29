"""Phase 8: demo assembly around the Phase 0-7 pipeline.

Nothing here is a new algorithm. Every number comes from an existing module:

* :mod:`depthwizard.demo.synthetic` -- the deterministic end-to-end demo on the
  Phase 0 SYNTHETIC fixture: ingest (Phase 1) -> shadow heights (Phase 2) ->
  georeferenced DSM (Phase 4b, which runs the Phase 4a synthetic calibration)
  -> validation (Phase 5) -> DTM / nDSM (Phase 6) -> glTF / Draco mesh (the
  Phase 7 viewer's own exporter, run under Node).
* :mod:`depthwizard.demo.process` -- one uploaded image: ingest, mode, metadata,
  and the Phase 3 RELATIVE (unitless) field where the input allows it. Metric
  products are reported as not available, with the reason; none are invented.
* :mod:`depthwizard.demo.api` -- a minimal FastAPI wrapper over both.

Run ``python -m depthwizard.demo --help``.
"""

from depthwizard.demo.common import NOT_AVAILABLE, NOT_MEASURED, REPO_ROOT

__all__ = ["NOT_AVAILABLE", "NOT_MEASURED", "REPO_ROOT"]
