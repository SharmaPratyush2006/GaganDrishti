"""DepthWizard - building height estimation from monocular satellite imagery.

SIH 2026, problem statement SIH26175.

Phase 0 (this release) provides the project scaffolding only:

* :mod:`depthwizard.config`        - YAML-backed dataclass configuration
* :mod:`depthwizard.logging_setup` - structured logging
* :mod:`depthwizard.physics.sun`   - sun/shadow geometry
* :mod:`depthwizard.ingest`        - GeoTIFF I/O and the synthetic fixture generator

The calibration, surfaces and validation subpackages are deliberate
placeholders; see the README checklist for what is and is not implemented.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
