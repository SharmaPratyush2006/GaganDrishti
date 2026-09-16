"""DepthWizard - building height estimation from monocular satellite imagery.

SIH 2026, problem statement SIH26175.

Implemented so far:

* :mod:`depthwizard.config`        - YAML-backed dataclass configuration
* :mod:`depthwizard.logging_setup` - structured logging
* :mod:`depthwizard.mode`          - absolute (metric) vs relative processing
* :mod:`depthwizard.physics.sun`   - sun/shadow geometry
* :mod:`depthwizard.ingest`        - loaders, metadata discovery, mode routing,
  radiometric normalisation, tiling, and the synthetic fixture generator

The calibration, surfaces and validation subpackages are deliberate
placeholders; see the README checklist for what is and is not implemented.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
