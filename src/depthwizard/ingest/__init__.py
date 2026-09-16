"""Data ingest: reading real imagery and generating synthetic fixtures.

Phase 0 implements:

* :mod:`depthwizard.ingest.geotiff`   - thin, typed GeoTIFF read/write helpers
* :mod:`depthwizard.ingest.synthetic` - the synthetic scene generator

Ingest of real satellite products is NOT implemented in Phase 0.
"""

from depthwizard.ingest.geotiff import (
    MetadataTags,
    RasterReadResult,
    read_raster,
    read_sun_metadata,
    write_single_band,
)
from depthwizard.ingest.synthetic import GeneratedFixture, generate_fixture, render_scene

__all__ = [
    "MetadataTags",
    "RasterReadResult",
    "read_raster",
    "read_sun_metadata",
    "write_single_band",
    "GeneratedFixture",
    "generate_fixture",
    "render_scene",
]
