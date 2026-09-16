"""Data ingest: reading imagery, resolving its metadata, and preparing tiles.

Phase 0 provided:

* :mod:`depthwizard.ingest.geotiff`   - thin, typed GeoTIFF read/write helpers
* :mod:`depthwizard.ingest.synthetic` - the synthetic scene generator

Phase 1 adds:

* :mod:`depthwizard.ingest.metadata`   - provider-agnostic metadata discovery
* :mod:`depthwizard.ingest.loaders`    - GeoTIFFLoader and ImageLoader
* :mod:`depthwizard.ingest.router`     - picks the loader and the processing mode
* :mod:`depthwizard.ingest.radiometry` - percentile stretch to 8-bit
* :mod:`depthwizard.ingest.tiling`     - 512x512 tiles with 64 px overlap

Typical use::

    from depthwizard.ingest import load_scene, normalize_to_uint8, Tiler

    scene = load_scene("scene.tif")          # routes to ABSOLUTE or RELATIVE
    eight_bit = normalize_to_uint8(scene.array).array
    grid = Tiler().build_grid_for_scene(scene)
"""

from depthwizard.ingest.geotiff import (
    MetadataTags,
    RasterReadResult,
    pixel_to_map,
    read_raster,
    read_sun_metadata,
    translate_transform,
    write_single_band,
)
from depthwizard.ingest.loaders import GeoTIFFLoader, ImageLoader, LoadedScene
from depthwizard.ingest.metadata import (
    GeoReference,
    MetadataError,
    MissingMetadataError,
    Provenance,
    RpcInfo,
    SceneMetadata,
    SunMetadata,
    UnsupportedInputError,
)
from depthwizard.ingest.radiometry import (
    BandStretch,
    StretchResult,
    apply_stretch,
    compute_band_stretches,
    normalize_to_uint8,
)
from depthwizard.ingest.router import RoutingDecision, load_scene, read_metadata, route
from depthwizard.ingest.synthetic import GeneratedFixture, generate_fixture, render_scene
from depthwizard.ingest.tiling import Tile, TileGrid, Tiler

__all__ = [
    # geotiff primitives
    "MetadataTags",
    "RasterReadResult",
    "pixel_to_map",
    "translate_transform",
    "read_raster",
    "read_sun_metadata",
    "write_single_band",
    # synthetic fixtures
    "GeneratedFixture",
    "generate_fixture",
    "render_scene",
    # metadata
    "GeoReference",
    "MetadataError",
    "MissingMetadataError",
    "Provenance",
    "RpcInfo",
    "SceneMetadata",
    "SunMetadata",
    "UnsupportedInputError",
    # loaders + routing
    "GeoTIFFLoader",
    "ImageLoader",
    "LoadedScene",
    "RoutingDecision",
    "route",
    "read_metadata",
    "load_scene",
    # radiometry
    "BandStretch",
    "StretchResult",
    "compute_band_stretches",
    "apply_stretch",
    "normalize_to_uint8",
    # tiling
    "Tile",
    "TileGrid",
    "Tiler",
]
