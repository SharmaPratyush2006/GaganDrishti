"""Mode router.

One entry point that looks at an input, picks the right loader, and decides
whether the scene can be processed in :attr:`Mode.ABSOLUTE` (heights in metres)
or only :attr:`Mode.RELATIVE` (heights relative to each other).

The rules:

* ``.png`` / ``.jpg`` and friends  -> always RELATIVE. A plain image has no
  scale and no sun geometry, and this is expected, not an error.
* ``.tif`` / ``.tiff`` with a CRS, a real geotransform and both sun angles
  -> ABSOLUTE.
* ``.tif`` / ``.tiff`` missing any of those -> RELATIVE, but **never quietly**.
  The downgrade is logged at WARNING, and the reason is recorded on the
  returned :class:`RoutingDecision` and on the scene metadata.
* Anything else -> :class:`UnsupportedInputError`.

Pass ``require=Mode.ABSOLUTE`` to turn a downgrade into a loud
:class:`MissingMetadataError` that lists every source probed and every metadata
key tried. Use that wherever metric output is non-negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from depthwizard.ingest.loaders import (
    GEOTIFF_SUFFIXES,
    IMAGE_SUFFIXES,
    GeoTIFFLoader,
    ImageLoader,
    LoadedScene,
)
from depthwizard.ingest.metadata import (
    SceneMetadata,
    UnsupportedInputError,
)
from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = ["RoutingDecision", "route", "load_scene", "read_metadata", "SUPPORTED_SUFFIXES"]

log = get_logger(__name__)

SUPPORTED_SUFFIXES: frozenset[str] = GEOTIFF_SUFFIXES | IMAGE_SUFFIXES


@dataclass(frozen=True)
class RoutingDecision:
    """Which loader handled an input, which mode it landed in, and why."""

    path: Path
    mode: Mode
    loader: str
    reason: str
    metadata: SceneMetadata

    @property
    def was_downgraded(self) -> bool:
        """True when a georeferenced input fell back to RELATIVE."""
        return self.mode is Mode.RELATIVE and self.loader == "GeoTIFFLoader"


def _select_loader(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in GEOTIFF_SUFFIXES:
        return "GeoTIFFLoader"
    if suffix in IMAGE_SUFFIXES:
        return "ImageLoader"
    raise UnsupportedInputError(
        f"unsupported input {path.name!r} (suffix {suffix!r}); "
        f"supported suffixes are {sorted(SUPPORTED_SUFFIXES)}"
    )


def read_metadata(path: str | Path, *, require: Mode | None = None) -> SceneMetadata:
    """Probe an input's metadata and resolve its mode, without reading pixels."""
    return route(path, require=require).metadata


def route(path: str | Path, *, require: Mode | None = None) -> RoutingDecision:
    """Decide how to process ``path``.

    Args:
        path: The input file.
        require: If :attr:`Mode.ABSOLUTE`, refuse to downgrade -- raise
            :class:`~depthwizard.ingest.metadata.MissingMetadataError` listing
            what was probed and what was missing. If None (the default), route
            to the best available mode and record any downgrade.

    Raises:
        UnsupportedInputError: for a file type neither loader handles.
    """
    path = Path(path)
    loader_name = _select_loader(path)

    if loader_name == "GeoTIFFLoader":
        # require_absolute is driven by the caller, so the loud failure and the
        # explicit-downgrade paths share exactly one implementation.
        loader = GeoTIFFLoader(require_absolute=require is Mode.ABSOLUTE)
        metadata = loader.read_metadata(path)
        reason = metadata.downgrade_reason or (
            "georeferenced raster with CRS, geotransform and both sun angles"
        )
    else:
        metadata = ImageLoader().read_metadata(path)
        reason = metadata.downgrade_reason or "plain image input"
        if require is Mode.ABSOLUTE:
            metadata.require_absolute()  # raises with the full search record

    decision = RoutingDecision(
        path=path,
        mode=metadata.mode,
        loader=loader_name,
        reason=reason,
        metadata=metadata,
    )

    if decision.was_downgraded:
        log.warning(
            "georeferenced input downgraded to relative mode",
            extra={"path": str(path), "reason": reason},
        )
    else:
        log.info(
            "routed input",
            extra={"path": str(path), "mode": decision.mode.value, "loader": loader_name},
        )
    return decision


def load_scene(path: str | Path, *, require: Mode | None = None) -> LoadedScene:
    """Route ``path`` to the right loader and read its pixels and metadata."""
    decision = route(path, require=require)
    if decision.loader == "GeoTIFFLoader":
        return GeoTIFFLoader(require_absolute=False).load(path)
    return ImageLoader().load(path)
