"""Processing modes.

DepthWizard can run in two fundamentally different regimes, and almost every
later stage branches on which one it is in. The distinction is about *units*:

:attr:`Mode.ABSOLUTE`
    The input is georeferenced and carries illumination geometry, so a shadow
    measured in pixels converts to a shadow measured in **metres**, and
    ``h = L * tan(theta)`` yields a height in metres. This needs a CRS, a
    geotransform, a ground sample distance and both sun angles.

:attr:`Mode.RELATIVE`
    The input is a plain image: no CRS, no geotransform, no sun angles. Nothing
    can be expressed in metres, because there is no scale and no illumination
    geometry. Only *relative* height -- "this building is taller than that one"
    -- is recoverable.

This enum lives at the top level rather than inside ``ingest`` because it is
cross-cutting: ingest decides the mode, and later phases consume it.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Mode"]


class Mode(str, Enum):
    """Whether heights can be expressed in metres or only relative to each other."""

    #: Georeferenced input with full sun geometry; heights in metres.
    ABSOLUTE = "absolute"

    #: Plain image or incomplete metadata; heights are unitless and relative.
    RELATIVE = "relative"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value

    @property
    def is_metric(self) -> bool:
        """True when outputs can carry real-world units."""
        return self is Mode.ABSOLUTE
