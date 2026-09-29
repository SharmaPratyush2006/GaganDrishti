"""Shared Phase 8 helpers: units, status labels, metadata summaries and paths.

The rules the demo enforces, in one place:

* A height is labelled ``metres`` only when it came out of the metric path
  (ABSOLUTE input *and* a metric product). A RELATIVE field is ``unitless``
  wherever it appears, including inside an ABSOLUTE scene.
* Anything the project has not measured is the string :data:`NOT_MEASURED`
  (``"not yet measured"``), never a number. Anything the input cannot support
  is a :func:`not_available` record with its reason.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from depthwizard.ingest.metadata import MetadataError, SceneMetadata
from depthwizard.mode import Mode
from depthwizard.validation.evaluation import NOT_MEASURED

__all__ = [
    "NOT_AVAILABLE",
    "NOT_MEASURED",
    "METRES",
    "UNITLESS",
    "REPO_ROOT",
    "OUTPUTS_ROOT",
    "metadata_summary",
    "mode_units",
    "not_available",
    "report_path",
    "url_for",
]

#: Status of a product the input (or the project) cannot provide.
NOT_AVAILABLE = "not available"
METRES = "metres"
UNITLESS = "unitless"

#: ``<repo>``: src/depthwizard/demo/common.py -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
#: The one directory the viewer's data route serves (``/data/outputs/*``).
OUTPUTS_ROOT = Path("data") / "outputs"


def not_available(reason: str) -> dict[str, Any]:
    """A product that was not produced, and why. Never a placeholder value."""
    return {"status": NOT_AVAILABLE, "reason": reason}


def report_path(path: str | Path) -> str:
    """``path`` as the reports record it: relative to the working directory when inside it.

    The viewer only follows repo-relative paths under ``data/outputs/``, so the
    demo is run from the repository root (the CLI and the API ensure that).
    """
    p = Path(path)
    try:
        rel = Path(os.path.relpath(p.resolve(), Path.cwd().resolve()))
    except ValueError:  # another drive on Windows
        return str(p)
    return str(p) if rel.parts[:1] == ("..",) else str(rel)


def url_for(path: str | Path) -> str | None:
    """URL of ``path`` on the viewer data route / the API's ``/outputs``, or None if not served."""
    rel = Path(report_path(path))
    if rel.is_absolute() or rel.parts[: len(OUTPUTS_ROOT.parts)] != OUTPUTS_ROOT.parts:
        return None
    return "/" + rel.as_posix()


def _provenance(p: Any) -> str | None:
    return p.describe() if p is not None else None


def metadata_summary(metadata: SceneMetadata) -> dict[str, Any]:
    """The Phase 1 metadata a judge needs, with provenance. Missing fields are None."""
    geo = metadata.georeference
    gsd: float | None = None
    gsd_problem: str | None = None
    if geo is not None:
        try:
            gsd = geo.gsd_m
        except MetadataError as exc:  # non-square pixels: report, do not pick one
            gsd_problem = str(exc)
    sun = metadata.sun
    return {
        "file": metadata.path.name,
        "driver": metadata.driver,
        "width_px": metadata.width,
        "height_px": metadata.height,
        "band_count": metadata.band_count,
        "dtype": metadata.dtype,
        "crs": geo.crs.to_string() if geo is not None else None,
        "geotransform": [float(v) for v in tuple(geo.transform)[:6]] if geo is not None else None,
        "gsd_m": gsd,
        "gsd_note": gsd_problem or (geo.gsd_note if geo is not None else None),
        "sun_elevation_deg": sun.elevation_deg if sun is not None else None,
        "sun_azimuth_deg": sun.azimuth_deg if sun is not None else None,
        "sun_elevation_source": _provenance(sun.elevation_provenance) if sun is not None else None,
        "sun_azimuth_source": _provenance(sun.azimuth_provenance) if sun is not None else None,
        "rpc_present": metadata.rpc is not None,
        "probed_sources": list(metadata.probed_sources),
        "downgrade_reason": metadata.downgrade_reason,
    }


def mode_units(mode: Mode, *, metric_product: bool) -> str:
    """Units of the height values a response carries: metres only on the metric path."""
    return METRES if (mode is Mode.ABSOLUTE and metric_product) else UNITLESS
