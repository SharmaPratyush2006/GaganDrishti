"""DSM assembly (Phase 4b): terrain + calibrated above-ground height.

::

    AGL(x) = a * exp(z_rel(x)) + b          (the frozen Phase 4a model)
    DSM(x) = T(x) + AGL(x)

``z_rel`` is the Phase 3 log-space relative field; ``a, b`` are calibration
parameters supplied by the caller (never fitted here); ``T`` is terrain already
on the same grid (:func:`depthwizard.calibration.terrain.load_terrain_on_grid`).

Terrain is additive only. Nothing here fits terrain against the relative field,
and no DTM is subtracted: Phase 3 already predicts AGL (= nDSM).

Invalid pixels propagate: where the terrain or the AGL is invalid, the DSM is
invalid (NaN in memory, the declared nodata on disk). Nothing is filled with 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DsmError", "DsmResult", "calibrated_agl", "fuse_dsm"]


class DsmError(RuntimeError):
    """The DSM cannot be assembled; the message says why."""


@dataclass(frozen=True)
class DsmResult:
    #: ``(H, W)`` float64 DSM in metres; NaN where invalid.
    dsm: np.ndarray
    #: ``(H, W)`` float64 calibrated AGL in metres; NaN where invalid.
    agl: np.ndarray
    valid: np.ndarray
    a: float
    b: float

    def summary(self) -> dict[str, Any]:
        return {
            "formula": "DSM = T + a*exp(z_rel) + b",
            "a": self.a,
            "b": self.b,
            "valid_pixels": int(self.valid.sum()),
            "nodata_pixels": int((~self.valid).sum()),
        }


def _check_params(a: float, b: float, require_positive_scale: bool) -> tuple[float, float]:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError) as exc:
        raise DsmError(f"calibration parameters must be real numbers, got a={a!r}, b={b!r}") from exc
    if not (math.isfinite(a) and math.isfinite(b)):
        raise DsmError(f"calibration parameters must be finite, got a={a}, b={b}")
    if require_positive_scale and a <= 0.0:
        raise DsmError(
            f"calibration scale a={a} is not positive: AGL would fall as the relative height "
            "rises. Refusing to export it as a DSM (pass require_positive_scale=False to override)"
        )
    return a, b


def calibrated_agl(
    z_rel: np.ndarray,
    a: float,
    b: float,
    *,
    valid: np.ndarray | None = None,
    require_positive_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``AGL = a * exp(z_rel) + b``; returns ``(agl, agl_valid)``.

    Raises:
        DsmError: on invalid parameters or when no pixel has a valid AGL.
    """
    a, b = _check_params(a, b, require_positive_scale)
    z = np.asarray(z_rel, dtype=np.float64)
    if z.ndim != 2:
        raise DsmError(f"z_rel must be 2-D, got shape {z.shape}")
    ok = np.isfinite(z)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != z.shape:
            raise DsmError(f"AGL validity mask {mask.shape} does not match z_rel {z.shape}")
        ok &= mask
    with np.errstate(over="ignore", invalid="ignore"):
        agl = a * np.exp(np.where(ok, z, 0.0)) + b
    ok &= np.isfinite(agl)
    agl = np.where(ok, agl, np.nan)
    if not ok.any():
        raise DsmError("no valid AGL pixel: z_rel is entirely invalid or overflows exp()")
    return agl, ok


def fuse_dsm(
    terrain: np.ndarray,
    terrain_valid: np.ndarray,
    z_rel: np.ndarray,
    a: float,
    b: float,
    *,
    agl_valid: np.ndarray | None = None,
    require_positive_scale: bool = True,
) -> DsmResult:
    """``DSM = T + a * exp(z_rel) + b``, pixel by pixel, on one grid.

    Raises:
        DsmError: on a shape mismatch, invalid parameters, or an entirely
            invalid terrain or AGL grid.
    """
    t = np.asarray(terrain, dtype=np.float64)
    t_ok = np.asarray(terrain_valid, dtype=bool) & np.isfinite(t)
    if t.shape != np.shape(z_rel):
        raise DsmError(f"terrain {t.shape} and z_rel {np.shape(z_rel)} are not on the same grid")
    if not t_ok.any():
        raise DsmError("terrain grid is entirely invalid")
    agl, a_ok = calibrated_agl(z_rel, a, b, valid=agl_valid, require_positive_scale=require_positive_scale)
    valid = t_ok & a_ok
    if not valid.any():
        raise DsmError("terrain and AGL have no valid pixel in common")
    dsm = np.where(valid, t + agl, np.nan)
    return DsmResult(dsm=dsm, agl=agl, valid=valid, a=float(a), b=float(b))
