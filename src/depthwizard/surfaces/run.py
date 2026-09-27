"""Phase 4b: georeferenced DSM export, validated on the SYNTHETIC fixture only.

::

    python -m depthwizard.surfaces.run synthetic --config configs/phase4.yaml

Pipeline (all SYNTHETIC)::

    Phase 0 fixture (EPSG:32643 image grid)          synthetic DEM (EPSG:4326, 1")
                 |                                                |
    z_rel = log(AGL_true + 1) + c   (stand-in for Phase 3)   reproject -> image grid (bilinear)
                 |                                                |
    AGL = a * exp(z_rel) + b   (a, b from the frozen Phase 4a     T(x)
                 |               SYNTHETIC calibration)           |
                 +-------------------- DSM = T + AGL -------------+
                                            |
                                      COG + validation

Why a constructed ``z_rel`` rather than a Phase 3 prediction: the fixture is a
flat grey rendering the Phase 3 network was never trained on, so its output
there would say nothing. The constructed field is the one the Phase 4a
SYNTHETIC control already uses, with a known answer (``a = exp(-c)``,
``b = -1``).

Two DSMs are produced and both scored against ``T_true + AGL_true``:

* ``synthetic_dsm.tif`` -- ``a, b`` recovered by the frozen Phase 4a synthetic
  calibration (shadows + WLS). Its error is the Phase 4a calibration error
  carried into the DSM.
* ``synthetic_dsm_known_calibration.tif`` -- the fixture's known ``a, b``. AGL
  is then exact, so any error is georeferencing / resampling: this is the
  geometric acceptance test.

Nothing here downloads a DEM, and nothing here applies to real DFC2019 tiles,
which carry no CRS or geotransform. The real-data georeferenced path is NOT
YET VERIFIED.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from depthwizard.calibration.config import load_phase4_config
from depthwizard.calibration.diagnostics import to_jsonable, write_json
from depthwizard.calibration.terrain import GridSpec, load_terrain_on_grid
from depthwizard.ingest.geotiff import validate_cog, write_cog
from depthwizard.ingest.synthetic import (
    TerrainSpec,
    generate_fixture,
    render_scene,
    terrain_truth_on_grid,
    write_synthetic_dem,
)
from depthwizard.logging_setup import get_logger
from depthwizard.surfaces.dsm import DsmError, fuse_dsm

__all__ = ["DSM_NODATA", "GEOMETRIC_TOLERANCE_M", "run_synthetic_dsm", "main"]

log = get_logger(__name__)

#: Written for invalid DSM/terrain/AGL pixels. Not a possible elevation, and
#: never 0 (a legitimate elevation).
DSM_NODATA = -9999.0

#: Acceptance bound for the known-calibration DSM against truth. The terrain
#: is a plane, which bilinear resampling reproduces exactly; what remains is
#: float32 storage of the DEM (~3e-5 m at 540 m) and the UTM <-> lon/lat
#: non-linearity across one 1" DEM cell (sub-millimetre). A half-pixel
#: georeferencing error would instead show as slope * 0.25 m = 5 mm on this
#: terrain, so 1 mm separates the two.
GEOMETRIC_TOLERANCE_M = 1.0e-3


def _errors(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    diff = predicted[valid] - truth[valid]
    return {
        "valid_pixels": int(valid.sum()),
        "max_abs_error_m": float(np.abs(diff).max()),
        "mae_m": float(np.abs(diff).mean()),
        "rmse_m": float(math.sqrt(float((diff**2).mean()))),
        "bias_m": float(diff.mean()),
    }


def _half_ulp(values: np.ndarray) -> float:
    """Half the float32 spacing at the largest magnitude written: the storage tolerance."""
    finite = values[np.isfinite(values)]
    return float(np.spacing(np.float32(np.abs(finite).max())) / 2.0) if finite.size else 0.0


def _write_and_validate(path: Path, array: np.ndarray, grid: GridSpec, tags: dict[str, Any]) -> dict[str, Any]:
    data = np.where(np.isfinite(array), array, np.nan).astype(np.float32)
    write_cog(path, data, crs=grid.crs, transform=grid.transform, nodata=DSM_NODATA, tags=tags)
    return validate_cog(
        path, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
        nodata=DSM_NODATA, dtype="float32", expected=data.astype(np.float64), atol=0.0,
    )


def run_synthetic_dsm(
    config: str | Path = "configs/phase4.yaml",
    *,
    output_dir: str | Path = "data/outputs/phase4b",
    terrain: TerrainSpec = TerrainSpec(),
) -> dict[str, Any]:
    """Run the SYNTHETIC Phase 4b acceptance pipeline and write its report."""
    from depthwizard.calibration.run import _synthetic_scene_run
    from depthwizard.config import load_config

    cfg = load_phase4_config(config)
    scene = load_config(cfg.synthetic.scene_config).scene
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The georeferenced image grid is the Phase 0 fixture's GeoTIFF.
    fixture = generate_fixture(scene, out / "fixture", run_id="phase4b")
    grid = GridSpec.from_raster(fixture.image_path)

    # 2. Synthetic DEM in a different CRS, then onto the image grid.
    dem = write_synthetic_dem(scene, out / "synthetic_dem_source.tif", terrain)
    placed = load_terrain_on_grid(dem.path, grid)
    terrain_truth = terrain_truth_on_grid(scene, terrain)

    # 3. The relative field and the calibration.
    render = render_scene(scene)
    agl_truth = render.height_m.astype(np.float64)
    c = cfg.synthetic.log_offset
    z_rel = (np.log(agl_truth + 1.0) + c).astype(np.float32)  # float32, like a Phase 3 output
    control = _synthetic_scene_run(scene, cfg, use_recovered_azimuth=False)["uncertainty_forms"]["explicit"]
    if not control["fit"]["ok"]:
        raise DsmError(f"the frozen Phase 4a synthetic calibration returned no a,b: {control['fit']['failure_reason']}")
    calibrations = {
        "phase4a_synthetic_calibration": (control["recovered_a"], control["recovered_b"]),
        "known_calibration": (math.exp(-c), -1.0),
    }

    dsm_truth = terrain_truth + agl_truth
    tags = {"SYNTHETIC": "true", "PRODUCER": "depthwizard.surfaces.run", "UNITS": "metres"}
    products: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for name, (a, b) in calibrations.items():
        fused = fuse_dsm(placed.values, placed.valid, z_rel, a, b)
        filename = "synthetic_dsm.tif" if name == "phase4a_synthetic_calibration" else "synthetic_dsm_known_calibration.tif"
        cog = _write_and_validate(out / filename, fused.dsm, grid,
                                  {**tags, "CONTENT": "DSM = T + a*exp(z_rel) + b", "CALIBRATION": name,
                                   "CALIB_A": a, "CALIB_B": b})
        # Score the DSM exactly as stored (float32 on disk).
        stored = np.where(fused.valid, fused.dsm.astype(np.float32).astype(np.float64), np.nan)
        results[name] = {
            "a": a,
            "b": b,
            **fused.summary(),
            "dsm_vs_truth": _errors(stored, dsm_truth, fused.valid),
            "agl_vs_truth": _errors(fused.agl, agl_truth, fused.valid),
            "cog": cog,
        }
        products[filename] = str(out / filename)

    known = results["known_calibration"]["dsm_vs_truth"]
    acceptance_ok = known["max_abs_error_m"] <= GEOMETRIC_TOLERANCE_M
    if not acceptance_ok:
        log.error("SYNTHETIC DSM geometric acceptance FAILED", extra=known)

    products["synthetic_dem_reprojected.tif"] = str(out / "synthetic_dem_reprojected.tif")
    dem_cog = _write_and_validate(out / "synthetic_dem_reprojected.tif", placed.values, grid,
                                  {**tags, "CONTENT": "terrain reprojected onto the image grid"})
    agl_known = fuse_dsm(placed.values, placed.valid, z_rel, *calibrations["phase4a_synthetic_calibration"])
    products["synthetic_agl.tif"] = str(out / "synthetic_agl.tif")
    agl_cog = _write_and_validate(out / "synthetic_agl.tif", agl_known.agl, grid,
                                  {**tags, "CONTENT": "AGL = a*exp(z_rel) + b (Phase 4a synthetic calibration)"})

    report = {
        "phase": "4b",
        "label": "SYNTHETIC georeferenced fixture only. The real-data (DFC2019) georeferenced path is NOT YET VERIFIED.",
        "formula": "DSM(x) = T(x) + a*exp(z_rel(x)) + b; the DEM is additive terrain, never a scale anchor",
        "external_dem_downloaded": False,
        "image_grid": {**grid.to_dict(), "source": str(fixture.image_path)},
        "dem": {
            "source_file": str(dem.path),
            "terrain_spec": terrain.to_dict(),
            "note": "SYNTHETIC tilted plane defined in the image CRS, sampled on a 1 arc-second EPSG:4326 grid",
        },
        "reprojection": placed.provenance,
        "terrain_vs_truth": _errors(placed.values, terrain_truth, placed.valid),
        "relative_field": {
            "z_rel": "log(AGL_true + 1) + c, float32 (SYNTHETIC stand-in for a Phase 3 output)",
            "c": c,
        },
        "calibrations": results,
        "geometric_acceptance": {
            "product": "synthetic_dsm_known_calibration.tif",
            "tolerance_m": GEOMETRIC_TOLERANCE_M,
            "max_abs_error_m": known["max_abs_error_m"],
            "passed": acceptance_ok,
        },
        "cog_validation": {
            "synthetic_dsm.tif": results["phase4a_synthetic_calibration"]["cog"]["ok"],
            "synthetic_dsm_known_calibration.tif": results["known_calibration"]["cog"]["ok"],
            "synthetic_dem_reprojected.tif": dem_cog["ok"],
            "synthetic_agl.tif": agl_cog["ok"],
        },
        "products": products,
        "qgis": "not performed (deferred); validation is programmatic via rasterio",
    }
    write_json(out / "phase4b_report.json", report)
    if not acceptance_ok:
        raise DsmError(f"SYNTHETIC DSM geometric acceptance failed: max |error| {known['max_abs_error_m']} m "
                       f"> {GEOMETRIC_TOLERANCE_M} m")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    from depthwizard.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="DepthWizard Phase 4b: georeferenced DSM export (SYNTHETIC)")
    parser.add_argument("target", choices=("synthetic",))
    parser.add_argument("--config", default="configs/phase4.yaml")
    parser.add_argument("--output-dir", default="data/outputs/phase4b")
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging({"level": "INFO", "format": "text"})
    report = run_synthetic_dsm(args.config, output_dir=args.output_dir)
    print(json.dumps(to_jsonable({
        "image_grid": report["image_grid"],
        "reprojection": report["reprojection"],
        "terrain_vs_truth": report["terrain_vs_truth"],
        "dsm_vs_truth": {k: v["dsm_vs_truth"] for k, v in report["calibrations"].items()},
        "geometric_acceptance": report["geometric_acceptance"],
        "cog_validation": report["cog_validation"],
    }), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
