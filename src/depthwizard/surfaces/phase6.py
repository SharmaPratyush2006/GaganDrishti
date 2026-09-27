"""Phase 6: DTM and nDSM from a DSM, exported and checked on the SYNTHETIC fixture.

::

    python -m depthwizard.surfaces.phase6 synthetic --config configs/phase6.yaml

Pipeline (DSM only -- no Phase 3/4a AGL, no DEM, no labels are read)::

    DSM --(ground.MorphologicalGroundExtractor)--> ground mask
        --(dtm.interpolate_dtm: keep ground, TIN refinement, TIN under objects,
           nearest ground outside the hull)--> DTM
        --(dtm.compute_ndsm)--> nDSM = DSM - DTM
        --> dsm.tif / dtm.tif / ndsm.tif / ground_mask.tif  (COGs, validated on read-back)

Phase 6 is an **independent DSM-only** product. The pipeline's own above-ground
product remains the Phase 3/4a calibrated AGL; Phase 6 recovers terrain and
object height from surface shape alone.

The acceptance ("click a known building, get its height above ground, not its
elevation") is :func:`height_at`: a map coordinate in, the raster value there
out. There is no UI in Phase 6.

SYNTHETIC acceptance and its tolerance
--------------------------------------
Input: the Phase 4b DSMs of the Phase 0 fixture (first in ``synthetic.dsm_products``
is the acceptance input). Truth: the analytic terrain plane ``T`` and the
specified building heights ``H``, so ``DSM_true = T + H``.

The tolerance is **derived, not chosen**. Let ``eps = max |DSM - DSM_true|``,
measured on the input at run time. Ground pixels keep their DSM value, so their
DTM error is at most ``eps``. A non-ground pixel inside the TIN gets a convex
combination of three ground values; because ``T`` is a plane, the same
combination of ``T`` is exact, so its DTM error is also at most ``eps``. Then
``|nDSM - H| <= |DSM - DSM_true| + |DTM - T| <= 2 * eps``. A float64 rounding
allowance of ``64 * machine_eps * max|DSM|`` is added (~1e-11 m). Pixels
filled by nearest-ground extrapolation additionally allow
``slope * distance`` (none occur on the fixture, whose buildings do not touch
the border; the run records the count either way).

Real DFC2019 DTM/nDSM validation is **not yet measured**: DFC2019 Track 1
provides no DSM, no DTM, no CRS and no DEM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from depthwizard.config import ConfigError
from depthwizard.logging_setup import get_logger
from depthwizard.surfaces.dtm import DtmError, compute_ndsm, interpolate_dtm
from depthwizard.surfaces.ground import GroundExtractionError, MorphologicalGroundExtractor
from depthwizard.surfaces.phase6_config import GroundFilterConfig, Phase6Config, load_phase6_config
from depthwizard.validation.evaluation import NOT_MEASURED

__all__ = [
    "Phase6Error",
    "SYNTHETIC_LABEL",
    "gsd_from_transform",
    "derive_surfaces",
    "export_surfaces",
    "height_at",
    "run_phase6_synthetic",
    "render_markdown",
    "main",
]

log = get_logger(__name__)

SYNTHETIC_LABEL = ("SYNTHETIC: terrain and building heights were SPECIFIED when the fixture was generated, "
                   "not measured. These results test the Phase 6 geometry, not real-world DTM accuracy.")
REAL_WORLD_STATUS = (f"{NOT_MEASURED}: DFC2019 Track 1 provides no DSM, no DTM, no CRS and no DEM, and no "
                     "manually measured building has been supplied")
GROUND_MASK_NODATA = 255


class Phase6Error(RuntimeError):
    """A Phase 6 input is missing or inconsistent."""


def gsd_from_transform(transform: Any) -> float:
    """Pixel size in metres from a north-up, square-pixel affine transform."""
    if transform.b != 0 or transform.d != 0:
        raise Phase6Error("rotated transforms are not supported: the opening window assumes north-up pixels")
    x, y = abs(transform.a), abs(transform.e)
    if not math.isclose(x, y, rel_tol=1e-9) or x <= 0:
        raise Phase6Error(f"pixels must be square and non-degenerate, got {x} x {y}")
    return float(x)


def derive_surfaces(
    dsm: np.ndarray,
    *,
    valid: np.ndarray,
    gsd_m: float,
    ground_cfg: GroundFilterConfig,
    max_building_extent_m: float,
) -> dict[str, Any]:
    """DSM -> ground mask -> DTM -> nDSM. Returns the three stage results."""
    extractor = MorphologicalGroundExtractor(
        max_building_extent_m=max_building_extent_m,
        max_terrain_slope=ground_cfg.max_terrain_slope,
        min_object_height_m=ground_cfg.min_object_height_m,
    )
    ground = extractor.extract(dsm, valid=valid, gsd_m=gsd_m)
    dtm = interpolate_dtm(
        dsm, ground.ground, ground.valid,
        readmit_within_m=ground_cfg.min_object_height_m if ground_cfg.tin_refinement_max_iterations else None,
        max_iterations=ground_cfg.tin_refinement_max_iterations,
    )
    ndsm = compute_ndsm(np.where(ground.valid, dsm, np.nan), dtm.dtm)
    return {"ground": ground, "dtm": dtm, "ndsm": ndsm}


def export_surfaces(
    out_dir: Path,
    *,
    dsm: np.ndarray,
    dtm: np.ndarray,
    ndsm: np.ndarray,
    ground: np.ndarray,
    valid: np.ndarray,
    crs: Any,
    transform: Any,
    nodata: float,
    tags: Mapping[str, Any],
) -> dict[str, Any]:
    """Write DSM, DTM, nDSM (float32) and the ground mask (uint8) as COGs; validate each."""
    from depthwizard.ingest.geotiff import validate_cog, write_cog

    height, width = np.shape(dsm)
    records: dict[str, Any] = {}
    contents = {
        "dsm.tif": (dsm, "DSM: absolute surface elevation (input, re-exported)"),
        "dtm.tif": (dtm, "DTM: absolute bare-ground elevation (ground kept, TIN under objects)"),
        "ndsm.tif": (ndsm, "nDSM = DSM - DTM: object height above the extracted ground"),
    }
    for name, (array, content) in contents.items():
        data = np.where(valid & np.isfinite(array), array, np.nan).astype(np.float32)
        write_cog(out_dir / name, data, crs=crs, transform=transform, nodata=nodata,
                  tags={**tags, "CONTENT": content, "UNITS": "metres"})
        records[name] = validate_cog(out_dir / name, crs=crs, transform=transform, width=width, height=height,
                                     nodata=nodata, dtype="float32", expected=data.astype(np.float64), atol=0.0)
    mask = np.where(valid, ground.astype(np.uint8), GROUND_MASK_NODATA).astype(np.uint8)
    write_cog(out_dir / "ground_mask.tif", mask, crs=crs, transform=transform, nodata=GROUND_MASK_NODATA,
              tags={**tags, "CONTENT": "inferred ground mask: 1 ground, 0 non-ground, 255 invalid"})
    records["ground_mask.tif"] = validate_cog(
        out_dir / "ground_mask.tif", crs=crs, transform=transform, width=width, height=height,
        nodata=GROUND_MASK_NODATA, dtype="uint8", expected=np.where(valid, mask, np.nan).astype(np.float64), atol=0.0)
    return records


def height_at(path: str | Path, easting: float, northing: float) -> float | None:
    """The raster value at a map coordinate (the programmatic "click").

    Returns None outside the raster or on nodata -- never a substitute value.
    """
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(path) as src:
        row, col = rowcol(src.transform, easting, northing)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        if src.nodata is not None and value == src.nodata:
            return None
        return float(value)


# ---------------------------------------------------------------------------
# SYNTHETIC acceptance
# ---------------------------------------------------------------------------


def _require(path: Path, what: str, hint: str) -> Path:
    if not path.is_file():
        raise Phase6Error(f"{what} not found: {path}\n  {hint}")
    return path


def _max_abs(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.abs(values).max()) if values.size else None


def _score(
    derived: Mapping[str, Any], dsm: np.ndarray, valid: np.ndarray, terrain: np.ndarray, heights: np.ndarray,
    building: np.ndarray, *, gsd_m: float, slope_magnitude: float,
) -> dict[str, Any]:
    ground, dtm, ndsm = derived["ground"], derived["dtm"], derived["ndsm"]
    eps = float(np.abs(dsm[valid] - (terrain + heights)[valid]).max())
    rounding = 64 * np.finfo(np.float64).eps * float(np.abs(dsm[valid]).max())
    extrap = dtm.extrapolated
    extrap_allowance = (slope_magnitude * gsd_m * float(dtm.distance_to_ground_px[extrap].max())) if extrap.any() else 0.0
    dtm_bound = eps + rounding + extrap_allowance
    ndsm_bound = 2 * eps + rounding + extrap_allowance
    dtm_err = np.where(valid, dtm.dtm - terrain, np.nan)
    ndsm_err = np.where(ndsm.valid, ndsm.ndsm - heights, np.nan)
    terrain_px = valid & ~building
    result = {
        "eps_max_abs_dsm_error_m": eps,
        "tolerance": {
            "dtm_m": dtm_bound,
            "ndsm_m": ndsm_bound,
            "derivation": "DTM <= eps (ground kept; TIN = convex combination on a planar terrain); "
                          "nDSM <= 2*eps; + float64 rounding 64*machine_eps*max|DSM|"
                          + (" + slope*distance for extrapolated pixels" if extrap.any() else ""),
            "float64_rounding_m": rounding,
            "extrapolation_allowance_m": extrap_allowance,
        },
        "classification": {
            "building_px_known": int((building & valid).sum()),
            "building_px_kept_as_ground": int((building & ground.ground).sum()),
            "terrain_px_flagged_non_ground_by_filter": int((terrain_px & ~ground.ground).sum()),
            "terrain_px_non_ground_after_tin_refinement": int((terrain_px & ~dtm.ground).sum()),
        },
        "dtm_vs_terrain_truth": {
            "all_valid_max_abs_m": _max_abs(dtm_err),
            "terrain_px_max_abs_m": _max_abs(np.where(terrain_px, dtm_err, np.nan)),
            "under_buildings_max_abs_m": _max_abs(np.where(building & valid, dtm_err, np.nan)),
        },
        "ndsm_vs_height_truth": {
            "all_valid_max_abs_m": _max_abs(ndsm_err),
            "building_px_max_abs_m": _max_abs(np.where(building, ndsm_err, np.nan)),
            "terrain_px_max_abs_m": _max_abs(np.where(terrain_px, ndsm_err, np.nan)),
        },
        "ndsm": ndsm.summary(),
        "extrapolated_px": int(extrap.sum()),
    }
    result["passed"] = bool(
        result["dtm_vs_terrain_truth"]["all_valid_max_abs_m"] <= dtm_bound
        and result["ndsm_vs_height_truth"]["all_valid_max_abs_m"] <= ndsm_bound
    )
    return result


def run_phase6_synthetic(cfg: Phase6Config, *, command: str = "", config_path: str | Path = "") -> dict[str, Any]:
    """Run Phase 6 on the Phase 4b SYNTHETIC DSMs and score against the fixture truth."""
    from depthwizard.calibration.diagnostics import write_json
    from depthwizard.config import load_config
    from depthwizard.ingest.geotiff import read_raster
    from depthwizard.ingest.synthetic import TerrainSpec, terrain_truth_on_grid

    s = cfg.synthetic
    hint = "run `python -m depthwizard.surfaces.run synthetic --config configs/phase4.yaml` first"
    report4b = json.loads(_require(s.phase4b_report, "Phase 4b report", hint).read_text(encoding="utf-8"))
    fixture = Path(report4b["image_grid"]["source"])
    truth_path = _require(fixture.with_name(f"{fixture.stem}_truth.json"), "fixture truth sidecar", hint)
    height_path = _require(fixture.with_name(f"{fixture.stem}_truth_height.tif"), "fixture truth height", hint)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if truth.get("synthetic") is not True:
        raise Phase6Error(f"{truth_path} is not marked synthetic")
    scene = load_config(s.scene_config).scene
    if scene.name != truth.get("scene_name"):
        raise Phase6Error(f"{s.scene_config} is scene {scene.name!r}, the fixture is {truth.get('scene_name')!r}")

    height_raster = read_raster(height_path)
    heights = height_raster.array.astype(np.float64)
    spec = TerrainSpec(**report4b["dem"]["terrain_spec"])
    terrain = terrain_truth_on_grid(scene, spec)
    gsd_m = gsd_from_transform(height_raster.transform)
    building = np.zeros(heights.shape, dtype=bool)
    extents = []
    for b in truth["buildings"]:
        fp = b["footprint_px"]
        building[fp["row_min"]:fp["row_max"], fp["col_min"]:fp["col_max"]] = True
        extents.append(max(fp["row_max"] - fp["row_min"], fp["col_max"] - fp["col_min"]) * gsd_m)

    ground_cfg = cfg.ground
    if ground_cfg.max_building_extent_m is None:
        max_extent, extent_source = max(extents), "derived from the fixture's known footprints (largest axis-aligned extent)"
    else:
        max_extent, extent_source = ground_cfg.max_building_extent_m, "configs/phase6.yaml"
    slope_magnitude = math.hypot(spec.slope_east, spec.slope_north)

    out = cfg.output_dir / "synthetic"
    products: dict[str, Any] = {}
    for index, name in enumerate(s.dsm_products):
        path = _require(Path(report4b["products"][name]), f"Phase 4b DSM {name}", hint)
        raster = read_raster(path)
        if raster.array.shape != heights.shape or raster.crs != height_raster.crs or raster.transform != height_raster.transform:
            raise Phase6Error(f"{path} is not on the fixture grid")
        dsm = raster.array.astype(np.float64)
        valid = np.isfinite(dsm) & ((dsm != raster.nodata) if raster.nodata is not None else True)
        dsm = np.where(valid, dsm, np.nan)
        derived = derive_surfaces(dsm, valid=valid, gsd_m=gsd_m, ground_cfg=ground_cfg, max_building_extent_m=max_extent)
        product_dir = out / Path(name).stem
        exports = export_surfaces(
            product_dir, dsm=dsm, dtm=derived["dtm"].dtm, ndsm=derived["ndsm"].ndsm, ground=derived["dtm"].ground,
            valid=valid, crs=raster.crs, transform=raster.transform, nodata=cfg.nodata,
            tags={"SYNTHETIC": "true", "PRODUCER": "depthwizard.surfaces.phase6", "SOURCE_DSM": str(path)},
        )
        score = _score(derived, dsm, valid, terrain, heights, building, gsd_m=gsd_m, slope_magnitude=slope_magnitude)
        clicks = []
        for b in truth["buildings"]:
            x, y = b["centroid_easting_m"], b["centroid_northing_m"]
            nd = height_at(product_dir / "ndsm.tif", x, y)
            clicks.append({
                "building": b["name"], "easting_m": x, "northing_m": y, "specified_height_m": b["height_m"],
                "ndsm_m": nd, "dsm_m": height_at(product_dir / "dsm.tif", x, y), "dtm_m": height_at(product_dir / "dtm.tif", x, y),
                "abs_error_m": abs(nd - b["height_m"]) if nd is not None else None,
                "within_tolerance": nd is not None and abs(nd - b["height_m"]) <= score["tolerance"]["ndsm_m"],
            })
        products[name] = {
            "role": "ACCEPTANCE INPUT" if index == 0 else "secondary diagnostic",
            "source_dsm": str(path),
            "grid": {"crs": raster.crs.to_string(), "transform": [float(v) for v in tuple(raster.transform)[:6]],
                     "width": raster.width, "height": raster.height, "gsd_m": gsd_m},
            "ground_filter": derived["ground"].diagnostics,
            "dtm": derived["dtm"].diagnostics,
            "score": score,
            "clicks": clicks,
            "clicks_passed": all(c["within_tolerance"] for c in clicks),
            "exports": {k: {kk: v[kk] for kk in ("path", "crs", "transform", "width", "height", "nodata", "dtype",
                                                 "layout", "ok", "max_abs_diff_vs_memory")} for k, v in exports.items()},
        }
        log.info("phase 6 product done", extra={"product": name, "passed": score["passed"]})

    primary = products[s.dsm_products[0]]
    report = {
        "phase": "6",
        "label": SYNTHETIC_LABEL,
        "definitions": {"DSM": "absolute surface elevation", "DTM": "absolute bare-ground elevation",
                        "nDSM": "DSM - DTM: object height above the extracted ground"},
        "independence": "DSM-only: no Phase 3/4a AGL, DEM or label is an input; fixture truth is used for scoring and, "
                        "when max_building_extent_m is null, for that one parameter",
        "parameters": {
            "max_building_extent_m": max_extent, "max_building_extent_source": extent_source,
            "max_terrain_slope": ground_cfg.max_terrain_slope, "min_object_height_m": ground_cfg.min_object_height_m,
            "tin_refinement_max_iterations": ground_cfg.tin_refinement_max_iterations,
            "fixture_terrain_slope_magnitude": slope_magnitude,
            "fixture_building_heights_m": sorted(b["height_m"] for b in truth["buildings"]),
        },
        "products": products,
        "acceptance": {
            "input": s.dsm_products[0],
            "passed": bool(primary["score"]["passed"] and primary["clicks_passed"]),
            "rule": "every valid pixel: |DTM - T| <= eps-bound and |nDSM - H| <= 2*eps-bound; "
                    "height_at(ndsm) at every building centroid within the nDSM bound",
        },
        "real_world_validation": REAL_WORLD_STATUS,
        "reproducibility": {"command": command or "python -m depthwizard.surfaces.phase6 synthetic",
                            "config_path": str(config_path), "config": cfg.to_dict(),
                            "inputs": {"phase4b_report": str(s.phase4b_report), "truth": str(truth_path),
                                       "truth_height": str(height_path)}},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "phase6_report.json", report)
    (out / "phase6_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.3e}" if value != 0 and abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """The Phase 6 acceptance report. Formats recorded values only."""
    p = report["parameters"]
    lines = ["# Phase 6 — DTM and nDSM (acceptance report)", "", f"> **{report['label']}**", ">",
             f"> Real-world DTM/nDSM validation: **{report['real_world_validation']}**", "",
             f"Acceptance ({report['acceptance']['input']}): **{'PASSED' if report['acceptance']['passed'] else 'FAILED'}**",
             f"— {report['acceptance']['rule']}.", "", "## Definitions", ""]
    lines += [f"- **{k}**: {v}" for k, v in report["definitions"].items()]
    lines += ["", f"Independence: {report['independence']}.", "", "## Parameters", "", "| parameter | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v) if not isinstance(v, (list, str)) else v} |" for k, v in p.items()]
    for name, prod in report["products"].items():
        sc = prod["score"]
        lines += ["", f"## {name} ({prod['role']})", "",
                  f"- eps = max|DSM − DSM_true| = {_fmt(sc['eps_max_abs_dsm_error_m'])} m; tolerance DTM "
                  f"{_fmt(sc['tolerance']['dtm_m'])} m, nDSM {_fmt(sc['tolerance']['ndsm_m'])} m ({sc['tolerance']['derivation']})",
                  f"- max |DTM − T|: all {_fmt(sc['dtm_vs_terrain_truth']['all_valid_max_abs_m'])} m, terrain "
                  f"{_fmt(sc['dtm_vs_terrain_truth']['terrain_px_max_abs_m'])} m, under buildings "
                  f"{_fmt(sc['dtm_vs_terrain_truth']['under_buildings_max_abs_m'])} m",
                  f"- max |nDSM − H|: all {_fmt(sc['ndsm_vs_height_truth']['all_valid_max_abs_m'])} m, buildings "
                  f"{_fmt(sc['ndsm_vs_height_truth']['building_px_max_abs_m'])} m, terrain "
                  f"{_fmt(sc['ndsm_vs_height_truth']['terrain_px_max_abs_m'])} m",
                  f"- classification: {sc['classification']}",
                  f"- nDSM: {sc['ndsm']}; extrapolated pixels: {sc['extrapolated_px']}",
                  f"- within derived bounds: **{sc['passed']}**", "",
                  "| building | specified H (m) | nDSM at centroid (m) | DSM there (absolute, m) | DTM there (m) | |error| (m) | within bound |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for c in prod["clicks"]:
            lines.append(f"| {c['building']} | {_fmt(c['specified_height_m'])} | {_fmt(c['ndsm_m'])} | {_fmt(c['dsm_m'])} | "
                         f"{_fmt(c['dtm_m'])} | {_fmt(c['abs_error_m'])} | {c['within_tolerance']} |")
        lines += ["", "Exports (validated on read-back): " + ", ".join(
            f"`{v['path']}` ({v['crs']}, {v['width']}×{v['height']}, {v['dtype']}, nodata {v['nodata']}, ok={v['ok']})"
            for v in prod["exports"].values())]
    lines += ["", "## Limitations", "",
              "- SYNTHETIC best case: planar terrain, flat roofs, sharp edges, no vegetation or noise.",
              "- DTM quality depends on ground classification, max_building_extent_m versus the true largest building, "
              "DSM resolution and noise, terrain curvature versus the thresholds, and scene borders (extrapolation).",
              "- The filter flags anything raised (trees, bridges, vehicles) as non-ground; it is not a building detector.",
              "- TIN refinement is simple re-admission, NOT Axelsson progressive TIN densification, NOT cloth simulation.",
              f"- Real-world DTM/nDSM validation: {report['real_world_validation']}.", "",
              "## Reproducibility", "", f"- command: `{report['reproducibility']['command']}`",
              f"- config: `{report['reproducibility']['config_path']}`", ""]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    import sys

    from depthwizard.logging_setup import setup_logging

    argv = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description="DepthWizard Phase 6: DTM and nDSM from a DSM")
    parser.add_argument("target", choices=("synthetic",))
    parser.add_argument("--config", default="configs/phase6.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    setup_logging({"level": "INFO", "format": "text"})
    command = "python -m depthwizard.surfaces.phase6 " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        cfg = load_phase6_config(args.config)
        if args.output_dir is not None:
            from dataclasses import replace

            cfg = replace(cfg, output_dir=Path(args.output_dir))
        report = run_phase6_synthetic(cfg, command=command, config_path=args.config)
    except (Phase6Error, ConfigError, GroundExtractionError, DtmError) as exc:
        print(f"PHASE 6 FAILED: {exc}")
        return 2
    print(SYNTHETIC_LABEL)
    for name, prod in report["products"].items():
        sc = prod["score"]
        print(f"[{name}] ({prod['role']}) eps={sc['eps_max_abs_dsm_error_m']:.3e} m  "
              f"max|DTM-T|={sc['dtm_vs_terrain_truth']['all_valid_max_abs_m']:.3e} (bound {sc['tolerance']['dtm_m']:.3e})  "
              f"max|nDSM-H|={sc['ndsm_vs_height_truth']['all_valid_max_abs_m']:.3e} (bound {sc['tolerance']['ndsm_m']:.3e})  "
              f"passed={sc['passed']}")
        for c in prod["clicks"]:
            print(f"    click {c['building']:8s} nDSM={c['ndsm_m']:.4f} m (H={c['specified_height_m']}) DSM={c['dsm_m']:.4f} m")
    print(f"acceptance: {'PASSED' if report['acceptance']['passed'] else 'FAILED'}")
    print(f"real-world DTM/nDSM validation: {report['real_world_validation']}")
    print(f"report: {Path(cfg.output_dir) / 'synthetic' / 'phase6_report.md'}")
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
