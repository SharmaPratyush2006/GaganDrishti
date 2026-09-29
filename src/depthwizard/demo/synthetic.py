"""The deterministic end-to-end demo, on the Phase 0 SYNTHETIC fixture.

::

    python -m depthwizard.demo synthetic            # -> data/outputs/demo/

Every stage is an existing module; this file only runs them in order, points
their outputs into one tree, and records what each one produced::

    Phase 4b  surfaces.run.run_synthetic_dsm      fixture + DEM + Phase 4a synthetic
                                                  calibration -> DSM (COG)
    Phase 1   ingest.route                        the fixture GeoTIFF -> ABSOLUTE,
                                                  CRS / GSD / sun with provenance
    Phase 2   shadows.ShadowHeightPipeline        shadow length -> h = L tan(theta)
                                                  (footprints SUPPLIED from the
                                                  fixture's truth sidecar)
    Phase 5   validation.run.run_synthetic        metrics, error map, confidence
    Phase 6   surfaces.phase6.run_phase6_synthetic  DSM -> ground -> DTM -> nDSM
    Phase 7   viewer/scripts/export-mesh.mjs      the viewer's own glTF / Draco exporter

Output tree (git-ignored, under ``data/outputs``)::

    demo/input/        the fixture image and truth sidecar (copies of processed/phase4b/fixture)
    demo/processed/    phase4b/ (DSM, DEM, AGL, fixture), phase6/synthetic/ (DSM, DTM, nDSM)
    demo/validation/   synthetic/ (Phase 5 report, error maps, confidence rasters)
    demo/mesh/         nDSM .glb (uncompressed and Draco) + export summary
    demo/reports/      demo_report.json / .md, Phase 2 table and figure

What this demo is NOT: the relative field fed to Phase 4b is the documented
SYNTHETIC stand-in ``z_rel = log(AGL_true + 1) + c``, not a Phase 3 network
prediction (the fixture is a flat grayscale rendering the network was never
trained on). Heights were SPECIFIED when the fixture was generated. Nothing
here is real-world accuracy.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from depthwizard.calibration.diagnostics import to_jsonable, write_json
from depthwizard.demo.common import (
    METRES,
    NOT_MEASURED,
    REPO_ROOT,
    metadata_summary,
    not_available,
    report_path,
    url_for,
)
from depthwizard.ingest.geotiff import read_raster
from depthwizard.ingest.router import route
from depthwizard.logging_setup import get_logger
from depthwizard.mode import Mode

__all__ = ["DEFAULT_OUTPUT_DIR", "SYNTHETIC_DEMO_LABEL", "DemoError", "run_synthetic_demo", "render_markdown"]

log = get_logger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/outputs/demo")
SUBDIRS = ("input", "processed", "validation", "mesh", "reports")
MESH_SCRIPT = REPO_ROOT / "viewer" / "scripts" / "export-mesh.mjs"

SYNTHETIC_DEMO_LABEL = (
    "SYNTHETIC: the fixture's terrain and building heights were SPECIFIED when it was generated, not measured. "
    "The relative field fed to Phase 4b is a constructed stand-in (log(AGL_true + 1) + c), not a Phase 3 "
    "prediction. These results demonstrate the pipeline's geometry and plumbing, not real-world accuracy."
)

#: What the project has not measured. Rendered verbatim; never replaced by a number.
NOT_YET_MEASURED = {
    "fps": NOT_MEASURED,
    "per_pixel_uncertainty": NOT_MEASURED,
    "real_world_accuracy_on_arbitrary_satellite_imagery": NOT_MEASURED,
    "real_world_dtm_ndsm_accuracy": NOT_MEASURED,
    "real_data_georeferenced_dsm_accuracy": NOT_MEASURED,
}


class DemoError(RuntimeError):
    """A demo stage failed; the message names the stage."""


def _clean(out: Path) -> None:
    """Remove only the demo's own sub-directories, so every run starts from nothing."""
    for name in SUBDIRS:
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)


def _phase2(image_path: Path, truth: dict[str, Any], metadata: Any) -> dict[str, Any]:
    """Run the Phase 2 pipeline on the fixture with its SUPPLIED footprints."""
    from depthwizard.shadows import BuildingFootprint, DetectionContext, ShadowHeightPipeline

    image = read_raster(image_path).array
    footprints = [
        BuildingFootprint.from_bbox(b["name"], shape=image.shape, **{k: b["footprint_px"][k] for k in
                                                                      ("row_min", "row_max", "col_min", "col_max")})
        for b in truth["buildings"]
    ]
    context = DetectionContext.from_scene_metadata(metadata)
    result = ShadowHeightPipeline().run(image, footprints, context)
    specified = {b["name"]: b["height_m"] for b in truth["buildings"]}
    rows = []
    for est in result:
        h = est.height_m
        rows.append({
            "building": est.building_id,
            "shadow_length_px": est.shadow_length_px,
            "shadow_length_m": est.shadow_length_m,
            "sun_elevation_deg": result.sun_elevation_deg,
            "tan_elevation": math.tan(math.radians(result.sun_elevation_deg)),
            "height_m": h,
            "specified_height_m": specified[est.building_id],
            "abs_error_m": abs(h - specified[est.building_id]) if h is not None else None,
            "confidence": est.confidence.value,
            "equation": f"h = L * tan(theta) = {est.shadow_length_m:.4f} m * tan({result.sun_elevation_deg:g} deg)"
                        if est.shadow_length_m is not None else None,
        })
    errors = [r["abs_error_m"] for r in rows if r["abs_error_m"] is not None]
    grid_bound = (math.sqrt(2.0) / 2.0) * result.gsd_m * math.tan(math.radians(result.sun_elevation_deg))
    return {
        "footprints": "SUPPLIED from the fixture's truth sidecar; nothing here detects buildings",
        "gsd_m": result.gsd_m,
        "sun_elevation_deg": result.sun_elevation_deg,
        "sun_azimuth_deg": result.sun_azimuth_deg,
        "shadow_azimuth_deg": result.shadow_azimuth_deg,
        "detector": result.shadow_mask.to_dict(),
        "buildings": rows,
        "succeeded": len(errors),
        "mae_m": float(np.mean(errors)) if errors else None,
        "max_abs_error_m": float(np.max(errors)) if errors else None,
        "pixel_grid_bound_m": grid_bound,
        "error_source": "pixel quantisation of the diagonal shadow (bound (sqrt(2)/2) px * GSD * tan(theta))",
        "_mask": result.shadow_mask.mask,
    }


def _phase2_figure(path: Path, image_path: Path, truth: dict[str, Any], phase2: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = read_raster(image_path).array
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    from matplotlib.colors import ListedColormap

    shadow = np.ma.masked_where(~phase2["_mask"], phase2["_mask"].astype(float))
    ax.imshow(shadow, cmap=ListedColormap(["#ff8c00"]), alpha=0.6, vmin=0, vmax=1)
    rows = {r["building"]: r for r in phase2["buildings"]}
    for b in truth["buildings"]:
        fp = b["footprint_px"]
        ax.add_patch(Rectangle((fp["col_min"] - 0.5, fp["row_min"] - 0.5), fp["col_max"] - fp["col_min"],
                               fp["row_max"] - fp["row_min"], fill=False, edgecolor="cyan", linewidth=1.5))
        r = rows[b["name"]]
        text = (f"{b['name']}\nL = {r['shadow_length_px']:.2f} px = {r['shadow_length_m']:.2f} m\n"
                f"h = L·tan{phase2['sun_elevation_deg']:g}° = {r['height_m']:.2f} m\nspecified {r['specified_height_m']:g} m")
        # Below the footprint: the shadows fall up-left (sun from the south-east).
        ax.text(fp["col_min"], fp["row_max"] + 4, text, color="black", fontsize=7.5, va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
    ax.set_title(f"Phase 2 shadow physics — SYNTHETIC fixture (sun {phase2['sun_elevation_deg']:g}° elev, "
                 f"{phase2['sun_azimuth_deg']:g}° az, GSD {phase2['gsd_m']:g} m)\n"
                 "orange = detected shadow, cyan = SUPPLIED footprints\n"
                 f"MAE {phase2['mae_m']:.3f} m, max {phase2['max_abs_error_m']:.3f} m vs SPECIFIED heights "
                 "(pixel quantisation; not real-world accuracy)", fontsize=9)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _export_mesh(phase6_report: Path, out: Path, node: str | None) -> dict[str, Any]:
    """Run the Phase 7 viewer's glTF/Draco exporter under Node on the nDSM."""
    node_bin = node or shutil.which("node")
    if node_bin is None:
        return not_available("Node.js was not found on PATH; the mesh is exported by the Phase 7 viewer's "
                             "JavaScript exporter (install Node and run `npm ci` in viewer/)")
    if not (REPO_ROOT / "viewer" / "node_modules").is_dir():
        return not_available("viewer/node_modules is missing; run `npm ci` in viewer/ first")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [node_bin, str(MESH_SCRIPT), "--phase6-report", Path(report_path(phase6_report)).as_posix(),
           "--out", str(out), "--outputs-dir", str(Path.cwd() / "data" / "outputs")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise DemoError(f"mesh export failed (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}")
    summary = json.loads((out / "mesh_summary.json").read_text(encoding="utf-8"))
    return {"status": "produced", "command": " ".join(cmd), **summary}


def _metric(entry: dict[str, Any], category: str, name: str) -> Any:
    m = entry["metrics"].get(category, {}).get(name, {})
    return m.get("value") if m.get("value") is not None else m.get("reason") or NOT_MEASURED


def run_synthetic_demo(
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    phase4_config: str | Path = "configs/phase4.yaml",
    phase5_config: str | Path = "configs/phase5.yaml",
    phase6_config: str | Path = "configs/phase6.yaml",
    export_mesh: bool = True,
    node: str | None = None,
) -> dict[str, Any]:
    """Run the SYNTHETIC end-to-end demo into ``out_dir`` and return its report.

    Run from the repository root, so the recorded paths are repo-relative and
    the viewer can follow them (the CLI and the API change into it).
    """
    from depthwizard.surfaces.phase6 import run_phase6_synthetic
    from depthwizard.surfaces.phase6_config import load_phase6_config
    from depthwizard.surfaces.run import run_synthetic_dsm
    from depthwizard.validation.config import load_phase5_config
    from depthwizard.validation.run import run_synthetic

    out = Path(report_path(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    _clean(out)
    stages: list[dict[str, Any]] = []

    def stage(name: str, phase: str, module: str, fn):
        try:
            value = fn()
        except DemoError:
            raise
        except Exception as exc:  # the stage name is what an operator needs
            raise DemoError(f"{name} ({phase}) failed: {exc}") from exc
        stages.append({"stage": name, "phase": phase, "module": module, "status": "ok"})
        return value

    # Phase 4b first: it generates the fixture (Phase 0) that everything else reads.
    processed = out / "processed"
    report4b = stage("georeferenced DSM", "4b (+ Phase 0 fixture, Phase 4a synthetic calibration)",
                     "depthwizard.surfaces.run.run_synthetic_dsm",
                     lambda: run_synthetic_dsm(phase4_config, output_dir=processed / "phase4b"))
    phase4b_report = processed / "phase4b" / "phase4b_report.json"

    fixture = Path(report4b["image_grid"]["source"])
    input_dir = out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".tif", "_truth.json"):
        shutil.copy2(fixture.with_name(fixture.stem + suffix), input_dir / (fixture.stem + suffix))
    image_path = input_dir / fixture.name
    truth = json.loads((input_dir / f"{fixture.stem}_truth.json").read_text(encoding="utf-8"))

    decision = stage("ingest + mode", "1", "depthwizard.ingest.route", lambda: route(image_path))
    if decision.mode is not Mode.ABSOLUTE:
        raise DemoError(f"the synthetic fixture routed to {decision.mode.value}: {decision.reason}")

    phase2 = stage("shadow physics", "2", "depthwizard.shadows.ShadowHeightPipeline",
                   lambda: _phase2(image_path, truth, decision.metadata))
    reports_dir = out / "reports"
    figure = _phase2_figure(reports_dir / "phase2_shadow_heights.png", image_path, truth, phase2)
    phase2.pop("_mask")
    write_json(reports_dir / "phase2_shadow_heights.json", {"label": SYNTHETIC_DEMO_LABEL, **phase2})

    cfg5 = load_phase5_config(phase5_config)
    cfg5 = replace(cfg5, output_dir=out / "validation", synthetic=replace(cfg5.synthetic, phase4b_report=phase4b_report))
    report5 = stage("validation", "5", "depthwizard.validation.run.run_synthetic",
                    lambda: run_synthetic(cfg5, command="python -m depthwizard.demo synthetic",
                                          config_path=str(phase5_config)))

    cfg6 = load_phase6_config(phase6_config)
    cfg6 = replace(cfg6, output_dir=processed / "phase6", synthetic=replace(cfg6.synthetic, phase4b_report=phase4b_report))
    report6 = stage("DTM + nDSM", "6", "depthwizard.surfaces.phase6.run_phase6_synthetic",
                    lambda: run_phase6_synthetic(cfg6, command="python -m depthwizard.demo synthetic",
                                                 config_path=str(phase6_config)))
    phase6_report = processed / "phase6" / "synthetic" / "phase6_report.json"
    if not report6["acceptance"]["passed"]:
        raise DemoError("Phase 6 SYNTHETIC acceptance did not pass; see " + str(phase6_report))

    if export_mesh:
        mesh = stage("glTF / Draco mesh", "7", "viewer/src/export/gltfExport.js (via viewer/scripts/export-mesh.mjs)",
                     lambda: _export_mesh(phase6_report, out / "mesh", node))
    else:
        mesh = not_available("mesh export was switched off for this run (--no-mesh)")

    acceptance_product = report6["acceptance"]["input"]
    report = {
        "phase": "8",
        "kind": "synthetic end-to-end demo",
        "label": SYNTHETIC_DEMO_LABEL,
        "mode": decision.mode.value.upper(),
        "units": METRES,
        "command": "python -m depthwizard.demo synthetic",
        "stages": stages,
        "input": {
            "image": report_path(image_path),
            "image_url": url_for(image_path),
            "routing": {"loader": decision.loader, "reason": decision.reason},
            "metadata": metadata_summary(decision.metadata),
            "buildings_specified": [{"name": b["name"], "height_m": b["height_m"]} for b in truth["buildings"]],
        },
        "phase2_shadow_physics": {**phase2, "figure": report_path(figure), "figure_url": url_for(figure),
                                  "label": "SYNTHETIC fixture; footprints supplied"},
        "phase4b_dsm": {
            "report": report_path(phase4b_report),
            "calibration": {
                "phase4a_synthetic": {k: report4b["calibrations"]["phase4a_synthetic_calibration"][k] for k in ("a", "b")},
                "known": {k: report4b["calibrations"]["known_calibration"][k] for k in ("a", "b")},
            },
            "dsm_vs_truth": {k: v["dsm_vs_truth"] for k, v in report4b["calibrations"].items()},
            "geometric_acceptance": report4b["geometric_acceptance"],
            "relative_field": report4b["relative_field"]["z_rel"],
        },
        "phase5_validation": {
            "report": report_path(out / "validation" / "synthetic" / "phase5_report.json"),
            "label": report5["label"],
            "products": [{
                "product": p["product"],
                "units": p["units"],
                "prediction": p["prediction_path"],
                "overall_mae": _metric(p, "overall", "mae"),
                "overall_rmse": _metric(p, "overall", "rmse"),
                "building_mae": _metric(p, "building", "mae"),
                "error_map_figure": p["error_map"]["figure"].get("path") if isinstance(p["error_map"]["figure"], dict)
                else p["error_map"]["figure"],
                "confidence_raster": p["confidence_raster"],
            } for p in report5["products"]],
            "real_world_validation": report5["real_world_validation"],
        },
        "phase6_surfaces": {
            "report": report_path(phase6_report),
            "acceptance_input": acceptance_product,
            "acceptance_passed": report6["acceptance"]["passed"],
            "clicks": report6["products"][acceptance_product]["clicks"],
            "real_world_validation": report6["real_world_validation"],
        },
        "mesh": mesh,
        "viewer": {
            "phase6_report": Path(report_path(phase6_report)).as_posix(),
            # "Load Phase 8 Demo" in the viewer; ?phase6Report=<phase6_report> loads the surfaces alone.
            "url": "http://localhost:5173/?demo=1",
            "overlays": "Phase 5 error/confidence overlays validate processed/phase4b/synthetic_dsm.tif, so they "
                        "appear for the Phase 6 product synthetic_dsm.tif with the DSM surface shown",
        },
        "uncertainty": NOT_MEASURED,
        "not_yet_measured": NOT_YET_MEASURED,
        "limitations": [
            "SYNTHETIC fixture: flat roofs, planar terrain, no noise, grayscale image, 4 axis-aligned buildings.",
            "Building footprints are SUPPLIED (fixture truth); nothing in DepthWizard detects buildings.",
            "The relative field is the constructed stand-in, not a Phase 3 prediction; Phase 3 is not applied "
            "to the grayscale fixture (it would have to fabricate RGB).",
            "Phase 5 overlays validate the Phase 4b DSM, not the Phase 6 nDSM.",
            f"Per-pixel uncertainty: {NOT_MEASURED}. FPS: {NOT_MEASURED}.",
        ],
    }
    write_json(reports_dir / "demo_report.json", report)
    (reports_dir / "demo_report.md").write_text(render_markdown(to_jsonable(report)), encoding="utf-8")
    log.info("synthetic demo complete", extra={"out": str(out)})
    return to_jsonable(report)


def _f(v: Any, digits: int = 4) -> str:
    if v is None:
        return NOT_MEASURED
    if isinstance(v, float):
        return f"{v:.3e}" if v != 0 and abs(v) < 1e-3 else f"{v:.{digits}f}"
    return str(v)


def render_markdown(r: dict[str, Any]) -> str:
    """Human-readable demo report. Formats recorded values only."""
    md = r["input"]["metadata"]
    p2 = r["phase2_shadow_physics"]
    lines = [
        "# DepthWizard — Phase 8 synthetic end-to-end demo", "",
        f"> **{r['label']}**", "",
        f"Mode: **{r['mode']}** · units: **{r['units']}** · command: `{r['command']}`", "",
        "## Stages", "", "| stage | phase | module | status |", "|---|---|---|---|",
        *[f"| {s['stage']} | {s['phase']} | `{s['module']}` | {s['status']} |" for s in r["stages"]],
        "", "## Input metadata (Phase 1)", "",
        f"- file: `{r['input']['image']}` ({md['width_px']} × {md['height_px']} px, {md['band_count']} band, {md['dtype']})",
        f"- CRS: {md['crs']} · GSD: {_f(md['gsd_m'])} m",
        f"- sun elevation: {_f(md['sun_elevation_deg'])}° ← {md['sun_elevation_source']}",
        f"- sun azimuth: {_f(md['sun_azimuth_deg'])}° ← {md['sun_azimuth_source']}",
        f"- routing: {r['input']['routing']['reason']}",
        "", "## Shadow physics (Phase 2)", "",
        f"Footprints: {p2['footprints']}.", "",
        "| building | L (px) | L (m) | h = L·tanθ (m) | specified (m) | abs. error (m) |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {b['building']} | {_f(b['shadow_length_px'])} | {_f(b['shadow_length_m'])} | {_f(b['height_m'])} | "
          f"{_f(b['specified_height_m'])} | {_f(b['abs_error_m'])} |" for b in p2["buildings"]],
        "", f"MAE {_f(p2['mae_m'])} m, max {_f(p2['max_abs_error_m'])} m; pixel-grid bound {_f(p2['pixel_grid_bound_m'])} m "
        f"({p2['error_source']}). Figure: `{p2['figure']}`.",
        "", "## DSM (Phase 4b)", "",
        f"- relative field: {r['phase4b_dsm']['relative_field']}",
        *[f"- {k}: max |DSM − truth| {_f(v['max_abs_error_m'])} m, MAE {_f(v['mae_m'])} m"
          for k, v in r["phase4b_dsm"]["dsm_vs_truth"].items()],
        "", "## Validation (Phase 5, SYNTHETIC)", "",
        "| product | units | overall MAE | overall RMSE | building MAE |", "|---|---|---:|---:|---:|",
        *[f"| {p['product']} | {p['units']} | {_f(p['overall_mae'])} | {_f(p['overall_rmse'])} | {_f(p['building_mae'])} |"
          for p in r["phase5_validation"]["products"]],
        "", f"Real-world validation: {r['phase5_validation']['real_world_validation']}.",
        "", "## DTM / nDSM (Phase 6) — click test at building centroids", "",
        "| building | specified H (m) | nDSM (m) | abs. error (m) |", "|---|---:|---:|---:|",
        *[f"| {c['building']} | {_f(c['specified_height_m'])} | {_f(c['ndsm_m'])} | {_f(c['abs_error_m'])} |"
          for c in r["phase6_surfaces"]["clicks"]],
        "", f"Acceptance ({r['phase6_surfaces']['acceptance_input']}): "
            f"**{'PASSED' if r['phase6_surfaces']['acceptance_passed'] else 'FAILED'}**",
        "", "## Mesh (Phase 7 exporter)", "",
    ]
    mesh = r["mesh"]
    if mesh.get("status") == "produced":
        lines += [f"- `{f['path']}`: {f['bytes']:,} B, {f['vertices']:,} vertices, {f['triangles']:,} triangles, "
                  f"Draco {'verified' if f['draco_verified'] else 'no'}" for f in mesh["files"]]
    else:
        lines.append(f"- not available: {mesh.get('reason')}")
    lines += ["", "## Viewer", "", f"Open {r['viewer']['url']} (Vite dev server running in `viewer/`).",
              "", "## Not yet measured", "", *[f"- {k}: {v}" for k, v in r["not_yet_measured"].items()],
              "", "## Limitations", "", *[f"- {x}" for x in r["limitations"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    from depthwizard.demo.__main__ import main

    sys.exit(main(["synthetic", *sys.argv[1:]]))
