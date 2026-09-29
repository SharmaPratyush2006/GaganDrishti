"""``python -m depthwizard.demo {synthetic,process,serve}``.

All three change into the repository root first, so recorded paths are
repo-relative and the viewer (which serves ``data/outputs``) can follow them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from depthwizard.demo.common import NOT_MEASURED, REPO_ROOT


def _to_repo_root() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        print(f"(working directory -> {REPO_ROOT})", file=sys.stderr)
        os.chdir(REPO_ROOT)


def main(argv: Iterable[str] | None = None) -> int:
    from depthwizard.logging_setup import setup_logging

    parser = argparse.ArgumentParser(prog="python -m depthwizard.demo", description="DepthWizard Phase 8 demo")
    sub = parser.add_subparsers(dest="command", required=True)
    syn = sub.add_parser("synthetic", help="deterministic end-to-end demo on the SYNTHETIC fixture")
    syn.add_argument("--out", default="data/outputs/demo")
    syn.add_argument("--no-mesh", action="store_true", help="skip the glTF/Draco export (needs Node)")
    proc = sub.add_parser("process", help="process one image (ingest, mode, Phase 3 relative height)")
    proc.add_argument("image")
    proc.add_argument("--out", default=None, help="workspace (default data/outputs/uploads/<run-id>)")
    proc.add_argument("--require", choices=("auto", "absolute"), default="auto")
    srv = sub.add_parser("serve", help="run the FastAPI demo server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Paths given on the command line are relative to where the user typed them.
    if getattr(args, "image", None):
        args.image = str(Path(args.image).resolve())
    if getattr(args, "out", None) and not Path(args.out).is_absolute() and args.command == "process":
        args.out = str(Path(args.out).resolve())
    _to_repo_root()
    setup_logging({"level": "INFO", "format": "text"})

    if args.command == "synthetic":
        from depthwizard.demo.synthetic import DemoError, run_synthetic_demo

        try:
            report = run_synthetic_demo(args.out, export_mesh=not args.no_mesh)
        except DemoError as exc:
            print(f"DEMO FAILED: {exc}")
            return 2
        print(report["label"])
        md = report["input"]["metadata"]
        print(f"mode {report['mode']} ({report['units']}); CRS {md['crs']}; GSD {md['gsd_m']} m; "
              f"sun {md['sun_elevation_deg']}/{md['sun_azimuth_deg']} deg")
        p2 = report["phase2_shadow_physics"]
        for b in p2["buildings"]:
            print(f"  Phase 2 {b['building']:8s} L={b['shadow_length_m']:.3f} m -> h={b['height_m']:.3f} m "
                  f"(specified {b['specified_height_m']} m)")
        print(f"  Phase 2 MAE {p2['mae_m']:.4f} m, max {p2['max_abs_error_m']:.4f} m (pixel-grid bound "
              f"{p2['pixel_grid_bound_m']:.4f} m)")
        for c in report["phase6_surfaces"]["clicks"]:
            print(f"  Phase 6 nDSM at {c['building']:8s} = {c['ndsm_m']:.4f} m (specified {c['specified_height_m']} m)")
        mesh = report["mesh"]
        for f in mesh.get("files", []):
            print(f"  mesh {f['path']} ({f['bytes']:,} B, draco verified: {f['draco_verified']})")
        if mesh.get("status") != "produced":
            print(f"  mesh: not available ({mesh.get('reason')})")
        print(f"uncertainty: {NOT_MEASURED}; FPS: {NOT_MEASURED}")
        print(f"report: {args.out}/reports/demo_report.md")
        print(f"viewer: {report['viewer']['url']}")
        return 0

    if args.command == "process":
        from depthwizard.demo.api import _run_id
        from depthwizard.demo.process import InputError, ProcessingError, process_image, write_response

        workspace = Path(args.out) if args.out else Path("data/outputs/uploads") / _run_id()
        try:
            record = process_image(args.image, workspace, require=args.require)
        except (InputError, ProcessingError) as exc:
            print(f"PROCESS FAILED: {exc}")
            return 2
        write_response(record, workspace)
        print(json.dumps(record, indent=2))
        return 0

    import uvicorn

    from depthwizard.demo.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
