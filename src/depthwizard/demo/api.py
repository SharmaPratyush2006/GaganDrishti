"""Minimal FastAPI wrapper around the Phase 8 demo (demo-grade: no auth, no queue, no database).

::

    python -m depthwizard.demo serve            # http://127.0.0.1:8000, from the repo root

Endpoints:

* ``GET  /health``          -- liveness.
* ``POST /process``         -- upload one image (multipart field ``file``; optional
  form field ``require`` = ``auto`` | ``absolute``). Runs
  :func:`depthwizard.demo.process.process_image`.
* ``POST /demo/synthetic``  -- run the deterministic SYNTHETIC end-to-end demo
  (:func:`depthwizard.demo.synthetic.run_synthetic_demo`).
* ``GET  /outputs/{path}``  -- read-only access to ``data/outputs`` (the same
  directory the viewer serves), with path escapes refused.

Errors are JSON ``{"error": <kind>, "message": ...}``; stack traces go to the
server log only.
"""

from __future__ import annotations

import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module level, not inside create_app(): with postponed annotations, FastAPI resolves
# the endpoint signatures (UploadFile, ...) from this module's globals.
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from depthwizard.demo.common import OUTPUTS_ROOT
from depthwizard.demo.process import InputError, ProcessingError, process_image, write_response
from depthwizard.ingest.router import SUPPORTED_SUFFIXES
from depthwizard.logging_setup import get_logger

__all__ = ["MAX_UPLOAD_BYTES", "create_app"]

log = get_logger(__name__)

#: Uploads larger than this are refused (demo-grade limit, not a pipeline limit).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

_STATUS = {"unsupported_format": 415, "corrupt_image": 422, "missing_metadata": 422, "invalid_mode": 400}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def create_app(outputs_root: str | Path = OUTPUTS_ROOT, *, checkpoint: str | Path | None = None,
               device: str | None = None) -> Any:
    """Build the app. ``outputs_root`` is the served directory (``data/outputs`` from the repo root)."""
    root = Path(outputs_root)
    uploads = root / "uploads"
    demo_lock = threading.Lock()
    app = FastAPI(title="DepthWizard demo API", version="0.1.0",
                  description="Phase 8 demo wrapper. RELATIVE results are unitless; metres only on the metric path.")

    def error(status: int, kind: str, message: str, **extra: Any) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": kind, "message": message, **extra})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request, exc):  # noqa: ANN001
        return error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_request, exc):  # noqa: ANN001
        fields = [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()]
        return error(422, "invalid_request", f"invalid or missing request fields: {', '.join(fields)}")

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc):  # noqa: ANN001
        log.exception("unhandled API error", exc_info=exc)
        return error(500, "internal_error", "internal error; see the server log")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "project": "DepthWizard", "phase": 8}

    @app.post("/process")
    async def process(file: UploadFile = File(...), require: str = Form("auto")):
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return error(415, "unsupported_format",
                         f"unsupported file type {suffix or '(none)'!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")
        if require not in ("auto", "absolute"):
            return error(400, "invalid_mode", f"require must be 'auto' or 'absolute', got {require!r}")
        run_id = _run_id()
        workspace = uploads / run_id
        input_path = workspace / "input" / (_SAFE_NAME.sub("_", Path(name).stem) or "upload")
        input_path = input_path.with_suffix(suffix)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with input_path.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(workspace, ignore_errors=True)
                    return error(413, "too_large", f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
                fh.write(chunk)
        if size == 0:
            shutil.rmtree(workspace, ignore_errors=True)
            return error(422, "corrupt_image", "the uploaded file is empty")
        try:
            record = process_image(input_path, workspace, require=require, checkpoint=checkpoint, device=device)
        except InputError as exc:
            shutil.rmtree(workspace, ignore_errors=True)
            return error(_STATUS.get(exc.kind, 422), exc.kind, str(exc), **exc.detail)
        except ProcessingError as exc:
            log.exception("processing failed", extra={"run_id": run_id})
            return error(500, "processing_failed", f"processing failed for run {run_id}; see the server log",
                         run_id=run_id, stage=str(exc).split(":")[0])
        record = {"run_id": run_id, **record}
        # URLs on THIS app's /outputs, relative to the root it serves (not to the working directory).
        base = root.resolve()
        record["outputs"] = {"report": f"/outputs/uploads/{run_id}/process_report.json"}
        for k, f in record["products"]["relative_height"].get("files", {}).items():
            target = Path(f).resolve()
            if base in target.parents:
                record["outputs"][k] = "/outputs/" + target.relative_to(base).as_posix()
        write_response(record, workspace)
        return record

    @app.post("/demo/synthetic")
    def demo_synthetic(mesh: bool = True):
        from depthwizard.demo.synthetic import DemoError, run_synthetic_demo

        if not demo_lock.acquire(blocking=False):
            return error(409, "busy", "a synthetic demo run is already in progress")
        try:
            report = run_synthetic_demo(root / "demo", export_mesh=mesh)
        except DemoError as exc:
            log.exception("synthetic demo failed")
            return error(500, "processing_failed", str(exc).split(":")[0] + ": see the server log")
        finally:
            demo_lock.release()
        return {"report": "/outputs/demo/reports/demo_report.json", **report}

    @app.get("/outputs/{path:path}")
    def outputs(path: str):
        base = root.resolve()
        parts = Path(path).parts
        if not path or ".." in parts or Path(path).is_absolute() or "\\" in path or "\0" in path:
            return error(403, "forbidden", "path is outside the outputs directory")
        target = (base / path).resolve()
        if base not in target.parents:
            return error(403, "forbidden", "path is outside the outputs directory")
        if not target.is_file():
            return error(404, "unavailable_output", f"output not found: {path}")
        return FileResponse(target)

    return app
