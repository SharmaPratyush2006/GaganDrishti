"""Production serving for the Phase 8 demo: the API plus the built viewer, on one port.

::

    npm ci --prefix viewer && npm run build --prefix viewer   # -> viewer/dist
    python -m depthwizard.demo synthetic                       # -> data/outputs/demo
    python -m depthwizard.demo.production                      # 0.0.0.0:$PORT (default 8000)

This wraps :func:`depthwizard.demo.api.create_app` unchanged (``/health``,
``/process``, ``/demo/synthetic``, ``/outputs/{path}``) and adds only what the
built viewer needs once the Vite dev server is gone:

* ``GET /__data_index`` and ``GET /data/outputs/{path}`` -- the same read-only
  view of ``data/outputs`` that ``viewer/server/dataRoute.js`` gives ``npm run dev``
  (same index shape, same path-escape refusals);
* ``viewer/dist`` at ``/`` (``index.html``, ``/assets/*``), mounted after every
  API route so it cannot shadow one. Only files inside ``viewer/dist`` are served.
"""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from depthwizard.demo.api import create_app
from depthwizard.demo.common import OUTPUTS_ROOT, REPO_ROOT

__all__ = ["DIST_DIR", "build_index", "create_production_app", "main"]

#: The Vite build output (``npm run build --prefix viewer``).
DIST_DIR = REPO_ROOT / "viewer" / "dist"

# Mirrors viewer/server/dataRoute.js.
_REPORT_RE = re.compile(r"^(?:phase\d+[a-z]?|demo)_report\.json$")
_MAX_DEPTH = 6

# Host MIME tables vary (the Windows registry can map .js to text/plain); browsers
# refuse module scripts and streaming WebAssembly served with the wrong type.
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".wasm", "application/wasm"), (".json", "application/json")):
    mimetypes.add_type(_type, _ext)


def build_index(outputs_root: str | Path) -> dict[str, Any]:
    """Reports and .npy arrays under ``outputs_root``, as ``dataRoute.js`` ``buildIndex`` lists them."""
    root = Path(outputs_root)
    reports: list[str] = []
    npy: list[str] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                walk(Path(e.path), depth + 1)
            elif e.is_file(follow_symlinks=False):
                rel = "data/outputs/" + Path(e.path).relative_to(root).as_posix()
                if _REPORT_RE.match(e.name):
                    reports.append(rel)
                elif e.name.endswith(".npy"):
                    npy.append(rel)

    walk(root, 0)
    return {"root": "data/outputs", "reports": sorted(reports), "npy": sorted(npy)}


def create_production_app(outputs_root: str | Path = REPO_ROOT / OUTPUTS_ROOT,
                          dist_dir: str | Path = DIST_DIR, **api_kwargs: Any) -> Any:
    """The demo API with the viewer's data route and ``viewer/dist`` added.

    Raises if ``dist_dir`` has no ``index.html`` (the viewer was not built).
    """
    dist = Path(dist_dir)
    if not (dist / "index.html").is_file():
        raise RuntimeError(f"{dist / 'index.html'} not found; run `npm run build --prefix viewer` first")
    root = Path(outputs_root)
    app = create_app(root, **api_kwargs)

    def forbidden() -> JSONResponse:
        return JSONResponse(status_code=403,
                            content={"error": "forbidden", "message": "path is outside the outputs directory"})

    @app.get("/__data_index")
    def data_index():
        return JSONResponse(build_index(root.resolve()), headers={"Cache-Control": "no-store"})

    @app.get("/data/outputs/{path:path}")
    def data_outputs(path: str):
        base = root.resolve()
        if not path or ".." in Path(path).parts or Path(path).is_absolute() or "\\" in path or "\0" in path:
            return forbidden()
        target = (base / path).resolve()
        if base not in target.parents:
            return forbidden()
        if not target.is_file():
            return JSONResponse(status_code=404,
                                content={"error": "unavailable_output", "message": f"output not found: {path}"})
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    # Last, so every route above wins. StaticFiles refuses paths outside `dist`.
    app.mount("/", StaticFiles(directory=dist, html=True), name="viewer")
    return app


def main() -> None:
    """Render entry point: 0.0.0.0 on ``$PORT`` (default 8000), from the repository root."""
    import uvicorn

    from depthwizard.logging_setup import setup_logging

    os.chdir(REPO_ROOT)  # report paths are recorded repo-relative, as the CLI does
    setup_logging({"level": "INFO", "format": "text"})
    uvicorn.run(create_production_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
