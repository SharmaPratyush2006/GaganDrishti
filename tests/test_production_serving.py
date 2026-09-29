"""Deployment serving (depthwizard.demo.production): viewer/dist + the demo API on one app.

A temporary ``dist`` and outputs root keep these independent of a local build
or a local demo run; the last two tests also check the real ones when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from depthwizard.demo.production import DIST_DIR, build_index, create_production_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = '<!doctype html><html><head><script type="module" src="/assets/index-abc.js"></script></head></html>'
REPORT = {"phase": "8", "kind": "synthetic end-to-end demo"}


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (d / "assets" / "index-abc.js").write_text("export const x = 1;\n", encoding="utf-8")
    (d / "assets" / "index-abc.css").write_text("body{}\n", encoding="utf-8")
    (d / "assets" / "draco.wasm").write_bytes(b"\0asm\1\0\0\0")
    return d


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    o = tmp_path / "outputs"
    (o / "demo" / "reports").mkdir(parents=True)
    (o / "demo" / "reports" / "demo_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    (o / "demo" / "processed" / "phase6").mkdir(parents=True)
    (o / "demo" / "processed" / "phase6" / "phase6_report.json").write_text("{}", encoding="utf-8")
    (o / "demo" / "rel.npy").write_bytes(b"\x93NUMPY")
    (o / "demo" / "reports" / "notes.json").write_text("{}", encoding="utf-8")
    (o.parent / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    return o


@pytest.fixture
def client(outputs: Path, dist: Path) -> TestClient:
    return TestClient(create_production_app(outputs, dist), raise_server_exceptions=False)


def test_root_serves_index_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == INDEX_HTML


@pytest.mark.parametrize(("name", "ctype"), [("index-abc.js", "text/javascript"), ("index-abc.css", "text/css"),
                                             ("draco.wasm", "application/wasm")])
def test_static_assets(client: TestClient, dist: Path, name: str, ctype: str) -> None:
    r = client.get(f"/assets/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(ctype)
    assert r.content == (dist / "assets" / name).read_bytes()


def test_missing_asset_is_404(client: TestClient) -> None:
    assert client.get("/assets/nope.js").status_code == 404


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "project": "DepthWizard", "phase": 8}


def test_outputs_demo_report(client: TestClient) -> None:
    r = client.get("/outputs/demo/reports/demo_report.json")
    assert r.status_code == 200
    assert r.json() == REPORT


def test_viewer_data_route(client: TestClient) -> None:
    r = client.get("/data/outputs/demo/reports/demo_report.json")
    assert r.status_code == 200
    assert r.json() == REPORT
    assert client.get("/data/outputs/demo/missing.json").status_code == 404


def test_data_index_matches_dev_route_shape(client: TestClient) -> None:
    r = client.get("/__data_index")
    assert r.status_code == 200
    assert r.json() == {
        "root": "data/outputs",
        "reports": ["data/outputs/demo/processed/phase6/phase6_report.json",
                    "data/outputs/demo/reports/demo_report.json"],
        "npy": ["data/outputs/demo/rel.npy"],
    }


def test_existing_api_routes_not_shadowed(client: TestClient) -> None:
    # POST routes still reach the API (validation error, not a static 404/405).
    r = client.post("/process")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"
    assert any(getattr(route, "path", None) == "/demo/synthetic" for route in client.app.routes)


@pytest.mark.parametrize("prefix", ["/outputs", "/data/outputs"])
@pytest.mark.parametrize("path", ["../secret.txt", "..%2Fsecret.txt", "demo/..%2F..%2Fsecret.txt",
                                  "..%5Csecret.txt", "%2E%2E/secret.txt"])
def test_output_path_traversal_blocked(client: TestClient, prefix: str, path: str) -> None:
    r = client.get(f"{prefix}/{path}")
    assert r.status_code in (403, 404)
    assert "TOP-SECRET" not in r.text


@pytest.mark.parametrize("path", ["/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt", "/%2E%2E/%2E%2E/secret.txt",
                                  "/assets/..%5C..%5Csecret.txt"])
def test_static_path_traversal_blocked(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code in (403, 404)
    assert "TOP-SECRET" not in r.text


def test_missing_dist_is_refused(outputs: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="npm run build"):
        create_production_app(outputs, tmp_path / "no-dist")


def test_build_index_empty_root(tmp_path: Path) -> None:
    assert build_index(tmp_path / "missing") == {"root": "data/outputs", "reports": [], "npy": []}


REAL_REPORT = REPO_ROOT / "data" / "outputs" / "demo" / "reports" / "demo_report.json"


@pytest.mark.skipif(not (DIST_DIR / "index.html").is_file() or not REAL_REPORT.is_file(),
                    reason="needs `npm run build --prefix viewer` and `python -m depthwizard.demo synthetic`")
def test_real_build_and_demo() -> None:
    client = TestClient(create_production_app(REPO_ROOT / "data" / "outputs", DIST_DIR))
    r = client.get("/")
    assert r.status_code == 200 and '<div id="root">' in r.text
    for asset in sorted((DIST_DIR / "assets").iterdir()):
        assert client.get(f"/assets/{asset.name}").status_code == 200
    r = client.get("/outputs/demo/reports/demo_report.json")
    assert r.status_code == 200
    assert r.json() == json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    assert "data/outputs/demo/reports/demo_report.json" in client.get("/__data_index").json()["reports"]
