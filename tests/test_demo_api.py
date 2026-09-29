"""Phase 8 API smoke tests (Step 3): /health, /process, /outputs, /demo/synthetic.

Every app here serves a temporary outputs root, so nothing is written into
data/outputs. Phase 3 needs the git-ignored checkpoint; where it is absent the
tests assert the honest "not available" record instead of a relative field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from depthwizard.demo import api as api_module  # noqa: E402
from depthwizard.demo.api import create_app  # noqa: E402
from depthwizard.demo.process import DEFAULT_CHECKPOINT, ProcessingError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_CHECKPOINT = (REPO_ROOT / DEFAULT_CHECKPOINT).is_file()
METRIC_PRODUCTS = ("shadow_heights", "calibrated_agl", "dsm", "dtm", "ndsm")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "outputs", checkpoint=REPO_ROOT / DEFAULT_CHECKPOINT),
                      raise_server_exceptions=False)


def _upload(client: TestClient, path: Path, **form: str) -> Any:
    with path.open("rb") as fh:
        return client.post("/process", files={"file": (path.name, fh, "application/octet-stream")}, data=form)


def _non_null_metre_keys(obj: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_m") and v is not None:
                found.append(prefix + k)
            found += _non_null_metre_keys(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            found += _non_null_metre_keys(v, prefix)
    return found


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "project": "DepthWizard", "phase": 8}


def test_process_absolute_single_band_fixture(client: TestClient, fixture_tif: Path) -> None:
    r = _upload(client, fixture_tif)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ABSOLUTE"
    assert body["metric_capable_input"] is True
    assert body["units"] is None  # no height product was produced
    assert body["available_products"] == []
    assert body["metadata"]["gsd_m"] == 0.5
    assert body["metadata"]["crs"] == "EPSG:32643"
    assert body["products"]["relative_height"]["status"] == "not available"
    assert "fabricate RGB" in body["products"]["relative_height"]["reason"]
    for name in METRIC_PRODUCTS:
        product = body["products"][name]
        assert product["status"] == "not available"
        assert "SUPPLIED building footprints" in product["reason"]
    assert "SYNTHETIC by design" in body["products"]["dsm"]["reason"]
    assert body["uncertainty"] == "not yet measured"


def test_process_relative_rgb_png(client: TestClient, rgb_png_path: Path) -> None:
    r = _upload(client, rgb_png_path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "RELATIVE"
    assert body["metric_capable_input"] is False
    for name in METRIC_PRODUCTS:
        assert body["products"][name]["status"] == "not available"
    assert _non_null_metre_keys(body) == []
    assert body["uncertainty"] == "not yet measured"
    rel = body["products"]["relative_height"]
    if HAS_CHECKPOINT:
        assert body["available_products"] == ["relative_height"]
        assert body["units"] == "unitless"
        assert rel["units"] == "unitless" and "NOT metres" in rel["output_units"]
        array = client.get(body["outputs"]["array"])
        assert array.status_code == 200 and array.content[:6] == b"\x93NUMPY"
    else:
        assert body["available_products"] == [] and body["units"] is None
        assert rel["status"] == "not available"
    report = client.get(body["outputs"]["report"])
    assert report.status_code == 200 and report.headers["content-type"].startswith("application/json")
    assert report.json()["mode"] == "RELATIVE"


@pytest.mark.parametrize(
    ("name", "content", "form", "status", "kind"),
    [
        ("notes.txt", b"hello", {}, 415, "unsupported_format"),
        ("broken.tif", b"II*\x00garbage-not-a-tiff", {}, 422, "corrupt_image"),
        ("empty.png", b"", {}, 422, "corrupt_image"),
        ("x.png", b"\x89PNG\r\n\x1a\n", {"require": "bogus"}, 400, "invalid_mode"),
    ],
)
def test_process_rejects_bad_input(client: TestClient, tmp_path: Path, name: str, content: bytes,
                                   form: dict[str, str], status: int, kind: str) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    r = _upload(client, path, **form)
    assert r.status_code == status, r.text
    assert r.json()["error"] == kind
    assert "Traceback" not in r.text


def test_process_require_absolute_refuses_plain_image(client: TestClient, png_path: Path) -> None:
    r = _upload(client, png_path, require="absolute")
    assert r.status_code == 422
    assert r.json()["error"] == "missing_metadata"


def test_process_failure_is_clean(client: TestClient, fixture_tif: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ProcessingError("Phase 3 relative-height inference failed: secret internal detail")

    monkeypatch.setattr(api_module, "process_image", boom)
    r = _upload(client, fixture_tif)
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "processing_failed" and "run_id" in body
    assert "secret internal detail" not in r.text and "Traceback" not in r.text


def test_outputs_serves_files_and_refuses_escapes(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "a.json").write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    client = TestClient(create_app(root), raise_server_exceptions=False)

    ok = client.get("/outputs/demo/a.json")
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("application/json")
    assert client.get("/outputs/demo/missing.json").status_code == 404
    for bad in ("/outputs/../secret.txt", "/outputs/%2e%2e/secret.txt", "/outputs/demo/..%2F..%2Fsecret.txt",
                "/outputs/..%5Csecret.txt", f"/outputs/{tmp_path.as_posix()}/secret.txt"):
        r = client.get(bad)
        assert r.status_code in (403, 404), bad
        assert "secret" not in r.text.replace("secret.txt", ""), bad


def test_demo_synthetic_runs_the_existing_chain(tmp_path: Path) -> None:
    """The deterministic chain (Phase 4b -> 1 -> 2 -> 5 -> 6); the mesh needs repo-relative paths, so it is off."""
    client = TestClient(create_app(tmp_path / "outputs"), raise_server_exceptions=False)
    r = client.post("/demo/synthetic", params={"mesh": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ABSOLUTE" and body["units"] == "metres"
    assert [s["phase"].split()[0] for s in body["stages"]] == ["4b", "1", "2", "5", "6"]
    assert "not a Phase 3 prediction" in body["label"]
    assert "log(AGL_true + 1) + c" in body["phase4b_dsm"]["relative_field"]
    p2 = body["phase2_shadow_physics"]
    assert "SUPPLIED" in p2["footprints"]
    assert p2["mae_m"] == pytest.approx(0.2301, abs=5e-5)
    assert p2["max_abs_error_m"] == pytest.approx(0.3431, abs=5e-5)
    assert p2["max_abs_error_m"] <= p2["pixel_grid_bound_m"]
    assert body["phase6_surfaces"]["acceptance_passed"] is True
    assert body["uncertainty"] == "not yet measured"
    assert all(v == "not yet measured" for v in body["not_yet_measured"].values())
    assert body["mesh"]["status"] == "not available"
    manifest = client.get(body["report"])
    assert manifest.status_code == 200 and manifest.json()["phase"] == "8"
